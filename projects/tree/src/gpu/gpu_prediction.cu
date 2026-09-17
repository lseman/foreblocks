#include "foretree/gpu/gpu_prediction.hpp"
#include "foretree/tree/packed_tree.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace foretree::cuda {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <class T> T* allocate(size_t count) {
    T* pointer = nullptr;
    check(cudaMalloc(&pointer, count * sizeof(T)), "cudaMalloc");
    return pointer;
}

// ---- Device MLP helper ----

__device__ float apply_activation(float x, uint8_t type) {
    switch (type) {
        case 1: return fmaxf(x, 0.0f);
        case 2: return 1.0f / (1.0f + expf(-x));
        case 3: return tanhf(x);
        default: return x;
    }
}

__device__ void mlp_apply_one_layer(const float* all_weights, const float* all_biases,
                                     const int* layer_sizes, uint8_t activation,
                                     const float* in_buf, float* out_buf, int layer_idx) {
    int in_dim = layer_sizes[layer_idx];
    int out_dim = layer_sizes[layer_idx + 1];
    size_t w_offset = 0;
    for (int l = 0; l < layer_idx; ++l)
        w_offset += static_cast<size_t>(layer_sizes[l]) * static_cast<size_t>(layer_sizes[l + 1]);
    size_t b_offset = 0;
    for (int l = 0; l < layer_idx; ++l)
        b_offset += static_cast<size_t>(layer_sizes[l + 1]);
    for (int j = 0; j < out_dim; ++j) {
        float sum = all_biases[b_offset + j];
        const float* wrow = all_weights + w_offset + static_cast<size_t>(j) * static_cast<size_t>(in_dim);
        for (int k = 0; k < in_dim; ++k)
            sum += wrow[k] * in_buf[k];
        out_buf[j] = apply_activation(sum, activation);
    }
}

__device__ inline int device_min(int a, int b) { return a < b ? a : b; }

// ---- Device tree walking helpers ----

__device__ inline bool is_leaf(const uint8_t* leaf_flags, int node) {
    return leaf_flags[node] != 0;
}

__device__ inline bool categorical_test(const int* offsets, const int* counts,
                                         const int* bins, int node, uint8_t bin_code) {
    int off = offsets[node];
    int cnt = counts[node];
    for (int i = 0; i < cnt; ++i)
        if (bins[off + i] == static_cast<int>(bin_code))
            return true;
    return false;
}

__device__ inline bool oblique_test(const int* oblique_offsets, const int* oblique_counts,
                                     const int* oblique_features, const double* oblique_weights,
                                     const double* oblique_thresholds, int node,
                                     const double* raw_data, int num_features) {
    int off = oblique_offsets[node];
    int cnt = oblique_counts[node];
    double dot = 0.0;
    for (int i = 0; i < cnt; ++i) {
        int feat = oblique_features[off + i];
        if (feat >= 0 && feat < num_features)
            dot += oblique_weights[off + i] * raw_data[feat];
    }
    return dot <= oblique_thresholds[node];
}

__device__ inline bool pair_test(const int* pair_features_a, const int* pair_features_b,
                                  const int* pair_thresholds_a, const int* pair_thresholds_b,
                                  const uint8_t* pair_quadrant_masks, int node,
                                  const double* raw_data, int num_features) {
    int feat_a = pair_features_a[node];
    int feat_b = pair_features_b[node];
    int thresh_a = pair_thresholds_a[node];
    int thresh_b = pair_thresholds_b[node];
    uint8_t mask = pair_quadrant_masks[node];
    double val_a = (feat_a >= 0 && feat_a < num_features) ? raw_data[feat_a] : 0.0;
    double val_b = (feat_b >= 0 && feat_b < num_features) ? raw_data[feat_b] : 0.0;
    int quadrant = (val_a <= thresh_a ? 1 : 0) | (val_b <= thresh_b ? 2 : 0);
    return (mask & (1 << quadrant)) != 0;
}

// ---- Prediction kernel (binned codes) ----

template <class CodeType>
__global__ void prediction_kernel(
    const int* features,
    const double* thresholds,
    const uint8_t* split_kinds,
    const uint8_t* missing_left,
    const int* left_children,
    const int* right_children,
    const uint8_t* leaf_flags,
    const double* leaf_values,
    const int* categorical_offsets,
    const int* categorical_counts,
    const int* categorical_bins,
    const int* oblique_offsets,
    const int* oblique_counts,
    const int* oblique_features,
    const double* oblique_weights,
    const double* oblique_thresholds,
    const int* pair_features_a,
    const int* pair_features_b,
    const int* pair_thresholds_a,
    const int* pair_thresholds_b,
    const uint8_t* pair_quadrant_masks,
    const CodeType* codes,
    int num_features,
    int num_trees,
    int num_outputs,
    int root_id,
    double* output,
    int* debug_leaf_idx) {
    int sample = blockIdx.x;
    if (sample < 0 || sample >= gridDim.x)
        return;

    const CodeType* sample_codes = codes + static_cast<size_t>(sample) * static_cast<size_t>(num_features);
    double partials[16] = {0.0};
    int max_outputs = device_min(num_outputs, 16);

    // For single-tree forest, walk from root; for multi-tree, tree t starts at node t
    int tree_root = root_id >= 0 ? root_id : 0;
    for (int t = 0; t < num_trees; ++t) {
        int node = (num_trees == 1) ? tree_root : t;
        if (node < 0) continue;
        int leaf_idx = 0;

        for (;;) {
            if (is_leaf(leaf_flags, node)) { leaf_idx = node; break; }

            uint8_t kind = split_kinds[node];
            int feature = features[node];
            bool go_left;

            switch (kind) {
                case 0: {
                    uint8_t bin_code = sample_codes[static_cast<size_t>(feature)];
                    go_left = static_cast<double>(bin_code) <= thresholds[node];
                    break;
                }
                case 1: {
                    uint8_t bin_code = sample_codes[feature];
                    go_left = categorical_test(categorical_offsets, categorical_counts,
                                                categorical_bins, node, bin_code);
                    break;
                }
                case 2: {
                    go_left = oblique_test(oblique_offsets, oblique_counts,
                                            oblique_features, oblique_weights,
                                            oblique_thresholds, node, nullptr, num_features);
                    break;
                }
                case 3: {
                    go_left = pair_test(pair_features_a, pair_features_b,
                                         pair_thresholds_a, pair_thresholds_b,
                                         pair_quadrant_masks, node, nullptr, num_features);
                    break;
                }
                default: go_left = false; break;
            }
            node = go_left ? left_children[node] : right_children[node];
        }

        // Debug: store leaf_idx for host inspection
        if (debug_leaf_idx)
            debug_leaf_idx[sample] = leaf_idx;
        double val = leaf_values[leaf_idx];
        partials[t % max_outputs] += val;
    }

    for (int i = 0; i < max_outputs; ++i)
        output[static_cast<size_t>(sample) * static_cast<size_t>(num_outputs) + i] = partials[i];
}

// ---- Prediction kernel (raw data) ----

__global__ void prediction_raw_kernel(
    const int* features,
    const double* thresholds,
    const uint8_t* split_kinds,
    const uint8_t* missing_left,
    const int* left_children,
    const int* right_children,
    const uint8_t* leaf_flags,
    const double* leaf_values,
    const int* categorical_offsets,
    const int* categorical_counts,
    const int* categorical_bins,
    const int* oblique_offsets,
    const int* oblique_counts,
    const int* oblique_features,
    const double* oblique_weights,
    const double* oblique_thresholds,
    const int* pair_features_a,
    const int* pair_features_b,
    const int* pair_thresholds_a,
    const int* pair_thresholds_b,
    const uint8_t* pair_quadrant_masks,
    const double* raw_data,
    int num_features,
    int num_trees,
    int num_outputs,
    int root_id,
    int mlp_input_dim,
    int mlp_output_dim,
    const float* mlp_weights,
    const float* mlp_biases,
    const int* mlp_layer_sizes,
    const uint8_t* mlp_activations,
    int mlp_layer_count,
    double* output) {
    int sample = blockIdx.x;
    if (sample < 0 || sample >= gridDim.x)
        return;

    const double* sample_raw = raw_data + static_cast<size_t>(sample) * static_cast<size_t>(num_features);
    double partials[16] = {0.0};
    int max_outputs = device_min(num_outputs, 16);

    int tree_root = root_id >= 0 ? root_id : 0;
    for (int t = 0; t < num_trees; ++t) {
        int node = (num_trees == 1) ? tree_root : t;
        if (node < 0) continue;
        int leaf_idx = 0;

        for (;;) {
            if (is_leaf(leaf_flags, node)) { leaf_idx = node; break; }

            uint8_t kind = split_kinds[node];
            int feature = features[node];
            bool go_left;

            switch (kind) {
                case 0:
                    go_left = sample_raw[feature] <= thresholds[node];
                    break;
                case 1:
                    go_left = false;  // categorical not supported in raw mode
                    break;
                case 2:
                    go_left = oblique_test(oblique_offsets, oblique_counts,
                                            oblique_features, oblique_weights,
                                            oblique_thresholds, node, sample_raw, num_features);
                    break;
                case 3:
                    go_left = pair_test(pair_features_a, pair_features_b,
                                         pair_thresholds_a, pair_thresholds_b,
                                         pair_quadrant_masks, node, sample_raw, num_features);
                    break;
                default: go_left = false; break;
            }
            node = go_left ? left_children[node] : right_children[node];
        }

        partials[t % max_outputs] += leaf_values[leaf_idx];
    }

    // Neural MLP post-processing
    if (mlp_layer_count > 0 && mlp_input_dim > 0) {
        float buf_a[64], buf_b[64];
        int dim = device_min(mlp_input_dim, 64);
        for (int i = 0; i < dim; ++i)
            buf_a[i] = static_cast<float>(partials[i]);

        for (int l = 0; l < mlp_layer_count; ++l) {
            uint8_t act = mlp_activations[l];
            mlp_apply_one_layer(mlp_weights, mlp_biases, mlp_layer_sizes, act, buf_a, buf_b, l);
            int out_dim = device_min(mlp_layer_sizes[l + 1], 64);
            for (int i = 0; i < out_dim; ++i)
                buf_a[i] = buf_b[i];
        }

        int out_dim_final = device_min(mlp_layer_sizes[mlp_layer_count], 64);
        for (int i = 0; i < out_dim_final; ++i)
            partials[i] = static_cast<double>(buf_b[i]);
    }

    for (int i = 0; i < max_outputs; ++i)
        output[static_cast<size_t>(sample) * static_cast<size_t>(num_outputs) + i] = partials[i];
}

// ---- Device memory layout (pinned to device) ----

struct DeviceTreeLayout {
    int* features = nullptr;
    double* thresholds = nullptr;
    uint8_t* split_kinds = nullptr;
    uint8_t* missing_left = nullptr;
    int* left_children = nullptr;
    int* right_children = nullptr;
    uint8_t* leaf_flags = nullptr;
    double* leaf_values = nullptr;
    int* categorical_offsets = nullptr;
    int* categorical_counts = nullptr;
    int* categorical_bins = nullptr;
    int* oblique_offsets = nullptr;
    int* oblique_counts = nullptr;
    int* oblique_features = nullptr;
    double* oblique_weights = nullptr;
    double* oblique_thresholds = nullptr;
    int* pair_features_a = nullptr;
    int* pair_features_b = nullptr;
    int* pair_thresholds_a = nullptr;
    int* pair_thresholds_b = nullptr;
    uint8_t* pair_quadrant_masks = nullptr;
    float* mlp_weights = nullptr;
    float* mlp_biases = nullptr;
    int* mlp_layer_sizes = nullptr;
    uint8_t* mlp_activations = nullptr;
    size_t allocated_bytes = 0;

    ~DeviceTreeLayout() {
        cudaFree(features); cudaFree(thresholds); cudaFree(split_kinds);
        cudaFree(missing_left); cudaFree(left_children); cudaFree(right_children);
        cudaFree(leaf_flags); cudaFree(leaf_values);
        cudaFree(categorical_offsets); cudaFree(categorical_counts); cudaFree(categorical_bins);
        cudaFree(oblique_offsets); cudaFree(oblique_counts); cudaFree(oblique_features);
        cudaFree(oblique_weights); cudaFree(oblique_thresholds);
        cudaFree(pair_features_a); cudaFree(pair_features_b);
        cudaFree(pair_thresholds_a); cudaFree(pair_thresholds_b); cudaFree(pair_quadrant_masks);
        cudaFree(mlp_weights); cudaFree(mlp_biases); cudaFree(mlp_layer_sizes); cudaFree(mlp_activations);
    }
};

// ---- Impl definition (visible to all code in foretree::cuda) ----

struct GpuPredictionEngine::Impl {
    int num_features = 0;
    int num_trees = 0;
    int root_id_ = -1;
    int num_outputs = 1;
    bool has_oblique_or_pair = false;

    DeviceTreeLayout d;
    size_t device_bytes = 0;

    bool has_neural = false;
    int mlp_input_dim = 0, mlp_output_dim = 0, mlp_layer_count = 0;
    std::vector<int> mlp_layer_sizes;
    std::vector<uint8_t> mlp_activations;

    Impl() = default;

    void copy_tree_from_host(const PackedTree& tree) {
        if (tree.empty())
            throw std::invalid_argument("GpuPredictionEngine: empty tree");

        int nodes = static_cast<int>(tree.node_count());
        // num_features = max feature ID + 1
        if (!tree.features.empty()) {
            int max_feat = 0;
            for (int f : tree.features) max_feat = std::max(max_feat, f);
            num_features = max_feat + 1;
        } else {
            num_features = 0;
        }
        num_trees = 1;  // single tree - kernel walks from root
        root_id_ = tree.root >= 0 ? tree.root : 0;
        num_outputs = tree.outputs > 0 ? tree.outputs : 1;

        d.features = allocate<int>(static_cast<size_t>(nodes));
        device_bytes += static_cast<size_t>(nodes) * sizeof(int);
        check(cudaMemcpy(d.features, tree.features.data(), static_cast<size_t>(nodes) * sizeof(int), cudaMemcpyHostToDevice), "copy features");

        d.thresholds = allocate<double>(static_cast<size_t>(nodes));
        device_bytes += static_cast<size_t>(nodes) * sizeof(double);
        // PackedTree::thresholds is std::vector<int>; convert to double for device
        std::vector<double> h_thresholds_d(static_cast<size_t>(nodes));
        for (int i = 0; i < nodes; ++i)
            h_thresholds_d[static_cast<size_t>(i)] = static_cast<double>(tree.thresholds[static_cast<size_t>(i)]);
        check(cudaMemcpy(d.thresholds, h_thresholds_d.data(), static_cast<size_t>(nodes) * sizeof(double), cudaMemcpyHostToDevice), "copy thresholds");

        d.split_kinds = allocate<uint8_t>(static_cast<size_t>(nodes));
        device_bytes += static_cast<size_t>(nodes) * sizeof(uint8_t);
        check(cudaMemcpy(d.split_kinds, tree.split_kinds.data(), static_cast<size_t>(nodes) * sizeof(uint8_t), cudaMemcpyHostToDevice), "copy split_kinds");

        d.missing_left = allocate<uint8_t>(static_cast<size_t>(nodes));
        device_bytes += static_cast<size_t>(nodes) * sizeof(uint8_t);
        check(cudaMemcpy(d.missing_left, tree.missing_left.data(), static_cast<size_t>(nodes) * sizeof(uint8_t), cudaMemcpyHostToDevice), "copy missing_left");

        d.left_children = allocate<int>(static_cast<size_t>(nodes));
        device_bytes += static_cast<size_t>(nodes) * sizeof(int);
        check(cudaMemcpy(d.left_children, tree.left_children.data(), static_cast<size_t>(nodes) * sizeof(int), cudaMemcpyHostToDevice), "copy left_children");

        d.right_children = allocate<int>(static_cast<size_t>(nodes));
        device_bytes += static_cast<size_t>(nodes) * sizeof(int);
        check(cudaMemcpy(d.right_children, tree.right_children.data(), static_cast<size_t>(nodes) * sizeof(int), cudaMemcpyHostToDevice), "copy right_children");

        d.leaf_flags = allocate<uint8_t>(static_cast<size_t>(nodes));
        device_bytes += static_cast<size_t>(nodes) * sizeof(uint8_t);
        check(cudaMemcpy(d.leaf_flags, tree.leaf_flags.data(), static_cast<size_t>(nodes) * sizeof(uint8_t), cudaMemcpyHostToDevice), "copy leaf_flags");

        d.leaf_values = allocate<double>(static_cast<size_t>(nodes));
        device_bytes += static_cast<size_t>(nodes) * sizeof(double);
        check(cudaMemcpy(d.leaf_values, tree.leaf_values.data(), static_cast<size_t>(nodes) * sizeof(double), cudaMemcpyHostToDevice), "copy leaf_values");

        if (!tree.categorical_offsets.empty()) {
            d.categorical_offsets = allocate<int>(static_cast<int>(tree.categorical_offsets.size()));
            device_bytes += tree.categorical_offsets.size() * sizeof(int);
            check(cudaMemcpy(d.categorical_offsets, tree.categorical_offsets.data(), tree.categorical_offsets.size() * sizeof(int), cudaMemcpyHostToDevice), "copy cat_offsets");
            d.categorical_counts = allocate<int>(static_cast<int>(tree.categorical_counts.size()));
            device_bytes += tree.categorical_counts.size() * sizeof(int);
            check(cudaMemcpy(d.categorical_counts, tree.categorical_counts.data(), tree.categorical_counts.size() * sizeof(int), cudaMemcpyHostToDevice), "copy cat_counts");
            d.categorical_bins = allocate<int>(static_cast<int>(tree.categorical_bins.size()));
            device_bytes += tree.categorical_bins.size() * sizeof(int);
            check(cudaMemcpy(d.categorical_bins, tree.categorical_bins.data(), tree.categorical_bins.size() * sizeof(int), cudaMemcpyHostToDevice), "copy cat_bins");
        }

        for (int i = 0; i < nodes; ++i) {
            uint8_t kind = i < static_cast<int>(tree.split_kinds.size()) ? tree.split_kinds[i] : 0;
            if (kind == 2 || kind == 3) { has_oblique_or_pair = true; break; }
        }

        if (!tree.oblique_offsets.empty()) {
            d.oblique_offsets = allocate<int>(static_cast<int>(tree.oblique_offsets.size()));
            device_bytes += tree.oblique_offsets.size() * sizeof(int);
            check(cudaMemcpy(d.oblique_offsets, tree.oblique_offsets.data(), tree.oblique_offsets.size() * sizeof(int), cudaMemcpyHostToDevice), "copy oblique_offsets");
            d.oblique_counts = allocate<int>(static_cast<int>(tree.oblique_counts.size()));
            device_bytes += tree.oblique_counts.size() * sizeof(int);
            check(cudaMemcpy(d.oblique_counts, tree.oblique_counts.data(), tree.oblique_counts.size() * sizeof(int), cudaMemcpyHostToDevice), "copy oblique_counts");
            d.oblique_features = allocate<int>(static_cast<int>(tree.oblique_features.size()));
            device_bytes += tree.oblique_features.size() * sizeof(int);
            check(cudaMemcpy(d.oblique_features, tree.oblique_features.data(), tree.oblique_features.size() * sizeof(int), cudaMemcpyHostToDevice), "copy oblique_features");
            d.oblique_weights = allocate<double>(static_cast<int>(tree.oblique_weights.size()));
            device_bytes += tree.oblique_weights.size() * sizeof(double);
            check(cudaMemcpy(d.oblique_weights, tree.oblique_weights.data(), tree.oblique_weights.size() * sizeof(double), cudaMemcpyHostToDevice), "copy oblique_weights");
            d.oblique_thresholds = allocate<double>(static_cast<int>(tree.oblique_thresholds.size()));
            device_bytes += tree.oblique_thresholds.size() * sizeof(double);
            check(cudaMemcpy(d.oblique_thresholds, tree.oblique_thresholds.data(), tree.oblique_thresholds.size() * sizeof(double), cudaMemcpyHostToDevice), "copy oblique_thresholds");
        }

        if (!tree.pair_features_a.empty()) {
            d.pair_features_a = allocate<int>(static_cast<int>(tree.pair_features_a.size()));
            device_bytes += tree.pair_features_a.size() * sizeof(int);
            check(cudaMemcpy(d.pair_features_a, tree.pair_features_a.data(), tree.pair_features_a.size() * sizeof(int), cudaMemcpyHostToDevice), "copy pair_a");
            d.pair_features_b = allocate<int>(static_cast<int>(tree.pair_features_b.size()));
            device_bytes += tree.pair_features_b.size() * sizeof(int);
            check(cudaMemcpy(d.pair_features_b, tree.pair_features_b.data(), tree.pair_features_b.size() * sizeof(int), cudaMemcpyHostToDevice), "copy pair_b");
            d.pair_thresholds_a = allocate<int>(static_cast<int>(tree.pair_thresholds_a.size()));
            device_bytes += tree.pair_thresholds_a.size() * sizeof(int);
            check(cudaMemcpy(d.pair_thresholds_a, tree.pair_thresholds_a.data(), tree.pair_thresholds_a.size() * sizeof(int), cudaMemcpyHostToDevice), "copy pair_thresh_a");
            d.pair_thresholds_b = allocate<int>(static_cast<int>(tree.pair_thresholds_b.size()));
            device_bytes += tree.pair_thresholds_b.size() * sizeof(int);
            check(cudaMemcpy(d.pair_thresholds_b, tree.pair_thresholds_b.data(), tree.pair_thresholds_b.size() * sizeof(int), cudaMemcpyHostToDevice), "copy pair_thresh_b");
            d.pair_quadrant_masks = allocate<uint8_t>(static_cast<int>(tree.pair_quadrant_masks.size()));
            device_bytes += tree.pair_quadrant_masks.size() * sizeof(uint8_t);
            check(cudaMemcpy(d.pair_quadrant_masks, tree.pair_quadrant_masks.data(), tree.pair_quadrant_masks.size() * sizeof(uint8_t), cudaMemcpyHostToDevice), "copy pair_masks");
        }
    }

    void configure_neural(const NeuralLeafConfig& config) {
        has_neural = true;
        mlp_layer_count = static_cast<int>(config.layer_count);
        mlp_layer_sizes = config.layer_sizes;
        mlp_activations = config.activations;
        mlp_input_dim = mlp_layer_sizes.empty() ? 0 : mlp_layer_sizes.front();
        mlp_output_dim = mlp_layer_sizes.empty() ? 0 : mlp_layer_sizes.back();

        if (!config.weights.empty()) {
            d.mlp_weights = allocate<float>(config.weights.size());
            device_bytes += config.weights.size() * sizeof(float);
            check(cudaMemcpy(d.mlp_weights, config.weights.data(), config.weights.size() * sizeof(float), cudaMemcpyHostToDevice), "copy mlp_weights");
        }
        if (!config.biases.empty()) {
            d.mlp_biases = allocate<float>(config.biases.size());
            device_bytes += config.biases.size() * sizeof(float);
            check(cudaMemcpy(d.mlp_biases, config.biases.data(), config.biases.size() * sizeof(float), cudaMemcpyHostToDevice), "copy mlp_biases");
        }
        if (!mlp_layer_sizes.empty()) {
            d.mlp_layer_sizes = allocate<int>(static_cast<int>(mlp_layer_sizes.size()));
            device_bytes += mlp_layer_sizes.size() * sizeof(int);
            check(cudaMemcpy(d.mlp_layer_sizes, mlp_layer_sizes.data(), mlp_layer_sizes.size() * sizeof(int), cudaMemcpyHostToDevice), "copy mlp_layer_sizes");
        }
        if (!mlp_activations.empty()) {
            d.mlp_activations = allocate<uint8_t>(mlp_activations.size());
            device_bytes += mlp_activations.size() * sizeof(uint8_t);
            check(cudaMemcpy(d.mlp_activations, mlp_activations.data(), mlp_activations.size() * sizeof(uint8_t), cudaMemcpyHostToDevice), "copy mlp_activations");
        }
    }
};

// ---- Constructor / Destructor ----

GpuPredictionEngine::GpuPredictionEngine(const PackedTree& tree, const NeuralLeafConfig* neural_config)
    : impl_(std::make_unique<Impl>()) {
    impl_->copy_tree_from_host(tree);
    if (neural_config) impl_->configure_neural(*neural_config);
}

GpuPredictionEngine::~GpuPredictionEngine() = default;
GpuPredictionEngine::GpuPredictionEngine(GpuPredictionEngine&&) noexcept = default;
GpuPredictionEngine& GpuPredictionEngine::operator=(GpuPredictionEngine&&) noexcept = default;

// Construct from raw arrays (used by Python binding).
GpuPredictionEngine::GpuPredictionEngine(
    const std::vector<std::span<const int>>& int_arrays,
    const std::vector<std::span<const double>>& double_arrays,
    const std::vector<std::span<const uint8_t>>& uint8_arrays,
    int num_features, int outputs)
    : impl_(std::make_unique<Impl>()) {
    // Build a temporary PackedTree from the arrays
    PackedTree tree;
    tree.root = 0;
    tree.outputs = outputs;
    // int_arrays[0]=features, [1]=thresholds, [5]=left_children, [6]=right_children,
    // [9]=cat_offsets, [10]=cat_counts, [11]=cat_bins,
    // [12]=pair_a, [13]=pair_b, [14]=pair_thresh_a, [15]=pair_thresh_b,
    // [17]=oblique_offsets, [18]=oblique_counts, [19]=oblique_features
    // double_arrays[0]=split_values, [1]=cover, [2]=oblique_weights, [3]=oblique_thresholds, [4]=leaf_values
    // uint8_arrays[0]=split_kinds, [1]=missing_left, [2]=leaf_flags, [3]=pair_quadrant_masks
    tree.features.assign(int_arrays[0].begin(), int_arrays[0].end());
    tree.thresholds.assign(int_arrays[1].begin(), int_arrays[1].end());
    tree.split_values.assign(double_arrays[0].begin(), double_arrays[0].end());
    tree.split_kinds.assign(uint8_arrays[0].begin(), uint8_arrays[0].end());
    tree.missing_left.assign(uint8_arrays[1].begin(), uint8_arrays[1].end());
    tree.left_children.assign(int_arrays[5].begin(), int_arrays[5].end());
    tree.right_children.assign(int_arrays[6].begin(), int_arrays[6].end());
    tree.leaf_flags.assign(uint8_arrays[2].begin(), uint8_arrays[2].end());
    tree.cover.assign(double_arrays[1].begin(), double_arrays[1].end());
    tree.categorical_offsets.assign(int_arrays[9].begin(), int_arrays[9].end());
    tree.categorical_counts.assign(int_arrays[10].begin(), int_arrays[10].end());
    tree.categorical_bins.assign(int_arrays[11].begin(), int_arrays[11].end());
    tree.pair_features_a.assign(int_arrays[12].begin(), int_arrays[12].end());
    tree.pair_features_b.assign(int_arrays[13].begin(), int_arrays[13].end());
    tree.pair_thresholds_a.assign(int_arrays[14].begin(), int_arrays[14].end());
    tree.pair_thresholds_b.assign(int_arrays[15].begin(), int_arrays[15].end());
    tree.pair_quadrant_masks.assign(uint8_arrays[3].begin(), uint8_arrays[3].end());
    tree.oblique_offsets.assign(int_arrays[17].begin(), int_arrays[17].end());
    tree.oblique_counts.assign(int_arrays[18].begin(), int_arrays[18].end());
    tree.oblique_features.assign(int_arrays[19].begin(), int_arrays[19].end());
    tree.oblique_weights.assign(double_arrays[2].begin(), double_arrays[2].end());
    tree.oblique_thresholds.assign(double_arrays[3].begin(), double_arrays[3].end());
    tree.leaf_values.assign(double_arrays[4].begin(), double_arrays[4].end());

    impl_->copy_tree_from_host(tree);
}

// ---- Prediction dispatch ----

template <class CodeType>
static std::vector<double> run_predict(const GpuPredictionEngine::Impl& impl, int num_samples, std::span<const CodeType> codes) {
    // Derive num_features from codes array shape (N, P) => P = codes.size() / N
    int num_features = static_cast<int>(codes.size()) / num_samples;
    if (static_cast<size_t>(num_samples * num_features) != codes.size())
        throw std::invalid_argument("GpuPredictionEngine: code size mismatch");

    int outputs = impl.num_outputs;
    size_t output_size = static_cast<size_t>(num_samples) * static_cast<size_t>(outputs);
    double* device_output = allocate<double>(output_size);
    CodeType* device_codes = allocate<CodeType>(codes.size());
    check(cudaMemcpy(device_codes, codes.data(), codes.size() * sizeof(CodeType), cudaMemcpyHostToDevice), "copy codes");

    int blocks = num_samples;
    int* debug_leaf = nullptr;
    if (num_samples <= 1024) {
        debug_leaf = allocate<int>(num_samples);
        check(cudaMemset(debug_leaf, -1, num_samples * sizeof(int)), "memset debug");
    }

    if constexpr (std::is_same_v<CodeType, uint8_t>) {
        prediction_kernel<uint8_t><<<blocks, 1>>>(
            impl.d.features, impl.d.thresholds, impl.d.split_kinds, impl.d.missing_left,
            impl.d.left_children, impl.d.right_children, impl.d.leaf_flags, impl.d.leaf_values,
            impl.d.categorical_offsets, impl.d.categorical_counts, impl.d.categorical_bins,
            impl.d.oblique_offsets, impl.d.oblique_counts, impl.d.oblique_features,
            impl.d.oblique_weights, impl.d.oblique_thresholds,
            impl.d.pair_features_a, impl.d.pair_features_b, impl.d.pair_thresholds_a,
            impl.d.pair_thresholds_b, impl.d.pair_quadrant_masks,
            device_codes, num_features, impl.num_trees, outputs, impl.root_id_, device_output, debug_leaf);
    } else {
        prediction_kernel<uint16_t><<<blocks, 1>>>(
            impl.d.features, impl.d.thresholds, impl.d.split_kinds, impl.d.missing_left,
            impl.d.left_children, impl.d.right_children, impl.d.leaf_flags, impl.d.leaf_values,
            impl.d.categorical_offsets, impl.d.categorical_counts, impl.d.categorical_bins,
            impl.d.oblique_offsets, impl.d.oblique_counts, impl.d.oblique_features,
            impl.d.oblique_weights, impl.d.oblique_thresholds,
            impl.d.pair_features_a, impl.d.pair_features_b, impl.d.pair_thresholds_a,
            impl.d.pair_thresholds_b, impl.d.pair_quadrant_masks,
            device_codes, num_features, impl.num_trees, outputs, impl.root_id_, device_output, debug_leaf);
    }

    check(cudaGetLastError(), "prediction kernel");
    check(cudaDeviceSynchronize(), "prediction synchronize");

    std::vector<double> result(output_size);
    check(cudaMemcpy(result.data(), device_output, output_size * sizeof(double), cudaMemcpyDeviceToHost), "copy result");

    if (debug_leaf) {
        check(cudaFree(debug_leaf), "free debug");
    }

    cudaFree(device_codes);
    cudaFree(device_output);
    return result;
}

static std::vector<double> run_predict_raw(const GpuPredictionEngine::Impl& impl, int num_samples, std::span<const double> raw_data) {
    // Derive num_features from raw_data array shape (N, P) => P = raw_data.size() / N
    int num_features = static_cast<int>(raw_data.size()) / num_samples;
    if (static_cast<size_t>(num_samples * num_features) != raw_data.size())
        throw std::invalid_argument("GpuPredictionEngine: raw data size mismatch");

    int outputs = impl.num_outputs;
    size_t output_size = static_cast<size_t>(num_samples) * static_cast<size_t>(outputs);
    double* device_output = allocate<double>(output_size);
    double* device_raw = allocate<double>(raw_data.size());
    check(cudaMemcpy(device_raw, raw_data.data(), raw_data.size() * sizeof(double), cudaMemcpyHostToDevice), "copy raw");

    int blocks = num_samples;
    prediction_raw_kernel<<<blocks, 1>>>(
        impl.d.features, impl.d.thresholds, impl.d.split_kinds, impl.d.missing_left,
        impl.d.left_children, impl.d.right_children, impl.d.leaf_flags, impl.d.leaf_values,
        impl.d.categorical_offsets, impl.d.categorical_counts, impl.d.categorical_bins,
        impl.d.oblique_offsets, impl.d.oblique_counts, impl.d.oblique_features,
        impl.d.oblique_weights, impl.d.oblique_thresholds,
        impl.d.pair_features_a, impl.d.pair_features_b, impl.d.pair_thresholds_a,
        impl.d.pair_thresholds_b, impl.d.pair_quadrant_masks,
        device_raw, num_features, impl.num_trees, outputs, impl.root_id_,
        impl.mlp_input_dim, impl.mlp_output_dim,
        impl.d.mlp_weights, impl.d.mlp_biases, impl.d.mlp_layer_sizes,
        impl.d.mlp_activations, impl.mlp_layer_count, device_output);

    check(cudaGetLastError(), "raw kernel");
    check(cudaDeviceSynchronize(), "raw synchronize");

    std::vector<double> result(output_size);
    check(cudaMemcpy(result.data(), device_output, output_size * sizeof(double), cudaMemcpyDeviceToHost), "copy result");
    cudaFree(device_raw);
    cudaFree(device_output);
    return result;
}

std::vector<double> GpuPredictionEngine::predict_binned_uint8(std::span<const uint8_t> codes) const {
    if (impl_->has_oblique_or_pair)
        throw std::invalid_argument("GpuPredictionEngine::predict_binned_uint8: oblique/pair splits require predict_raw");
    int ns = static_cast<int>(codes.size()) / impl_->num_features;
    return run_predict(*impl_, ns, codes);
}

std::vector<double> GpuPredictionEngine::predict_binned_uint16(std::span<const uint16_t> codes) const {
    if (impl_->has_oblique_or_pair)
        throw std::invalid_argument("GpuPredictionEngine::predict_binned_uint16: oblique/pair splits require predict_raw");
    int ns = static_cast<int>(codes.size()) / impl_->num_features;
    return run_predict(*impl_, ns, codes);
}

std::vector<double> GpuPredictionEngine::predict_binned_uint16(std::span<const uint16_t> codes, int num_samples) const {
    if (impl_->has_oblique_or_pair)
        throw std::invalid_argument("GpuPredictionEngine::predict_binned_uint16: oblique/pair splits require predict_raw");
    return run_predict(*impl_, num_samples, codes);
}

std::vector<double> GpuPredictionEngine::predict_raw(std::span<const double> raw_data, std::span<const uint8_t>) const {
    int ns = static_cast<int>(raw_data.size()) / impl_->num_features;
    return run_predict_raw(*impl_, ns, raw_data);
}

std::vector<double> GpuPredictionEngine::predict_raw_uint16(std::span<const double> raw_data, std::span<const uint16_t>) const {
    int ns = static_cast<int>(raw_data.size()) / impl_->num_features;
    return run_predict_raw(*impl_, ns, raw_data);
}

int GpuPredictionEngine::outputs() const noexcept { return impl_->num_outputs; }
int GpuPredictionEngine::trees() const noexcept { return impl_->num_trees; }
size_t GpuPredictionEngine::device_bytes() const noexcept { return impl_->device_bytes; }

} // namespace foretree::cuda
