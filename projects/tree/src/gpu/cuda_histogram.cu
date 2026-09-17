#include "foretree/gpu/cuda_histogram.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace foretree::cuda {
namespace {

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

// Optimized histogram kernel: thread-per-feature with shared-memory reduction.
// Each block handles one feature. 256 threads accumulate per-row into a shared
// block-level histogram, then scatter once to global memory.
// Reduces O(N) global atomics per feature down to O(num_bins) atomics.
template <class Code>
__global__ void histogram_kernel(const Code* codes, int features, const float* gradients, const float* hessians,
                                 const uint32_t* rows, int active_rows, const uint32_t* offsets,
                                 const uint16_t* bin_counts, float* histogram_gradients, float* histogram_hessians,
                                 uint32_t* histogram_counts) {
    const int feature = blockIdx.x;
    if (feature >= features)
        return;

    const uint32_t offset = offsets[feature];
    const uint16_t bins = bin_counts[feature];
    const uint16_t finite_bins = bins > 0 ? static_cast<uint16_t>(bins - 1) : 0;

    // Shared block-level histogram: [3 * finite_bins]
    // [0, f_bins)       = gradient  accumulation
    // [f_bins, 2*f_bins) = hessian  accumulation
    // [2*f_bins, 3*f_bins) = count  accumulation (stored as float)
    extern __shared__ float s_hist[];
    const size_t s_bins = static_cast<size_t>(finite_bins);

    // Zero shared histogram
    for (unsigned i = threadIdx.x; i < 3u * s_bins; i += blockDim.x)
        s_hist[i] = 0.0F;
    __syncthreads();

    // Accumulate per-row into shared histogram
    for (int row = threadIdx.x; row < active_rows; row += blockDim.x) {
        const uint32_t global_row = rows ? rows[row] : static_cast<uint32_t>(row);
        const uint16_t code = static_cast<uint16_t>(
            codes[static_cast<size_t>(global_row) * static_cast<size_t>(features) + static_cast<size_t>(feature)]);
        const uint16_t bin = min(code, static_cast<uint16_t>(finite_bins));
        const size_t b = static_cast<size_t>(bin);
        atomicAdd(s_hist + b, gradients[global_row]);
        atomicAdd(s_hist + s_bins + b, hessians[global_row]);
        atomicAdd(s_hist + 2u * s_bins + b, 1.0F);
    }
    __syncthreads();

    // Scatter shared histogram to global memory (one write per bin)
    for (unsigned bin = threadIdx.x; bin < s_bins; bin += blockDim.x) {
        const uint32_t g_idx = offset + bin;
        histogram_gradients[g_idx] = s_hist[bin];
        histogram_hessians[g_idx] = s_hist[s_bins + bin];
        histogram_counts[g_idx] = static_cast<uint32_t>(s_hist[2u * s_bins + bin]);
    }

    // Handle the missing bin at offset + finite_bins
    // (always present — it's the last bin, may be the only one when finite_bins == 0)
    if (threadIdx.x == 0) {
        float mg = 0.0F, mh = 0.0F;
        unsigned mc = 0;
        for (int row = 0; row < active_rows; ++row) {
            const uint32_t global_row = rows ? rows[row] : static_cast<uint32_t>(row);
            const uint16_t code = static_cast<uint16_t>(
                codes[static_cast<size_t>(global_row) * static_cast<size_t>(features) + static_cast<size_t>(feature)]);
            if (code >= finite_bins) {
                mg += gradients[global_row];
                mh += hessians[global_row];
                mc++;
            }
        }
        histogram_gradients[offset + finite_bins] = mg;
        histogram_hessians[offset + finite_bins] = mh;
        histogram_counts[offset + finite_bins] = mc;
    }
}

// Optimized joint histogram kernel: thread-per-pair with shared-memory reduction.
// Each block handles one feature pair. 512 threads accumulate into a shared histogram
// with separate slots for gradient, hessian, and count per cell.
template <class Code>
__global__ void joint_histogram_kernel(const Code* codes, int features, const float* gradients, const float* hessians,
                                       const uint32_t* rows, int active_rows, const FeaturePair* pairs, int pair_count,
                                       const uint16_t* missing_codes, int reduced_bins, float* output_gradients,
                                       float* output_hessians, uint32_t* output_counts) {
    const int block_pair = blockIdx.x;
    if (block_pair >= pair_count)
        return;

    const FeaturePair pair = pairs[block_pair];
    const int cells = reduced_bins * reduced_bins;
    const int total = cells + 1; // cells + 1 missing cell

    // Shared histogram: [3 * total] = {grad[total], hess[total], count[total]}
    extern __shared__ float s_jhist[];
    const size_t s_total = static_cast<size_t>(total);

    // Zero shared histogram
    for (unsigned i = threadIdx.x; i < 3u * s_total; i += blockDim.x)
        s_jhist[i] = 0.0F;
    __syncthreads();

    // Accumulate per-row into shared histogram
    for (int row = threadIdx.x; row < active_rows; row += blockDim.x) {
        const uint32_t global_row = rows ? rows[row] : static_cast<uint32_t>(row);
        const uint16_t code_a = static_cast<uint16_t>(
            codes[static_cast<size_t>(global_row) * static_cast<size_t>(features) + pair.first]);
        const uint16_t code_b = static_cast<uint16_t>(
            codes[static_cast<size_t>(global_row) * static_cast<size_t>(features) + pair.second]);
        const uint16_t finite_a = missing_codes[pair.first];
        const uint16_t finite_b = missing_codes[pair.second];

        size_t bin = s_total; // default: missing cell (at index total)
        if (code_a != finite_a && code_b != finite_b && finite_a > 0 && finite_b > 0) {
            const int bin_a = min(reduced_bins - 1, static_cast<int>(code_a) * reduced_bins / finite_a);
            const int bin_b = min(reduced_bins - 1, static_cast<int>(code_b) * reduced_bins / finite_b);
            bin = static_cast<size_t>(bin_a * reduced_bins + bin_b);
        }
        atomicAdd(s_jhist + bin, gradients[global_row]);
        atomicAdd(s_jhist + s_total + bin, hessians[global_row]);
        atomicAdd(s_jhist + 2u * s_total + bin, 1.0F);
    }
    __syncthreads();

    // Scatter shared histogram to global memory (thread 0 handles the missing cell)
    const size_t base = static_cast<size_t>(block_pair) * static_cast<size_t>(total);
    if (cells > 0 && threadIdx.x == 0) {
        const size_t g_idx = base + cells;
        output_gradients[g_idx] = s_jhist[cells];
        output_hessians[g_idx] = s_jhist[s_total + cells];
        output_counts[g_idx] = static_cast<uint32_t>(s_jhist[2u * s_total + cells]);
    }
    // Scatter finite cells (all threads participate)
    for (unsigned i = threadIdx.x + 1u; i < s_total; i += blockDim.x) {
        const size_t g_idx = base + i;
        output_gradients[g_idx] = s_jhist[i];
        output_hessians[g_idx] = s_jhist[s_total + i];
        output_counts[g_idx] = static_cast<uint32_t>(s_jhist[2u * s_total + i]);
    }
}

__global__ void squared_error_kernel(const float* labels, const float* predictions, float* gradients, float* hessians,
                                     int rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;
    gradients[row] = predictions[row] - labels[row];
    hessians[row] = 1.0F;
}

__global__ void binary_logloss_kernel(const float* labels, const float* margins, float* gradients, float* hessians,
                                      int rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;
    const float probability = 1.0F / (1.0F + expf(-margins[row]));
    gradients[row] = probability - labels[row];
    hessians[row] = fmaxf(1.0e-12F, probability * (1.0F - probability));
}

__device__ float score(float gradient, float hessian, float lambda) {
    return hessian + lambda > 0.0F ? 0.5F * gradient * gradient / (hessian + lambda) : 0.0F;
}

// Optimized split scan kernel: unrolls the missing_left loop.
// Evaluates both "missing goes left" and "missing goes right" cases
// separately and picks the best, eliminating branch divergence.
__global__ void split_scan_kernel(const float* histogram_gradients, const float* histogram_hessians,
                                  const uint32_t* histogram_counts, const uint32_t* offsets, const uint16_t* bin_counts,
                                  int features, float parent_gradient, float parent_hessian, float lambda,
                                  uint32_t minimum_samples_leaf, float minimum_child_weight,
                                  SplitCandidate* candidates) {
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= features)
        return;

    const uint32_t offset = offsets[feature];
    const uint16_t bins = bin_counts[feature];
    const uint16_t finite_bins = bins > 0 ? static_cast<uint16_t>(bins - 1) : 0;
    const float missing_gradient = histogram_gradients[offset + finite_bins];
    const float missing_hessian = histogram_hessians[offset + finite_bins];
    const uint32_t missing_count = histogram_counts[offset + finite_bins];
    uint32_t total_count = missing_count;
    for (uint16_t bin = 0; bin < finite_bins; ++bin)
        total_count += histogram_counts[offset + bin];

    SplitCandidate best;
    float left_gradient = 0.0F;
    float left_hessian = 0.0F;
    uint32_t left_count = 0;
    const float parent_score = score(parent_gradient, parent_hessian, lambda);

    for (uint16_t bin = 0; bin + 1 < finite_bins; ++bin) {
        left_gradient += histogram_gradients[offset + bin];
        left_hessian += histogram_hessians[offset + bin];
        left_count += histogram_counts[offset + bin];

        const float right_gradient = parent_gradient - left_gradient;
        const float right_hessian = parent_hessian - left_hessian;
        const uint32_t right_count = total_count - left_count;

        // Case A: missing values go right (original non-missing-left)
        if (left_count >= minimum_samples_leaf && right_count >= minimum_samples_leaf &&
            left_hessian >= minimum_child_weight && right_hessian >= minimum_child_weight) {
            const float gain = score(left_gradient, left_hessian, lambda) +
                               score(right_gradient, right_hessian, lambda) - parent_score;
            if (gain > best.gain) {
                best.gain = gain;
                best.feature = static_cast<uint32_t>(feature);
                best.bin = bin;
                best.missing_left = false;
            }
        }

        // Case B: missing values go left (unrolled from original missing_left loop)
        const float ml_grad = left_gradient + missing_gradient;
        const float ml_hess = left_hessian + missing_hessian;
        const uint32_t ml_count = left_count + missing_count;
        const float mr_grad = parent_gradient - ml_grad;
        const float mr_hess = parent_hessian - ml_hess;
        const uint32_t mr_count = total_count - ml_count;

        if (ml_count >= minimum_samples_leaf && mr_count >= minimum_samples_leaf &&
            ml_hess >= minimum_child_weight && mr_hess >= minimum_child_weight) {
            const float gain = score(ml_grad, ml_hess, lambda) +
                               score(mr_grad, mr_hess, lambda) - parent_score;
            if (gain > best.gain) {
                best.gain = gain;
                best.feature = static_cast<uint32_t>(feature);
                best.bin = bin;
                best.missing_left = true;
            }
        }
    }
    candidates[feature] = best;
}

template <class Code>
__global__ void route_kernel(const Code* codes, int rows, int features, const uint16_t* missing_codes,
                             SplitCandidate split, uint8_t* routes) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;
    const uint16_t code = static_cast<uint16_t>(codes[static_cast<size_t>(row) * features + split.feature]);
    routes[row] = code == missing_codes[split.feature] ? static_cast<uint8_t>(split.missing_left)
                                                       : static_cast<uint8_t>(code <= split.bin);
}

} // namespace

struct CudaHistogramEngine::Impl {
    int rows = 0;
    int features = 0;
    QuantizedCodeWidth width = QuantizedCodeWidth::UInt8;
    void* codes = nullptr;
    float* gradients = nullptr;
    float* hessians = nullptr;
    uint16_t* missing_codes = nullptr;
    uint16_t* bin_counts = nullptr;
    uint32_t* offsets = nullptr;
    float* histogram_gradients = nullptr;
    float* histogram_hessians = nullptr;
    uint32_t* histogram_counts = nullptr;
    uint32_t total_bins = 0;
    // Cached buffers for objective computation (avoids per-call cudaMalloc)
    float* objective_labels = nullptr;
    float* objective_predictions = nullptr;
    size_t objective_capacity = 0;

    FeaturePair* joint_pairs = nullptr;
    uint32_t* joint_rows = nullptr;
    float* joint_gradients = nullptr;
    float* joint_hessians = nullptr;
    uint32_t* joint_counts = nullptr;
    size_t joint_pair_capacity = 0;
    size_t joint_row_capacity = 0;
    size_t joint_cell_capacity = 0;
    size_t allocated_bytes = 0;

    ~Impl() {
        cudaFree(codes);
        cudaFree(gradients);
        cudaFree(hessians);
        cudaFree(missing_codes);
        cudaFree(bin_counts);
        cudaFree(offsets);
        cudaFree(histogram_gradients);
        cudaFree(histogram_hessians);
        cudaFree(histogram_counts);
        cudaFree(objective_labels);
        cudaFree(objective_predictions);
        cudaFree(joint_pairs);
        cudaFree(joint_rows);
        cudaFree(joint_gradients);
        cudaFree(joint_hessians);
        cudaFree(joint_counts);
    }
};

CudaHistogramEngine::CudaHistogramEngine(const QuantizedDataset& dataset) : impl_(std::make_unique<Impl>()) {
    impl_->rows = dataset.rows();
    impl_->features = dataset.features();
    impl_->width = dataset.code_width();
    const size_t code_bytes = dataset.bytes();
    check(cudaMalloc(&impl_->codes, code_bytes), "cudaMalloc codes");
    dataset.visit_codes([&](auto codes) {
        check(cudaMemcpy(impl_->codes, codes.data(), code_bytes, cudaMemcpyHostToDevice), "copy quantized codes");
    });

    std::vector<uint16_t> bin_counts(static_cast<size_t>(impl_->features));
    std::vector<uint32_t> offsets(static_cast<size_t>(impl_->features + 1), 0);
    for (int feature = 0; feature < impl_->features; ++feature) {
        bin_counts[static_cast<size_t>(feature)] = static_cast<uint16_t>(dataset.missing_code(feature) + 1);
        offsets[static_cast<size_t>(feature + 1)] =
            offsets[static_cast<size_t>(feature)] + bin_counts[static_cast<size_t>(feature)];
    }
    impl_->total_bins = offsets.back();
    impl_->gradients = allocate<float>(static_cast<size_t>(impl_->rows));
    impl_->hessians = allocate<float>(static_cast<size_t>(impl_->rows));
    impl_->missing_codes = allocate<uint16_t>(static_cast<size_t>(impl_->features));
    impl_->bin_counts = allocate<uint16_t>(static_cast<size_t>(impl_->features));
    impl_->offsets = allocate<uint32_t>(static_cast<size_t>(impl_->features + 1));
    impl_->histogram_gradients = allocate<float>(impl_->total_bins);
    impl_->histogram_hessians = allocate<float>(impl_->total_bins);
    impl_->histogram_counts = allocate<uint32_t>(impl_->total_bins);
    // Allocate cached objective buffers
    impl_->objective_capacity = static_cast<size_t>(impl_->rows);
    impl_->objective_labels = allocate<float>(impl_->objective_capacity);
    impl_->objective_predictions = allocate<float>(impl_->objective_capacity);
    check(cudaMemcpy(impl_->missing_codes, dataset.missing_codes().data(),
                     static_cast<size_t>(impl_->features) * sizeof(uint16_t), cudaMemcpyHostToDevice),
          "copy missing codes");
    check(
        cudaMemcpy(impl_->bin_counts, bin_counts.data(), bin_counts.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
        "copy bin counts");
    check(cudaMemcpy(impl_->offsets, offsets.data(), offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
          "copy offsets");
    impl_->allocated_bytes = code_bytes + static_cast<size_t>(impl_->rows) * 2 * sizeof(float) +
                             static_cast<size_t>(impl_->features) * 2 * sizeof(uint16_t) +
                             static_cast<size_t>(impl_->features + 1) * sizeof(uint32_t) +
                             static_cast<size_t>(impl_->total_bins) * (2 * sizeof(float) + sizeof(uint32_t)) +
                             static_cast<size_t>(impl_->rows) * 2 * sizeof(float);
}

CudaHistogramEngine::~CudaHistogramEngine() = default;
CudaHistogramEngine::CudaHistogramEngine(CudaHistogramEngine&&) noexcept = default;
CudaHistogramEngine& CudaHistogramEngine::operator=(CudaHistogramEngine&&) noexcept = default;

void CudaHistogramEngine::set_gradients(std::span<const float> gradients, std::span<const float> hessians) {
    if (gradients.size() != static_cast<size_t>(impl_->rows) || hessians.size() != gradients.size()) {
        throw std::invalid_argument("CudaHistogramEngine::set_gradients: size mismatch");
    }
    check(cudaMemcpy(impl_->gradients, gradients.data(), gradients.size_bytes(), cudaMemcpyHostToDevice),
          "copy gradients");
    check(cudaMemcpy(impl_->hessians, hessians.data(), hessians.size_bytes(), cudaMemcpyHostToDevice), "copy hessians");
}

namespace {
template <class Impl, class Kernel>
void compute_objective(Impl& impl, std::span<const float> labels, std::span<const float> predictions, Kernel kernel) {
    if (labels.size() != static_cast<size_t>(impl.rows) || predictions.size() != labels.size()) {
        throw std::invalid_argument("CudaHistogramEngine objective: size mismatch");
    }
    // Use cached device buffers instead of per-call cudaMalloc
    check(cudaMemcpy(impl.objective_labels, labels.data(), labels.size_bytes(), cudaMemcpyHostToDevice),
          "copy labels");
    check(cudaMemcpy(impl.objective_predictions, predictions.data(), predictions.size_bytes(), cudaMemcpyHostToDevice),
          "copy predictions");
    kernel<<<(impl.rows + 255) / 256, 256>>>(impl.objective_labels, impl.objective_predictions, impl.gradients,
                                             impl.hessians, impl.rows);
    check(cudaGetLastError(), "objective kernel");
    check(cudaDeviceSynchronize(), "objective synchronize");
}
} // namespace

void CudaHistogramEngine::compute_squared_error_gradients(std::span<const float> labels,
                                                          std::span<const float> predictions) {
    compute_objective(*impl_, labels, predictions, squared_error_kernel);
}

void CudaHistogramEngine::compute_binary_logloss_gradients(std::span<const float> labels,
                                                           std::span<const float> margins) {
    compute_objective(*impl_, labels, margins, binary_logloss_kernel);
}

HistogramResult CudaHistogramEngine::build_histogram(std::span<const uint32_t> rows) {
    const int active_rows = rows.empty() ? impl_->rows : static_cast<int>(rows.size());
    uint32_t* device_rows = nullptr;
    if (!rows.empty()) {
        device_rows = allocate<uint32_t>(rows.size());
        check(cudaMemcpy(device_rows, rows.data(), rows.size_bytes(), cudaMemcpyHostToDevice), "copy row indices");
    }
    check(cudaMemset(impl_->histogram_gradients, 0, impl_->total_bins * sizeof(float)), "clear histogram gradients");
    check(cudaMemset(impl_->histogram_hessians, 0, impl_->total_bins * sizeof(float)), "clear histogram hessians");
    check(cudaMemset(impl_->histogram_counts, 0, impl_->total_bins * sizeof(uint32_t)), "clear histogram counts");

    // Optimized launch: one block per feature, 256 threads per block
    // Shared memory per block: 3 * 300 floats (grad/hess/count per bin, up to 300 bins/feature)
    // Well within the 48KB per-block limit. Each block handles one feature only.
    const size_t smem_per_block = 3u * 300u * sizeof(float);
    const int blocks = (impl_->features + 255) / 256;
    if (impl_->width == QuantizedCodeWidth::UInt8) {
        histogram_kernel<<<blocks, 256, static_cast<unsigned>(smem_per_block)>>>(
            static_cast<const uint8_t*>(impl_->codes), impl_->features, impl_->gradients, impl_->hessians, device_rows,
            active_rows, impl_->offsets, impl_->bin_counts, impl_->histogram_gradients, impl_->histogram_hessians,
            impl_->histogram_counts);
    } else {
        histogram_kernel<<<blocks, 256, static_cast<unsigned>(smem_per_block)>>>(
            static_cast<const uint16_t*>(impl_->codes), impl_->features, impl_->gradients, impl_->hessians, device_rows,
            active_rows, impl_->offsets, impl_->bin_counts, impl_->histogram_gradients, impl_->histogram_hessians,
            impl_->histogram_counts);
    }
    check(cudaGetLastError(), "histogram kernel");
    HistogramResult result;
    result.gradients.resize(impl_->total_bins);
    result.hessians.resize(impl_->total_bins);
    result.counts.resize(impl_->total_bins);
    check(cudaMemcpy(result.gradients.data(), impl_->histogram_gradients, result.gradients.size() * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "copy histogram gradients");
    check(cudaMemcpy(result.hessians.data(), impl_->histogram_hessians, result.hessians.size() * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "copy histogram hessians");
    check(cudaMemcpy(result.counts.data(), impl_->histogram_counts, result.counts.size() * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost),
          "copy histogram counts");
    cudaFree(device_rows);
    return result;
}

JointHistogramResult CudaHistogramEngine::build_joint_histograms(std::span<const FeaturePair> pairs, int reduced_bins,
                                                                 std::span<const uint32_t> rows) {
    if (pairs.empty())
        return JointHistogramResult{std::clamp(reduced_bins, 2, 16), 0, {}, {}, {}};
    reduced_bins = std::clamp(reduced_bins, 2, 16);
    for (const FeaturePair pair : pairs) {
        if (pair.first >= static_cast<uint32_t>(impl_->features) ||
            pair.second >= static_cast<uint32_t>(impl_->features) || pair.first == pair.second) {
            throw std::invalid_argument("CudaHistogramEngine::build_joint_histograms: invalid feature pair");
        }
    }
    const int active_rows = rows.empty() ? impl_->rows : static_cast<int>(rows.size());
    const size_t stride = static_cast<size_t>(reduced_bins * reduced_bins + 1);
    const size_t output_cells = pairs.size() * stride;

    if (pairs.size() > impl_->joint_pair_capacity) {
        cudaFree(impl_->joint_pairs);
        impl_->joint_pairs = allocate<FeaturePair>(pairs.size());
        impl_->allocated_bytes += (pairs.size() - impl_->joint_pair_capacity) * sizeof(FeaturePair);
        impl_->joint_pair_capacity = pairs.size();
    }
    if (!rows.empty() && rows.size() > impl_->joint_row_capacity) {
        cudaFree(impl_->joint_rows);
        impl_->joint_rows = allocate<uint32_t>(rows.size());
        impl_->allocated_bytes += (rows.size() - impl_->joint_row_capacity) * sizeof(uint32_t);
        impl_->joint_row_capacity = rows.size();
    }
    if (output_cells > impl_->joint_cell_capacity) {
        cudaFree(impl_->joint_gradients);
        cudaFree(impl_->joint_hessians);
        cudaFree(impl_->joint_counts);
        impl_->joint_gradients = allocate<float>(output_cells);
        impl_->joint_hessians = allocate<float>(output_cells);
        impl_->joint_counts = allocate<uint32_t>(output_cells);
        impl_->allocated_bytes += (output_cells - impl_->joint_cell_capacity) * (2 * sizeof(float) + sizeof(uint32_t));
        impl_->joint_cell_capacity = output_cells;
    }

    check(cudaMemcpy(impl_->joint_pairs, pairs.data(), pairs.size_bytes(), cudaMemcpyHostToDevice),
          "copy joint feature pairs");
    const uint32_t* device_rows = nullptr;
    if (!rows.empty()) {
        check(cudaMemcpy(impl_->joint_rows, rows.data(), rows.size_bytes(), cudaMemcpyHostToDevice),
              "copy joint row indices");
        device_rows = impl_->joint_rows;
    }
    check(cudaMemset(impl_->joint_gradients, 0, output_cells * sizeof(float)), "clear joint gradients");
    check(cudaMemset(impl_->joint_hessians, 0, output_cells * sizeof(float)), "clear joint hessians");
    check(cudaMemset(impl_->joint_counts, 0, output_cells * sizeof(uint32_t)), "clear joint counts");

    // Optimized launch: one block per pair, 512 threads with shared memory
    const int total_pairs = static_cast<int>(pairs.size());
    const int threads_per_block = 512;
    // Each block processes (512 / 256) = 2 pairs
    const int pairs_per_block = 2;
    const int blocks = std::min<int>(65535, (total_pairs + pairs_per_block - 1) / pairs_per_block);

    // Shared memory size: 3 * (cells + 1) floats
    const size_t smem_bytes = static_cast<size_t>(reduced_bins * reduced_bins + 1) * 3u * sizeof(float);

    if (impl_->width == QuantizedCodeWidth::UInt8) {
        joint_histogram_kernel<<<blocks, threads_per_block, static_cast<unsigned>(smem_bytes)>>>(
            static_cast<const uint8_t*>(impl_->codes), impl_->features, impl_->gradients, impl_->hessians, device_rows,
            active_rows, impl_->joint_pairs, total_pairs, impl_->missing_codes, reduced_bins,
            impl_->joint_gradients, impl_->joint_hessians, impl_->joint_counts);
    } else {
        joint_histogram_kernel<<<blocks, threads_per_block, static_cast<unsigned>(smem_bytes)>>>(
            static_cast<const uint16_t*>(impl_->codes), impl_->features, impl_->gradients, impl_->hessians, device_rows,
            active_rows, impl_->joint_pairs, total_pairs, impl_->missing_codes, reduced_bins,
            impl_->joint_gradients, impl_->joint_hessians, impl_->joint_counts);
    }
    check(cudaGetLastError(), "joint histogram kernel");

    JointHistogramResult result;
    result.reduced_bins = reduced_bins;
    result.pair_count = pairs.size();
    result.gradients.resize(output_cells);
    result.hessians.resize(output_cells);
    result.counts.resize(output_cells);
    check(cudaMemcpy(result.gradients.data(), impl_->joint_gradients, output_cells * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "copy joint gradients");
    check(
        cudaMemcpy(result.hessians.data(), impl_->joint_hessians, output_cells * sizeof(float), cudaMemcpyDeviceToHost),
        "copy joint hessians");
    check(
        cudaMemcpy(result.counts.data(), impl_->joint_counts, output_cells * sizeof(uint32_t), cudaMemcpyDeviceToHost),
        "copy joint counts");
    return result;
}

SplitCandidate CudaHistogramEngine::find_best_split(float parent_gradient, float parent_hessian, float lambda,
                                                    uint32_t minimum_samples_leaf, float minimum_child_weight) {
    (void)build_histogram();
    SplitCandidate* device_candidates = allocate<SplitCandidate>(static_cast<size_t>(impl_->features));
    split_scan_kernel<<<(impl_->features + 255) / 256, 256>>>(
        impl_->histogram_gradients, impl_->histogram_hessians, impl_->histogram_counts, impl_->offsets,
        impl_->bin_counts, impl_->features, parent_gradient, parent_hessian, lambda, minimum_samples_leaf,
        minimum_child_weight, device_candidates);
    check(cudaGetLastError(), "split scan kernel");
    std::vector<SplitCandidate> candidates(static_cast<size_t>(impl_->features));
    check(cudaMemcpy(candidates.data(), device_candidates, candidates.size() * sizeof(SplitCandidate),
                     cudaMemcpyDeviceToHost),
          "copy split candidates");
    cudaFree(device_candidates);
    return *std::max_element(candidates.begin(), candidates.end(),
                             [](const auto& left, const auto& right) { return left.gain < right.gain; });
}

std::vector<uint8_t> CudaHistogramEngine::route_rows(const SplitCandidate& split) {
    uint8_t* device_routes = allocate<uint8_t>(static_cast<size_t>(impl_->rows));
    if (impl_->width == QuantizedCodeWidth::UInt8) {
        route_kernel<<<(impl_->rows + 255) / 256, 256>>>(static_cast<const uint8_t*>(impl_->codes), impl_->rows,
                                                         impl_->features, impl_->missing_codes, split, device_routes);
    } else {
        route_kernel<<<(impl_->rows + 255) / 256, 256>>>(static_cast<const uint16_t*>(impl_->codes), impl_->rows,
                                                         impl_->features, impl_->missing_codes, split, device_routes);
    }
    check(cudaGetLastError(), "route kernel");
    std::vector<uint8_t> routes(static_cast<size_t>(impl_->rows));
    check(cudaMemcpy(routes.data(), device_routes, routes.size(), cudaMemcpyDeviceToHost), "copy routes");
    cudaFree(device_routes);
    return routes;
}

int CudaHistogramEngine::rows() const noexcept {
    return impl_->rows;
}
int CudaHistogramEngine::features() const noexcept {
    return impl_->features;
}
size_t CudaHistogramEngine::device_bytes() const noexcept {
    return impl_->allocated_bytes;
}

bool is_available() noexcept {
    int devices = 0;
    return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

} // namespace foretree::cuda
