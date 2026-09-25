#include "foretree/gpu/gpu_trainer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_partition.cuh>

namespace foretree::cuda {
namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("GpuTreeTrainer: ") + what + ": " + cudaGetErrorString(status));
}

template <class T> T* device_alloc(size_t count) {
    T* ptr = nullptr;
    if (count > 0)
        check(cudaMalloc(&ptr, count * sizeof(T)), "cudaMalloc");
    return ptr;
}

// Gradients are quantized to int32 levels of |x| <= 2^22; per-bin sums are
// int64 (exact for up to 2^41 rows).
constexpr double kLevels = 4194304.0;  // 2^22

struct Candidate {
    double gain;
    double GL, HL;
    double Gp, Hp;  // node totals (the same for every feature)
    long long CL;
    int feature;
    int bin;
    int missing_left;
    int pad;
};

__device__ inline double soft_threshold(double g, double alpha) {
    if (alpha <= 0.0)
        return g;
    if (g > alpha)
        return g - alpha;
    if (g < -alpha)
        return g + alpha;
    return 0.0;
}

__device__ inline double leaf_objective(double G, double H, double lambda, double alpha) {
    const double denom = H + lambda;
    if (!(denom > 0.0))
        return 0.0;
    const double s = soft_threshold(G, alpha);
    return 0.5 * s * s / denom;
}

// ---------------------------------------------------------------- gradients

__global__ void gradients_kernel(const float* labels, const float* weights, const float* margins, float* g,
                                 float* h, int n, int objective, int num_classes, int class_index,
                                 unsigned int* max_bits) {
    __shared__ float block_g, block_h;
    if (threadIdx.x == 0) {
        block_g = 0.0F;
        block_h = 0.0F;
    }
    __syncthreads();
    float local_g = 0.0F, local_h = 0.0F;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float gi, hi;
        if (objective == 2) {
            // Softmax over explicit class margins plus the implicit class (logit 0).
            float max_f = 0.0F;
            for (int k = 0; k < num_classes; ++k)
                max_f = fmaxf(max_f, margins[static_cast<size_t>(k) * n + i]);
            float sum = expf(-max_f);
            for (int k = 0; k < num_classes; ++k)
                sum += expf(margins[static_cast<size_t>(k) * n + i] - max_f);
            const float p = expf(margins[static_cast<size_t>(class_index) * n + i] - max_f) / sum;
            const float target = static_cast<int>(labels[i]) == class_index ? 1.0F : 0.0F;
            gi = p - target;
            hi = fmaxf(1e-12F, p * (1.0F - p));
        } else if (objective == 1) {
            const float p = 1.0F / (1.0F + expf(-margins[i]));
            gi = p - labels[i];
            hi = fmaxf(1e-12F, p * (1.0F - p));
        } else {
            gi = margins[i] - labels[i];
            hi = 1.0F;
        }
        if (weights) {
            gi *= weights[i];
            hi *= weights[i];
        }
        g[i] = gi;
        h[i] = hi;
        local_g = fmaxf(local_g, fabsf(gi));
        local_h = fmaxf(local_h, hi);
    }
    // Non-negative floats order like their bit patterns.
    atomicMax(reinterpret_cast<unsigned int*>(&block_g), __float_as_uint(local_g));
    atomicMax(reinterpret_cast<unsigned int*>(&block_h), __float_as_uint(local_h));
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicMax(&max_bits[0], __float_as_uint(block_g));
        atomicMax(&max_bits[1], __float_as_uint(block_h));
    }
}

__global__ void quantize_kernel(const float* g, const float* h, int* gq, int* hq, int n, double g_inv, double h_inv) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        gq[i] = __double2int_rn(static_cast<double>(g[i]) * g_inv);
        hq[i] = __double2int_rn(static_cast<double>(h[i]) * h_inv);
    }
}

__global__ void iota_kernel(int* rows, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        rows[i] = i;
}

// ---------------------------------------------------------------- histogram

// Histogram of rows[begin, end) into one node slot. blockIdx.x: feature,
// blockIdx.y: strided share of the rows. Shared-memory integer histogram, then
// integer atomics into the slot (zeroed beforehand). Codes are feature-major.
template <class Code>
__global__ void histogram_kernel(const Code* codes, int n_rows, const int* rows, int begin, int end,
                                 const int* gq, const int* hq, const int* offsets, const int* bins,
                                 long long* slot_g, long long* slot_h, int* slot_c) {
    const int f = blockIdx.x;
    const int nb = bins[f];
    extern __shared__ unsigned char smem[];
    long long* sg = reinterpret_cast<long long*>(smem);
    long long* sh = sg + nb;
    int* sc = reinterpret_cast<int*>(sh + nb);
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        sg[b] = 0;
        sh[b] = 0;
        sc[b] = 0;
    }
    __syncthreads();
    const Code* column = codes + static_cast<size_t>(f) * static_cast<size_t>(n_rows);
    const int last = nb - 1;
    const int stride = blockDim.x * gridDim.y;
    for (int i = begin + blockIdx.y * blockDim.x + threadIdx.x; i < end; i += stride) {
        const int row = rows[i];
        const int code = static_cast<int>(column[row]);
        const int b = code < last ? code : last;
        atomicAdd(reinterpret_cast<unsigned long long*>(&sg[b]), static_cast<unsigned long long>(static_cast<long long>(gq[row])));
        atomicAdd(reinterpret_cast<unsigned long long*>(&sh[b]), static_cast<unsigned long long>(static_cast<long long>(hq[row])));
        atomicAdd(&sc[b], 1);
    }
    __syncthreads();
    const int off = offsets[f];
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        if (sc[b] == 0)
            continue;
        atomicAdd(reinterpret_cast<unsigned long long*>(&slot_g[off + b]), static_cast<unsigned long long>(sg[b]));
        atomicAdd(reinterpret_cast<unsigned long long*>(&slot_h[off + b]), static_cast<unsigned long long>(sh[b]));
        atomicAdd(&slot_c[off + b], sc[b]);
    }
}

// larger = parent - smaller, in place in the parent's slot.
__global__ void subtract_kernel(long long* pg, long long* ph, int* pc, const long long* sg, const long long* sh,
                                const int* sc, int total) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        pg[i] -= sg[i];
        ph[i] -= sh[i];
        pc[i] -= sc[i];
    }
}

// ------------------------------------------------------------ split search

struct SplitParams {
    double lambda, alpha, gamma, min_child_weight;
    long long min_samples_leaf;
    int missing_policy;
    double g_scale, h_scale;
};

// Best split of one (node, feature): blockIdx.x = feature, blockIdx.y = node.
// One thread per bin: exact integer prefix sums (block scans), each thread
// scores its threshold for both missing directions, then a block reduction.
// Tie-breaking matches the CPU scan order: missing-left before missing-right,
// then the earliest threshold.
struct Scored {
    double gain;
    int key;  // direction * 65536 + bin; smaller key wins ties
};
struct BetterScored {
    __device__ Scored operator()(const Scored& a, const Scored& b) const {
        if (a.gain > b.gain)
            return a;
        if (b.gain > a.gain)
            return b;
        return a.key <= b.key ? a : b;
    }
};

template <int BLOCK>
__global__ void split_kernel(const long long* hist_g, const long long* hist_h, const int* hist_c,
                             const int* node_slots, int total_bins, const int* offsets, const int* bins,
                             int n_features, SplitParams p, const unsigned char* feature_mask, Candidate* out) {
    using ScanLL = cub::BlockScan<long long, BLOCK>;
    using ReduceS = cub::BlockReduce<Scored, BLOCK>;
    __shared__ union {
        typename ScanLL::TempStorage scan;
        typename ReduceS::TempStorage reduce;
    } temp;
    __shared__ long long totals[3];
    __shared__ Scored winner;

    const int f = blockIdx.x;
    const int node = blockIdx.y;
    const size_t base = static_cast<size_t>(node_slots[node]) * static_cast<size_t>(total_bins) + offsets[f];
    const int nb = bins[f];
    const int miss = nb - 1;
    const int t = threadIdx.x;

    const long long g = t < nb ? hist_g[base + t] : 0;
    const long long h = t < nb ? hist_h[base + t] : 0;
    const long long c = t < nb ? static_cast<long long>(hist_c[base + t]) : 0;
    long long gl, hl, cl, agg;
    ScanLL(temp.scan).InclusiveSum(g, gl, agg);
    if (t == 0) totals[0] = agg;
    __syncthreads();
    ScanLL(temp.scan).InclusiveSum(h, hl, agg);
    if (t == 0) totals[1] = agg;
    __syncthreads();
    ScanLL(temp.scan).InclusiveSum(c, cl, agg);
    if (t == 0) totals[2] = agg;
    __syncthreads();

    const long long Ct = totals[2];
    const double Gp = totals[0] * p.g_scale, Hp = totals[1] * p.h_scale;
    const double parent = leaf_objective(Gp, Hp, p.lambda, p.alpha);
    // Missing bin sums (the missing bin is the last one).
    const long long cm = static_cast<long long>(hist_c[base + miss]);
    const double gm = hist_g[base + miss] * p.g_scale, hm = hist_h[base + miss] * p.h_scale;
    const bool has_miss = cm > 0;

    // Masked features (column subsampling) still report the node totals.
    const bool enabled = feature_mask == nullptr || feature_mask[static_cast<size_t>(node) * n_features + f] != 0;
    Scored mine{-INFINITY, 0x7fffffff};
    if (enabled && t < miss && c > 0) {
        auto score = [&](bool miss_left) -> double {
            const long long cl_x = cl + (miss_left && has_miss ? cm : 0);
            const long long cr = Ct - cl_x;
            if (cl_x < p.min_samples_leaf || cr < p.min_samples_leaf)
                return -INFINITY;
            const double GL = gl * p.g_scale + (miss_left && has_miss ? gm : 0.0);
            const double HL = hl * p.h_scale + (miss_left && has_miss ? hm : 0.0);
            const double GR = Gp - GL, HR = Hp - HL;
            if (HL < p.min_child_weight || HR < p.min_child_weight)
                return -INFINITY;
            return leaf_objective(GL, HL, p.lambda, p.alpha) + leaf_objective(GR, HR, p.lambda, p.alpha) - parent -
                   p.gamma;
        };
        if (!has_miss || p.missing_policy != 0) {
            const bool left = p.missing_policy != 2;
            mine = Scored{score(left), (left ? 0 : 1) * 65536 + t};
        } else {
            const Scored a{score(true), t};
            const Scored b{score(false), 65536 + t};
            mine = BetterScored{}(a, b);
        }
    }
    const Scored best = ReduceS(temp.reduce).Reduce(mine, BetterScored{});
    if (t == 0) winner = best;
    __syncthreads();

    // The winning thread writes the candidate (it holds the prefix sums).
    Candidate* o = out + static_cast<size_t>(node) * n_features + f;
    if (t == 0 && !(winner.gain > -INFINITY)) {
        Candidate none{};
        none.gain = -INFINITY;
        none.Gp = Gp;
        none.Hp = Hp;
        none.feature = f;
        none.bin = -1;
        none.missing_left = 1;
        *o = none;
    }
    if (winner.gain > -INFINITY && (winner.key & 0xFFFF) == t) {
        const bool miss_left = (winner.key >> 16) == 0;
        Candidate cand{};
        cand.gain = winner.gain;
        cand.Gp = Gp;
        cand.Hp = Hp;
        cand.GL = gl * p.g_scale + (miss_left && has_miss ? gm : 0.0);
        cand.HL = hl * p.h_scale + (miss_left && has_miss ? hm : 0.0);
        cand.CL = cl + (miss_left && has_miss ? cm : 0);
        cand.feature = f;
        cand.bin = t;
        cand.missing_left = miss_left ? 1 : 0;
        *o = cand;
    }
}

// ------------------------------------------------------------ partition

template <class Code> struct GoesLeft {
    const Code* column;
    int threshold;
    int missing_code;
    bool missing_left;
    __device__ bool operator()(int row) const {
        const int code = static_cast<int>(column[row]);
        return code == missing_code ? missing_left : code <= threshold;
    }
};

// ------------------------------------------------------------ margins

__global__ void update_margins_kernel(float* margins, const int* rows, const int* leaf_begin,
                                      const float* leaf_delta, int n_leaves, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        int lo = 0, hi = n_leaves - 1;  // last leaf whose begin <= i
        while (lo < hi) {
            const int mid = (lo + hi + 1) / 2;
            if (leaf_begin[mid] <= i)
                lo = mid;
            else
                hi = mid - 1;
        }
        margins[rows[i]] += leaf_delta[lo];
    }
}

// margins[row] += delta[leaf reached by row], traversing the tree on the codes.
template <class Code>
__global__ void traverse_update_kernel(const Code* codes, int n, const int* missing_codes, const int* feature,
                                       const int* threshold, const unsigned char* missing_left, const int* left,
                                       const int* right, const float* delta, float* margins) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        int node = 0;
        while (feature[node] >= 0) {
            const int f = feature[node];
            const int code = static_cast<int>(codes[static_cast<size_t>(f) * n + row]);
            const bool go_left = code == missing_codes[f] ? missing_left[node] != 0 : code <= threshold[node];
            node = go_left ? left[node] : right[node];
        }
        margins[row] += delta[node];
    }
}

int grid_for(int n, int block = 256) {
    return std::max(1, std::min((n + block - 1) / block, 4096));
}

}  // namespace

struct GpuTreeTrainer::Impl {
    int n = 0;
    int p = 0;
    bool wide = false;  // uint16 codes
    void* codes = nullptr;
    float* labels = nullptr;
    float* weights = nullptr;
    float* margins = nullptr;
    float* g = nullptr;
    float* h = nullptr;
    int* gq = nullptr;
    int* hq = nullptr;
    int* rows = nullptr;
    int* rows_tmp = nullptr;
    int* offsets = nullptr;
    int* bins = nullptr;
    unsigned int* max_bits = nullptr;
    std::vector<int> host_offsets, host_bins, host_missing;
    int total_bins = 0;
    int max_feature_bins = 0;
    // Histogram slots (one per live node).
    int n_slots = 0;
    long long* hist_g = nullptr;
    long long* hist_h = nullptr;
    int* hist_c = nullptr;
    // Split launch buffers.
    int* node_slots = nullptr;
    Candidate* candidates = nullptr;
    // CUB partition scratch.
    void* cub_temp = nullptr;
    size_t cub_temp_bytes = 0;
    int* num_selected = nullptr;
    // Margin update buffers.
    int* leaf_begin = nullptr;
    float* leaf_delta = nullptr;
    int leaf_capacity = 0;
    GpuObjective objective = GpuObjective::SquaredError;
    int num_classes = 1;
    double g_scale = 1.0, h_scale = 1.0;
    // Rows of the last grown tree: all rows (leaf ranges cover every row) or a
    // subsample (margin updates traverse the tree).
    bool last_all_rows = true;
    int active_rows = 0;
    // Column subsampling masks for one split launch (<= 2 nodes x p).
    unsigned char* feature_mask = nullptr;
    int* missing_codes = nullptr;
    // Traversal buffers (one tree).
    int* t_feature = nullptr;
    int* t_threshold = nullptr;
    unsigned char* t_missing_left = nullptr;
    int* t_left = nullptr;
    int* t_right = nullptr;
    float* t_delta = nullptr;
    int t_capacity = 0;

    ~Impl() {
        for (void* ptr : std::initializer_list<void*>{codes, labels, weights, margins, g, h, gq, hq, rows, rows_tmp,
                                                      offsets, bins, max_bits, hist_g, hist_h, hist_c, node_slots,
                                                      candidates, cub_temp, num_selected, leaf_begin, leaf_delta,
                                                      feature_mask, missing_codes, t_feature, t_threshold,
                                                      t_missing_left, t_left, t_right, t_delta})
            cudaFree(ptr);
    }

    void ensure_slots(int count) {
        if (count <= n_slots)
            return;
        cudaFree(hist_g);
        cudaFree(hist_h);
        cudaFree(hist_c);
        const size_t cells = static_cast<size_t>(count) * static_cast<size_t>(total_bins);
        hist_g = device_alloc<long long>(cells);
        hist_h = device_alloc<long long>(cells);
        hist_c = device_alloc<int>(cells);
        cudaFree(node_slots);
        cudaFree(candidates);
        node_slots = device_alloc<int>(2);
        candidates = device_alloc<Candidate>(2 * static_cast<size_t>(p));
        n_slots = count;
    }

    void clear_slot(int slot) {
        const size_t off = static_cast<size_t>(slot) * static_cast<size_t>(total_bins);
        check(cudaMemsetAsync(hist_g + off, 0, total_bins * sizeof(long long)), "clear hist");
        check(cudaMemsetAsync(hist_h + off, 0, total_bins * sizeof(long long)), "clear hist");
        check(cudaMemsetAsync(hist_c + off, 0, total_bins * sizeof(int)), "clear hist");
    }

    void build_histogram(int slot, int begin, int end) {
        clear_slot(slot);
        const int count = end - begin;
        if (count <= 0)
            return;
        const size_t off = static_cast<size_t>(slot) * static_cast<size_t>(total_bins);
        constexpr int threads = 256;
        const int chunks = std::clamp((count + threads * 16 - 1) / (threads * 16), 1, 128);
        const dim3 grid(static_cast<unsigned>(p), static_cast<unsigned>(chunks));
        const size_t smem = static_cast<size_t>(max_feature_bins) * (2 * sizeof(long long) + sizeof(int));
        if (wide)
            histogram_kernel<uint16_t><<<grid, threads, smem>>>(static_cast<const uint16_t*>(codes), n, rows, begin,
                                                                end, gq, hq, offsets, bins, hist_g + off,
                                                                hist_h + off, hist_c + off);
        else
            histogram_kernel<uint8_t><<<grid, threads, smem>>>(static_cast<const uint8_t*>(codes), n, rows, begin,
                                                               end, gq, hq, offsets, bins, hist_g + off,
                                                               hist_h + off, hist_c + off);
        check(cudaGetLastError(), "histogram kernel");
    }

    void subtract(int parent_slot, int small_slot) {
        const size_t po = static_cast<size_t>(parent_slot) * static_cast<size_t>(total_bins);
        const size_t so = static_cast<size_t>(small_slot) * static_cast<size_t>(total_bins);
        subtract_kernel<<<grid_for(total_bins), 256>>>(hist_g + po, hist_h + po, hist_c + po, hist_g + so,
                                                       hist_h + so, hist_c + so, total_bins);
        check(cudaGetLastError(), "subtract kernel");
    }

    // Best split for up to two node slots; returns one candidate per node.
    // `masks`: empty (all features) or slots.size() * p flags.
    std::vector<Candidate> find_splits(const std::vector<int>& slots, const GpuTreeParams& params,
                                       const std::vector<unsigned char>& masks = {}) {
        check(cudaMemcpy(node_slots, slots.data(), slots.size() * sizeof(int), cudaMemcpyHostToDevice),
              "copy node slots");
        const unsigned char* mask_ptr = nullptr;
        if (!masks.empty()) {
            check(cudaMemcpy(feature_mask, masks.data(), masks.size(), cudaMemcpyHostToDevice), "copy masks");
            mask_ptr = feature_mask;
        }
        SplitParams sp{params.lambda, params.alpha, params.gamma, params.min_child_weight,
                       static_cast<long long>(std::max(1, params.min_samples_leaf)), params.missing_policy,
                       g_scale, h_scale};
        const dim3 grid(static_cast<unsigned>(p), static_cast<unsigned>(slots.size()));
        if (max_feature_bins <= 256)
            split_kernel<256><<<grid, 256>>>(hist_g, hist_h, hist_c, node_slots, total_bins, offsets, bins, p, sp, mask_ptr,
                                             candidates);
        else if (max_feature_bins <= 512)
            split_kernel<512><<<grid, 512>>>(hist_g, hist_h, hist_c, node_slots, total_bins, offsets, bins, p, sp, mask_ptr,
                                             candidates);
        else
            split_kernel<1024><<<grid, 1024>>>(hist_g, hist_h, hist_c, node_slots, total_bins, offsets, bins, p, sp, mask_ptr,
                                               candidates);
        check(cudaGetLastError(), "split kernel");
        std::vector<Candidate> all(slots.size() * static_cast<size_t>(p));
        check(cudaMemcpy(all.data(), candidates, all.size() * sizeof(Candidate), cudaMemcpyDeviceToHost),
              "copy candidates");
        // Best over features, in feature order with strict improvement (as on CPU).
        std::vector<Candidate> best(slots.size());
        for (size_t node = 0; node < slots.size(); ++node) {
            Candidate b{};
            b.gain = -std::numeric_limits<double>::infinity();
            b.bin = -1;
            b.Gp = all[node * static_cast<size_t>(p)].Gp;
            b.Hp = all[node * static_cast<size_t>(p)].Hp;
            for (int f = 0; f < p; ++f) {
                const Candidate& c = all[node * static_cast<size_t>(p) + static_cast<size_t>(f)];
                if (c.bin >= 0 && c.gain > b.gain)
                    b = c;
            }
            best[node] = b;
        }
        return best;
    }

    void partition(int begin, int end, int feature, int threshold, bool missing_left) {
        const int count = end - begin;
        const int missing_code = host_missing[static_cast<size_t>(feature)];
        auto run = [&](auto code_tag) {
            using Code = decltype(code_tag);
            const Code* column = static_cast<const Code*>(codes) + static_cast<size_t>(feature) * static_cast<size_t>(n);
            GoesLeft<Code> pred{column, threshold, missing_code, missing_left};
            size_t bytes = cub_temp_bytes;
            check(cub::DevicePartition::If(cub_temp, bytes, rows + begin, rows_tmp, num_selected, count, pred),
                  "partition");
        };
        if (wide)
            run(uint16_t{});
        else
            run(uint8_t{});
        // Selected (left) rows keep their order; right rows come reversed.
        check(cudaMemcpyAsync(rows + begin, rows_tmp, static_cast<size_t>(count) * sizeof(int),
                              cudaMemcpyDeviceToDevice),
              "copy partition");
    }
};

GpuTreeTrainer::GpuTreeTrainer(const QuantizedDataset& dataset, std::span<const double> labels,
                               std::span<const double> weights, GpuObjective objective, double base_score,
                               int num_classes)
    : impl_(std::make_unique<Impl>()) {
    Impl& s = *impl_;
    s.n = dataset.rows();
    s.p = dataset.features();
    s.objective = objective;
    s.num_classes = std::max(1, num_classes);
    if (objective != GpuObjective::Multiclass && s.num_classes != 1)
        throw std::invalid_argument("GpuTreeTrainer: num_classes > 1 needs the multiclass objective");
    if (static_cast<int>(labels.size()) != s.n)
        throw std::invalid_argument("GpuTreeTrainer: labels size mismatch");
    if (!weights.empty() && static_cast<int>(weights.size()) != s.n)
        throw std::invalid_argument("GpuTreeTrainer: weights size mismatch");

    dataset.visit_feature_major_codes([&](auto codes) {
        using Code = typename decltype(codes)::value_type;
        s.wide = sizeof(Code) == 2;
        s.codes = device_alloc<Code>(codes.size());
        check(cudaMemcpy(s.codes, codes.data(), codes.size() * sizeof(Code), cudaMemcpyHostToDevice), "copy codes");
    });

    s.host_offsets.assign(static_cast<size_t>(s.p) + 1, 0);
    s.host_bins.resize(static_cast<size_t>(s.p));
    s.host_missing.resize(static_cast<size_t>(s.p));
    for (int f = 0; f < s.p; ++f) {
        s.host_missing[static_cast<size_t>(f)] = dataset.missing_code(f);
        s.host_bins[static_cast<size_t>(f)] = dataset.missing_code(f) + 1;
        s.host_offsets[static_cast<size_t>(f) + 1] = s.host_offsets[static_cast<size_t>(f)] + s.host_bins[static_cast<size_t>(f)];
        s.max_feature_bins = std::max(s.max_feature_bins, s.host_bins[static_cast<size_t>(f)]);
    }
    s.total_bins = s.host_offsets.back();
    if (s.max_feature_bins > 1024)
        throw std::invalid_argument("GpuTreeTrainer: at most 1024 bins per feature (max_bins <= 1023)");
    s.offsets = device_alloc<int>(s.host_offsets.size());
    s.bins = device_alloc<int>(s.host_bins.size());
    check(cudaMemcpy(s.offsets, s.host_offsets.data(), s.host_offsets.size() * sizeof(int), cudaMemcpyHostToDevice),
          "copy offsets");
    check(cudaMemcpy(s.bins, s.host_bins.data(), s.host_bins.size() * sizeof(int), cudaMemcpyHostToDevice),
          "copy bins");

    std::vector<float> buffer(static_cast<size_t>(s.n));
    std::transform(labels.begin(), labels.end(), buffer.begin(), [](double v) { return static_cast<float>(v); });
    s.labels = device_alloc<float>(buffer.size());
    check(cudaMemcpy(s.labels, buffer.data(), buffer.size() * sizeof(float), cudaMemcpyHostToDevice), "copy labels");
    if (!weights.empty()) {
        std::transform(weights.begin(), weights.end(), buffer.begin(), [](double v) { return static_cast<float>(v); });
        s.weights = device_alloc<float>(buffer.size());
        check(cudaMemcpy(s.weights, buffer.data(), buffer.size() * sizeof(float), cudaMemcpyHostToDevice),
              "copy weights");
    }
    // Margins are class-major: margins[class * n + row].
    std::vector<float> init(static_cast<size_t>(s.n) * static_cast<size_t>(s.num_classes),
                            static_cast<float>(base_score));
    s.margins = device_alloc<float>(init.size());
    check(cudaMemcpy(s.margins, init.data(), init.size() * sizeof(float), cudaMemcpyHostToDevice),
          "copy margins");
    s.feature_mask = device_alloc<unsigned char>(2 * static_cast<size_t>(s.p));
    s.missing_codes = device_alloc<int>(static_cast<size_t>(s.p));
    check(cudaMemcpy(s.missing_codes, s.host_missing.data(), s.host_missing.size() * sizeof(int),
                     cudaMemcpyHostToDevice),
          "copy missing codes");

    s.g = device_alloc<float>(static_cast<size_t>(s.n));
    s.h = device_alloc<float>(static_cast<size_t>(s.n));
    s.gq = device_alloc<int>(static_cast<size_t>(s.n));
    s.hq = device_alloc<int>(static_cast<size_t>(s.n));
    s.rows = device_alloc<int>(static_cast<size_t>(s.n));
    s.rows_tmp = device_alloc<int>(static_cast<size_t>(s.n));
    s.max_bits = device_alloc<unsigned int>(2);
    s.num_selected = device_alloc<int>(1);

    // CUB scratch sized for the largest partition (all rows).
    GoesLeft<uint8_t> probe{nullptr, 0, 0, true};
    size_t bytes = 0;
    check(cub::DevicePartition::If(nullptr, bytes, s.rows, s.rows_tmp, s.num_selected, s.n, probe),
          "partition scratch size");
    GoesLeft<uint16_t> probe16{nullptr, 0, 0, true};
    size_t bytes16 = 0;
    check(cub::DevicePartition::If(nullptr, bytes16, s.rows, s.rows_tmp, s.num_selected, s.n, probe16),
          "partition scratch size");
    s.cub_temp_bytes = std::max(bytes, bytes16);
    s.cub_temp = device_alloc<unsigned char>(s.cub_temp_bytes);
}

GpuTreeTrainer::~GpuTreeTrainer() = default;

int GpuTreeTrainer::rows() const noexcept {
    return impl_->n;
}
int GpuTreeTrainer::features() const noexcept {
    return impl_->p;
}

std::vector<GpuTreeNode> GpuTreeTrainer::grow(const GpuTreeParams& params, int class_index,
                                              std::span<const int> rows) {
    Impl& s = *impl_;
    const int max_leaves = std::max(1, params.max_leaves);
    s.ensure_slots(max_leaves + 1);
    if (class_index < 0 || class_index >= s.num_classes)
        throw std::invalid_argument("GpuTreeTrainer::grow: class_index out of range");

    // Gradients, their ranges, then fixed-point levels.
    const int objective = s.objective == GpuObjective::Multiclass      ? 2
                          : s.objective == GpuObjective::BinaryLogloss ? 1
                                                                       : 0;
    check(cudaMemset(s.max_bits, 0, 2 * sizeof(unsigned int)), "clear max");
    gradients_kernel<<<grid_for(s.n), 256>>>(s.labels, s.weights, s.margins, s.g, s.h, s.n, objective,
                                             s.num_classes, class_index, s.max_bits);
    check(cudaGetLastError(), "gradients kernel");
    unsigned int bits[2];
    check(cudaMemcpy(bits, s.max_bits, sizeof(bits), cudaMemcpyDeviceToHost), "copy max");
    float g_max, h_max;
    std::memcpy(&g_max, &bits[0], sizeof(float));
    std::memcpy(&h_max, &bits[1], sizeof(float));
    s.g_scale = g_max > 0.0F ? static_cast<double>(g_max) / kLevels : 1.0;
    s.h_scale = h_max > 0.0F ? static_cast<double>(h_max) / kLevels : 1.0;
    quantize_kernel<<<grid_for(s.n), 256>>>(s.g, s.h, s.gq, s.hq, s.n, 1.0 / s.g_scale, 1.0 / s.h_scale);
    check(cudaGetLastError(), "quantize kernel");
    // Root rows: all rows, or the given subsample.
    if (rows.empty()) {
        iota_kernel<<<grid_for(s.n), 256>>>(s.rows, s.n);
        check(cudaGetLastError(), "iota kernel");
        s.active_rows = s.n;
    } else {
        if (static_cast<int>(rows.size()) > s.n)
            throw std::invalid_argument("GpuTreeTrainer::grow: more rows than the dataset");
        check(cudaMemcpy(s.rows, rows.data(), rows.size_bytes(), cudaMemcpyHostToDevice), "copy rows");
        s.active_rows = static_cast<int>(rows.size());
    }
    s.last_all_rows = s.active_rows == s.n;

    // Column subsampling: a feature pool per tree, a subset of it per node.
    std::mt19937_64 rng(params.seed ^ 0x5DEECE66DULL);
    std::vector<int> pool(static_cast<size_t>(s.p));
    std::iota(pool.begin(), pool.end(), 0);
    if (params.feature_bagging_k > 0) {
        const int k = std::min(params.feature_bagging_k, s.p);
        std::vector<int> chosen;
        if (params.feature_bagging_with_replacement) {
            std::uniform_int_distribution<int> pick(0, s.p - 1);
            for (int i = 0; i < k; ++i)
                chosen.push_back(pick(rng));
            std::sort(chosen.begin(), chosen.end());
            chosen.erase(std::unique(chosen.begin(), chosen.end()), chosen.end());
        } else {
            std::shuffle(pool.begin(), pool.end(), rng);
            chosen.assign(pool.begin(), pool.begin() + k);
        }
        pool = std::move(chosen);
    } else if (params.tree_feature_percent < 100) {
        const int k = std::max(1, s.p * std::max(1, params.tree_feature_percent) / 100);
        std::shuffle(pool.begin(), pool.end(), rng);
        pool.resize(static_cast<size_t>(k));
    }
    const bool sample_features = static_cast<int>(pool.size()) < s.p || params.node_feature_percent < 100;
    auto node_masks = [&](size_t n_nodes) {
        std::vector<unsigned char> masks;
        if (!sample_features)
            return masks;
        masks.assign(n_nodes * static_cast<size_t>(s.p), 0);
        for (size_t node = 0; node < n_nodes; ++node) {
            std::vector<int> chosen = pool;
            if (params.node_feature_percent < 100) {
                const int k = std::max(1, static_cast<int>(pool.size()) * std::max(1, params.node_feature_percent) / 100);
                std::shuffle(chosen.begin(), chosen.end(), rng);
                chosen.resize(static_cast<size_t>(k));
            }
            for (int f : chosen)
                masks[node * static_cast<size_t>(s.p) + static_cast<size_t>(f)] = 1;
        }
        return masks;
    };

    struct Work {
        GpuTreeNode node;
        int slot = -1;
        Candidate split{};
    };
    std::vector<Work> nodes;
    nodes.reserve(static_cast<size_t>(2 * max_leaves + 1));
    std::vector<int> free_slots;
    for (int slot = max_leaves; slot >= 0; --slot)
        free_slots.push_back(slot);
    auto take_slot = [&] {
        const int slot = free_slots.back();
        free_slots.pop_back();
        return slot;
    };

    const long long min_leaf = std::max(1, params.min_samples_leaf);
    auto can_split = [&](const GpuTreeNode& node) {
        if (params.max_depth > 0 && node.depth >= params.max_depth)
            return false;
        return node.count >= 2 * min_leaf;
    };
    auto valid = [](const Candidate& c) { return c.bin >= 0 && c.gain > 0.0; };

    struct Item {
        double gain;
        int node;
        bool operator<(const Item& o) const {
            if (gain != o.gain)
                return gain < o.gain;
            return node > o.node;  // earlier node first on ties
        }
    };
    std::priority_queue<Item> queue;

    // Root.
    Work root;
    root.node.count = s.active_rows;
    root.node.row_begin = 0;
    root.node.row_end = s.active_rows;
    root.slot = take_slot();
    s.build_histogram(root.slot, 0, s.active_rows);
    nodes.push_back(root);
    {
        const auto best = s.find_splits({root.slot}, params, node_masks(1));
        nodes[0].split = best[0];
        nodes[0].node.G = best[0].Gp;
        nodes[0].node.H = best[0].Hp;
        if (can_split(nodes[0].node) && valid(best[0]))
            queue.push({best[0].gain, 0});
    }

    int leaves = 1;
    while (!queue.empty() && leaves < max_leaves) {
        const int id = queue.top().node;
        queue.pop();
        Work& parent = nodes[static_cast<size_t>(id)];
        const Candidate sp = parent.split;
        const int begin = parent.node.row_begin, end = parent.node.row_end;
        const int n_left = static_cast<int>(sp.CL);
        s.partition(begin, end, sp.feature, sp.bin, sp.missing_left != 0);

        GpuTreeNode left, right;
        left.depth = right.depth = parent.node.depth + 1;
        left.row_begin = begin;
        left.row_end = begin + n_left;
        right.row_begin = begin + n_left;
        right.row_end = end;
        left.count = n_left;
        right.count = end - begin - n_left;
        left.G = sp.GL;
        left.H = sp.HL;
        right.G = parent.node.G - sp.GL;
        right.H = parent.node.H - sp.HL;

        parent.node.is_leaf = false;
        parent.node.feature = sp.feature;
        parent.node.threshold = sp.bin;
        parent.node.missing_left = sp.missing_left != 0;
        parent.node.gain = sp.gain;
        const int parent_slot = parent.slot;
        parent.slot = -1;
        const int left_id = static_cast<int>(nodes.size());
        const int right_id = left_id + 1;
        parent.node.left = left_id;
        parent.node.right = right_id;
        ++leaves;

        Work lw, rw;
        lw.node = left;
        rw.node = right;
        const bool more = leaves < max_leaves;
        const bool left_splittable = more && can_split(left);
        const bool right_splittable = more && can_split(right);
        if (left_splittable || right_splittable) {
            // Smaller child built directly, larger = parent - smaller.
            const bool left_small = left.count <= right.count;
            Work& small = left_small ? lw : rw;
            Work& large = left_small ? rw : lw;
            small.slot = take_slot();
            s.build_histogram(small.slot, small.node.row_begin, small.node.row_end);
            s.subtract(parent_slot, small.slot);
            large.slot = parent_slot;
            std::vector<int> slots;
            std::vector<Work*> evaluated;
            if (left_splittable) {
                slots.push_back(lw.slot);
                evaluated.push_back(&lw);
            }
            if (right_splittable) {
                slots.push_back(rw.slot);
                evaluated.push_back(&rw);
            }
            const auto best = s.find_splits(slots, params, node_masks(slots.size()));
            for (size_t k = 0; k < evaluated.size(); ++k)
                evaluated[k]->split = best[k];
            // Children that cannot split give their slot back.
            for (Work* w : {&lw, &rw}) {
                const bool splittable = (w == &lw) ? left_splittable : right_splittable;
                if (!splittable && w->slot >= 0) {
                    free_slots.push_back(w->slot);
                    w->slot = -1;
                }
            }
        } else {
            free_slots.push_back(parent_slot);
        }
        nodes.push_back(lw);
        nodes.push_back(rw);
        for (int child : {left_id, right_id}) {
            Work& w = nodes[static_cast<size_t>(child)];
            if (w.slot >= 0 && valid(w.split))
                queue.push({w.split.gain, child});
            else if (w.slot >= 0) {
                free_slots.push_back(w.slot);
                w.slot = -1;
            }
        }
    }

    std::vector<GpuTreeNode> out;
    out.reserve(nodes.size());
    for (auto& w : nodes)
        out.push_back(w.node);
    return out;
}

void GpuTreeTrainer::update_margins(const std::vector<GpuTreeNode>& nodes, const std::vector<double>& values,
                                    double scale, int class_index) {
    Impl& s = *impl_;
    if (class_index < 0 || class_index >= s.num_classes)
        throw std::invalid_argument("GpuTreeTrainer::update_margins: class_index out of range");
    float* margins = s.margins + static_cast<size_t>(class_index) * static_cast<size_t>(s.n);

    if (!s.last_all_rows) {
        // The tree saw a subsample: route every row through it.
        const size_t count = nodes.size();
        if (static_cast<int>(count) > s.t_capacity) {
            for (void* ptr : std::initializer_list<void*>{s.t_feature, s.t_threshold, s.t_missing_left, s.t_left,
                                                          s.t_right, s.t_delta})
                cudaFree(ptr);
            s.t_feature = device_alloc<int>(count);
            s.t_threshold = device_alloc<int>(count);
            s.t_missing_left = device_alloc<unsigned char>(count);
            s.t_left = device_alloc<int>(count);
            s.t_right = device_alloc<int>(count);
            s.t_delta = device_alloc<float>(count);
            s.t_capacity = static_cast<int>(count);
        }
        std::vector<int> feature(count), threshold(count), left(count), right(count);
        std::vector<unsigned char> missing_left(count);
        std::vector<float> delta(count);
        for (size_t i = 0; i < count; ++i) {
            feature[i] = nodes[i].is_leaf ? -1 : nodes[i].feature;
            threshold[i] = nodes[i].threshold;
            missing_left[i] = nodes[i].missing_left ? 1 : 0;
            left[i] = nodes[i].left;
            right[i] = nodes[i].right;
            delta[i] = nodes[i].is_leaf ? static_cast<float>(scale * values[i]) : 0.0F;
        }
        check(cudaMemcpy(s.t_feature, feature.data(), count * sizeof(int), cudaMemcpyHostToDevice), "copy tree");
        check(cudaMemcpy(s.t_threshold, threshold.data(), count * sizeof(int), cudaMemcpyHostToDevice), "copy tree");
        check(cudaMemcpy(s.t_missing_left, missing_left.data(), count, cudaMemcpyHostToDevice), "copy tree");
        check(cudaMemcpy(s.t_left, left.data(), count * sizeof(int), cudaMemcpyHostToDevice), "copy tree");
        check(cudaMemcpy(s.t_right, right.data(), count * sizeof(int), cudaMemcpyHostToDevice), "copy tree");
        check(cudaMemcpy(s.t_delta, delta.data(), count * sizeof(float), cudaMemcpyHostToDevice), "copy tree");
        if (s.wide)
            traverse_update_kernel<uint16_t><<<grid_for(s.n), 256>>>(
                static_cast<const uint16_t*>(s.codes), s.n, s.missing_codes, s.t_feature, s.t_threshold,
                s.t_missing_left, s.t_left, s.t_right, s.t_delta, margins);
        else
            traverse_update_kernel<uint8_t><<<grid_for(s.n), 256>>>(
                static_cast<const uint8_t*>(s.codes), s.n, s.missing_codes, s.t_feature, s.t_threshold,
                s.t_missing_left, s.t_left, s.t_right, s.t_delta, margins);
        check(cudaGetLastError(), "traversal kernel");
        return;
    }

    std::vector<std::pair<int, float>> leaves;
    for (size_t i = 0; i < nodes.size(); ++i)
        if (nodes[i].is_leaf && nodes[i].row_end > nodes[i].row_begin)
            leaves.emplace_back(nodes[i].row_begin, static_cast<float>(scale * values[i]));
    if (leaves.empty())
        return;
    std::sort(leaves.begin(), leaves.end());
    if (static_cast<int>(leaves.size()) > s.leaf_capacity) {
        cudaFree(s.leaf_begin);
        cudaFree(s.leaf_delta);
        s.leaf_capacity = static_cast<int>(leaves.size());
        s.leaf_begin = device_alloc<int>(leaves.size());
        s.leaf_delta = device_alloc<float>(leaves.size());
    }
    std::vector<int> begins(leaves.size());
    std::vector<float> deltas(leaves.size());
    for (size_t i = 0; i < leaves.size(); ++i) {
        begins[i] = leaves[i].first;
        deltas[i] = leaves[i].second;
    }
    check(cudaMemcpy(s.leaf_begin, begins.data(), begins.size() * sizeof(int), cudaMemcpyHostToDevice),
          "copy leaf begins");
    check(cudaMemcpy(s.leaf_delta, deltas.data(), deltas.size() * sizeof(float), cudaMemcpyHostToDevice),
          "copy leaf deltas");
    update_margins_kernel<<<grid_for(s.n), 256>>>(margins, s.rows, s.leaf_begin, s.leaf_delta,
                                                  static_cast<int>(leaves.size()), s.n);
    check(cudaGetLastError(), "margin kernel");
}

std::vector<double> GpuTreeTrainer::margins() const {
    const Impl& s = *impl_;
    const size_t K = static_cast<size_t>(s.num_classes);
    std::vector<float> host(static_cast<size_t>(s.n) * K);
    check(cudaMemcpy(host.data(), s.margins, host.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy margins");
    // Class-major on the device -> row-major [row][class].
    std::vector<double> out(host.size());
    for (size_t k = 0; k < K; ++k)
        for (size_t i = 0; i < static_cast<size_t>(s.n); ++i)
            out[i * K + k] = host[k * static_cast<size_t>(s.n) + i];
    return out;
}

}  // namespace foretree::cuda
