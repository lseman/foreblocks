// tree/include/foretree/core/data_binner.hpp
#pragma once

#include "foretree/core/parallel_executor.hpp"
#include <algorithm>
#include <cassert>
#include <cmath> // nextafter, isfinite, abs
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "foretree/core/dataset.hpp"

namespace foretree {

// ============================================================================
// Helpers
// ============================================================================

// Ensure strictly increasing edges (double64) with nextafter tie-breaking.
// If size<2, fall back to [0,1] to avoid edge cases downstream.
inline void _strict_increasing(std::vector<double>& e) {
    if (e.size() < 2) {
        e = {0.0, 1.0};
        return;
    }
    for (size_t i = 1; i < e.size(); ++i) {
        if (!(e[i] > e[i - 1])) {
            const double prev = e[i - 1];
            double next = std::nextafter(prev, std::numeric_limits<double>::infinity());
            if (next == prev)
                next = prev + std::numeric_limits<double>::epsilon();
            e[i] = next;
        }
    }
}

// Per-feature uniform bin metadata (lo, 1/width) for O(1) binning.
struct UniformMeta {
    std::vector<uint8_t> is_uniform; // 0/1
    std::vector<double> lo;          // e[0]
    std::vector<double> invw;        // 1/mean_step
    std::vector<int> nb;             // per-feature #finite bins
};

// Compute uniform meta for a single edge vector (small, used for overrides).
inline void compute_uniform_meta_single(const std::vector<double>& e, uint8_t& is_uniform, double& lo, double& invw,
                                        int& nb, double tol = 1e-9) {
    const int n = static_cast<int>(e.size());
    nb = std::max(0, n - 1);
    is_uniform = 0u;
    lo = 0.0;
    invw = 1.0;
    if (nb <= 0)
        return;

    double total = 0.0;
    for (int k = 0; k < nb; ++k)
        total += (e[k + 1] - e[k]);
    const double mean = total / nb;
    if (mean <= 0.0)
        return;

    double max_dev = 0.0;
    for (int k = 0; k < nb; ++k) {
        const double dev = std::abs((e[k + 1] - e[k]) - mean);
        if (dev > max_dev)
            max_dev = dev;
    }
    const double threshold = tol * std::max(1.0, std::abs(mean));
    if (max_dev <= threshold) {
        is_uniform = 1u;
        lo = e.front();
        invw = 1.0 / mean;
    }
}

inline UniformMeta compute_uniform_meta(const std::vector<std::vector<double>>& edges, double tol = 1e-9) {
    UniformMeta metadata;
    metadata.is_uniform.resize(edges.size());
    metadata.lo.resize(edges.size());
    metadata.invw.resize(edges.size());
    metadata.nb.resize(edges.size());
    for (size_t f = 0; f < edges.size(); ++f)
        compute_uniform_meta_single(edges[f], metadata.is_uniform[f], metadata.lo[f], metadata.invw[f],
                                    metadata.nb[f], tol);
    return metadata;
}

// ============================================================================
// EdgeSet: per-mode edges with per-feature bin capacities
// ============================================================================

struct EdgeSet {
    // edges_per_feat[j] has size nb_j+1, strictly increasing.
    std::vector<std::vector<double>> edges_per_feat;

    // Per-feature number of finite bins
    std::vector<int> finite_bins_per_feat;
    // Per-feature missing bin IDs (== finite_bins_per_feat[j])
    std::vector<int> missing_bin_id_per_feat;
    std::vector<FeatureType> feature_types;

    // Mode-wise capacity (max finite bins across features in this
    // mode).
    int finite_bins = 256;
    // Reserved id for "missing" (== finite_bins).
    int missing_bin_id = 256;
};

// ============================================================================
// DataBinner registers per-feature edges by mode. Individual nodes can
// override those edges within the mode's maximum code capacity.
// ============================================================================

class DataBinner {
    using FeatureOverrides = std::unordered_map<int, std::vector<double>>;
    struct Mode {
        EdgeSet defaults;
        UniformMeta uniform;
        std::unordered_map<int, FeatureOverrides> overrides;
    };

public:
    explicit DataBinner(int P) : P_(P) {
        if (P_ <= 0)
            throw std::invalid_argument("P must be positive");
    }

    // Re-registering a mode replaces its edges and clears its node overrides.
    void register_edges(const std::string& mode, EdgeSet e) {
        if (static_cast<int>(e.edges_per_feat.size()) != P_) {
            throw std::invalid_argument("EdgeSet.features != P");
        }
        for (auto& col : e.edges_per_feat)
            _strict_increasing(col);

        // Compute per-feature capacities
        e.finite_bins_per_feat.resize(P_);
        e.missing_bin_id_per_feat.resize(P_);
        if (e.feature_types.empty())
            e.feature_types.assign(static_cast<size_t>(P_), FeatureType::Numerical);
        if (static_cast<int>(e.feature_types.size()) != P_)
            throw std::invalid_argument("EdgeSet.feature_types size must match P");
        int max_cap = 0;

        for (int j = 0; j < P_; ++j) {
            const int nb = static_cast<int>(e.edges_per_feat[j].size()) - 1;
            if (nb < 0)
                throw std::invalid_argument("edges must have size >= 2");

            e.finite_bins_per_feat[j] = nb;
            e.missing_bin_id_per_feat[j] = nb; // missing id = number of finite bins
            max_cap = std::max(max_cap, nb);
        }

        if (max_cap < 0 || max_cap > std::numeric_limits<uint16_t>::max()) {
            throw std::invalid_argument("finite_bins exceeds uint16_t capacity");
        }

        // Set global mode capacities (backward compatibility)
        e.finite_bins = max_cap;
        e.missing_bin_id = max_cap;

        auto uniform = compute_uniform_meta(e.edges_per_feat);
        modes_.insert_or_assign(mode, Mode{std::move(e), std::move(uniform), {}});
    }

    void set_node_override(const std::string& mode, int node_id, int feat, const std::vector<double>& edges) {
        if (feat < 0 || feat >= P_)
            throw std::out_of_range("feature index");
        auto it = modes_.find(mode);
        if (it == modes_.end())
            throw std::invalid_argument("mode not registered");

        std::vector<double> e = edges;
        _strict_increasing(e);
        const int nb = static_cast<int>(e.size()) - 1;
        if (nb <= 0) {
            throw std::invalid_argument("override edges must have len >= 2");
        }

        // Check against global max capacity
        if (nb > it->second.defaults.finite_bins) {
            throw std::invalid_argument("override exceeds mode capacity (finite_bins); re-register "
                                        "mode with larger capacity");
        }

        it->second.overrides[node_id][feat] = std::move(e);
    }

    std::pair<std::shared_ptr<std::vector<uint16_t>>, int> prebin(const double* X, int N, int P,
                                                                  const std::string& mode, int node_id = -1) const {
        const Mode& registered = validate_prebin_(X, N, P, mode);
        auto codes = std::make_shared<std::vector<uint16_t>>(static_cast<size_t>(N) * static_cast<size_t>(P_));
        bin_into_(X, N, registered, node_id, codes->data());
        return {codes, registered.defaults.missing_bin_id};
    }

    QuantizedDatasetPtr prebin_compact(const double* X, int N, int P, const std::string& mode, int node_id = -1) const {
        const Mode& registered = validate_prebin_(X, N, P, mode);
        const EdgeSet& edges = registered.defaults;
        std::vector<uint16_t> missing_codes;
        const size_t size = static_cast<size_t>(N) * static_cast<size_t>(P);
        if (edges.missing_bin_id <= std::numeric_limits<uint8_t>::max()) {
            std::vector<uint8_t> codes(size);
            bin_into_(X, N, registered, node_id, codes.data(), &missing_codes);
            return std::make_shared<QuantizedDataset>(
                QuantizedDataset::from_u8(N, P, std::move(codes), static_cast<uint8_t>(edges.missing_bin_id),
                                          std::move(missing_codes), edges.feature_types));
        }
        std::vector<uint16_t> codes(size);
        bin_into_(X, N, registered, node_id, codes.data(), &missing_codes);
        return std::make_shared<QuantizedDataset>(
            QuantizedDataset::from_u16(N, P, std::move(codes), static_cast<uint16_t>(edges.missing_bin_id),
                                       std::move(missing_codes), edges.feature_types));
    }

    int prebin_into(const double* X, int N, int P, const std::string& mode, uint16_t* out_codes,
                    int node_id = -1) const {
        if (!out_codes)
            throw std::invalid_argument("out_codes is null");
        const Mode& registered = validate_prebin_(X, N, P, mode);
        bin_into_(X, N, registered, node_id, out_codes);
        return registered.defaults.missing_bin_id;
    }

    const EdgeSet* get_edgeset_(const std::string& mode) const {
        auto it = modes_.find(mode);
        return (it == modes_.end()) ? nullptr : &it->second.defaults;
    }

    // Get max finite bins across all features (backward
    // compatibility)
    int finite_bins(const std::string& mode) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es)
            throw std::invalid_argument("mode not registered");
        return es->finite_bins;
    }

    // Get finite bins for specific feature
    int finite_bins(const std::string& mode, int feat) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es)
            throw std::invalid_argument("mode not registered");
        if (feat < 0 || feat >= P_)
            throw std::out_of_range("feature index");
        return es->finite_bins_per_feat[feat];
    }

    // Get global missing bin ID (backward compatibility)
    int missing_bin_id(const std::string& mode) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es)
            throw std::invalid_argument("mode not registered");
        return es->missing_bin_id;
    }

    // Get missing bin ID for specific feature
    int missing_bin_id(const std::string& mode, int feat) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es)
            throw std::invalid_argument("mode not registered");
        if (feat < 0 || feat >= P_)
            throw std::out_of_range("feature index");
        return es->missing_bin_id_per_feat[feat];
    }

    // Get max total bins (backward compatibility)
    int total_bins(const std::string& mode) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es)
            throw std::invalid_argument("mode not registered");
        return es->finite_bins + 1; // include reserved missing
    }

    // Get total bins for specific feature
    int total_bins(const std::string& mode, int feat) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es)
            throw std::invalid_argument("mode not registered");
        if (feat < 0 || feat >= P_)
            throw std::out_of_range("feature index");
        return es->finite_bins_per_feat[feat] + 1; // include missing
    }

    // Get all finite bin counts for a mode
    const std::vector<int>& finite_bins_per_feat(const std::string& mode) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es)
            throw std::invalid_argument("mode not registered");
        return es->finite_bins_per_feat;
    }

    // Get all missing bin IDs for a mode
    const std::vector<int>& missing_bin_ids_per_feat(const std::string& mode) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es)
            throw std::invalid_argument("mode not registered");
        return es->missing_bin_id_per_feat;
    }

    int P() const {
        return P_;
    }

private:
    const Mode& validate_prebin_(const double* X, int N, int P, const std::string& mode) const {
        if (!X)
            throw std::invalid_argument("X is null");
        if (N < 0)
            throw std::invalid_argument("X rows must be non-negative");
        if (P != P_)
            throw std::invalid_argument("X columns != P");
        const auto it = modes_.find(mode);
        if (it == modes_.end())
            throw std::invalid_argument("mode not registered");
        return it->second;
    }

    static int encode_value_(double value, const std::vector<double>& edges, const UniformMeta& metadata, size_t f) {
        const int bins = metadata.nb[f];
        if (!std::isfinite(value))
            return bins;
        if (bins == 1 || !(value >= edges[1]))
            return 0;
        if (value >= edges[bins - 1])
            return bins - 1;

        // Verify the estimate against the actual cuts before using it.
        if (metadata.is_uniform[f]) {
            const double estimate = (value - metadata.lo[f]) * metadata.invw[f];
            if (estimate >= 0 && estimate < bins) {
                const int code = static_cast<int>(estimate);
                if (edges[code] <= value && value < edges[code + 1])
                    return code;
            }
        }
        // Keep the bin in [first, first + count). Both outcomes retain the
        // same window length so each iteration can select its start directly.
        int first = 0;
        for (int count = bins; count > 1;) {
            const int half = count / 2;
            const int middle = first + half;
            first = value >= edges[middle] ? middle : first;
            count -= half;
        }
        return first;
    }

    template <class Code>
    void bin_into_(const double* X, int N, const Mode& mode, int node_id, Code* output,
                   std::vector<uint16_t>* missing_codes = nullptr) const {
        const auto node = node_id < 0 ? mode.overrides.end() : mode.overrides.find(node_id);
        UniformMeta local;
        std::vector<const std::vector<double>*> local_edges;
        if (node != mode.overrides.end()) {
            local = mode.uniform;
            local_edges.reserve(P_);
            for (const auto& edges : mode.defaults.edges_per_feat)
                local_edges.push_back(&edges);
            for (const auto& [f, edges] : node->second) {
                local_edges[f] = &edges;
                compute_uniform_meta_single(edges, local.is_uniform[f], local.lo[f], local.invw[f], local.nb[f]);
            }
        }
        const auto& metadata = local_edges.empty() ? mode.uniform : local;
        if (missing_codes)
            missing_codes->assign(metadata.nb.begin(), metadata.nb.end());

        auto encode_rows = [&](int row_begin, int row_end) {
            for (int row = row_begin; row < row_end; ++row) {
                const size_t base = static_cast<size_t>(row) * P_;
                for (int f = 0; f < P_; ++f) {
                    const auto& edges = local_edges.empty() ? mode.defaults.edges_per_feat[f] : *local_edges[f];
                    output[base + f] = static_cast<Code>(encode_value_(X[base + f], edges, metadata, f));
                }
            }
        };
        const int64_t cells = static_cast<int64_t>(N) * P_;
        if (cells >= (1 << 18))
            default_parallel_executor()->parallel_for(0, N, std::max(256, (1 << 16) / std::max(1, P_)), encode_rows);
        else
            encode_rows(0, N);
    }

    int P_ = 0;
    std::unordered_map<std::string, Mode> modes_;
};

} // namespace foretree
