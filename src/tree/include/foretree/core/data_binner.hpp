// tree/include/foretree/core/data_binner.hpp
#pragma once
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

inline UniformMeta compute_uniform_meta(const std::vector<std::vector<double>>& edges, double tol = 1e-9) {
    const int P = static_cast<int>(edges.size());
    UniformMeta m;
    m.is_uniform.assign(P, 0);
    m.lo.assign(P, 0.0);
    m.invw.assign(P, 1.0);
    m.nb.assign(P, 0);

    for (int j = 0; j < P; ++j) {
        const auto& e = edges[j];
        const int nb = static_cast<int>(e.size()) - 1;
        m.nb[j] = std::max(nb, 0);
        if (nb <= 0)
            continue;

        double total = 0.0;
        for (int k = 0; k < nb; ++k)
            total += (e[k + 1] - e[k]);
        const double mean = total / nb;
        if (mean <= 0.0)
            continue;

        double max_dev = 0.0;
        for (int k = 0; k < nb; ++k) {
            const double dev = std::abs((e[k + 1] - e[k]) - mean);
            if (dev > max_dev)
                max_dev = dev;
        }
        const double threshold = tol * std::max(1.0, std::abs(mean));
        if (max_dev <= threshold) {
            m.is_uniform[j] = 1u;
            m.lo[j] = e.front();
            m.invw[j] = 1.0 / mean;
        }
    }
    return m;
}

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

// ============================================================================
// EdgeSet: per-mode edges with per-feature bin capacities
// ============================================================================

struct EdgeSet {
    // edges_per_feat[j] has size nb_j+1, strictly increasing.
    std::vector<std::vector<double>> edges_per_feat;

    // NEW: Per-feature number of finite bins
    std::vector<int> finite_bins_per_feat;
    // NEW: Per-feature missing bin IDs (== finite_bins_per_feat[j])
    std::vector<int> missing_bin_id_per_feat;
    std::vector<FeatureType> feature_types;

    // EXISTING: Mode-wise capacity (max finite bins across features in this
    // mode).
    int finite_bins = 256;
    // EXISTING: Reserved id for "missing" (== finite_bins).
    int missing_bin_id = 256;
};

// ============================================================================
// DataBinner
//  - Supports multiple "modes" (sets of edges).
//  - Per-node, per-feature overrides.
//  - Fast uniform-binning when edges are near-uniform.
//  - NOW: Different bin sizes per feature with proper range tracking.
//  - MAINTAINS: All existing method signatures and behavior
// ============================================================================

class DataBinner {
public:
    explicit DataBinner(int P) : P_(P) {
        if (P_ <= 0)
            throw std::invalid_argument("P must be positive");
    }

    // EXISTING SIGNATURE: Register edges for a mode with per-feature bin size
    // awareness
    void register_edges(const std::string& mode, EdgeSet e) {
        if (static_cast<int>(e.edges_per_feat.size()) != P_) {
            throw std::invalid_argument("EdgeSet.features != P");
        }
        for (auto& col : e.edges_per_feat)
            _strict_increasing(col);

        // NEW: Compute per-feature capacities
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

        // EXISTING: Set global mode capacities (backward compatibility)
        e.finite_bins = max_cap;
        e.missing_bin_id = max_cap;

        modes_[mode] = std::move(e);
        overrides_[mode].clear();
        uniform_[mode] = compute_uniform_meta(modes_[mode].edges_per_feat);
    }

    // EXISTING SIGNATURE: Node override that respects per-feature capacity
    // limits
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

        // EXISTING: Check against global max capacity
        if (nb > it->second.finite_bins) {
            throw std::invalid_argument("override exceeds mode capacity (finite_bins); re-register "
                                        "mode with larger capacity");
        }

        overrides_[mode][{node_id, feat}] = std::move(e);
    }

    // EXISTING SIGNATURE: Prebin with per-feature awareness under the hood
    std::pair<std::shared_ptr<std::vector<uint16_t>>, int> prebin(const double* X, int N, int P,
                                                                  const std::string& mode, int node_id = -1) const {
        const EdgeSet& edges = validate_prebin_(X, N, P, mode);
        auto codes = std::make_shared<std::vector<uint16_t>>(static_cast<size_t>(N) * static_cast<size_t>(P_));
        bin_into_(X, N, mode, node_id, edges, codes->data());
        return {codes, edges.missing_bin_id};
    }

    QuantizedDatasetPtr prebin_compact(const double* X, int N, int P, const std::string& mode, int node_id = -1) const {
        const EdgeSet& edges = validate_prebin_(X, N, P, mode);
        std::vector<uint16_t> missing_codes(static_cast<size_t>(P));
        for (int feature = 0; feature < P; ++feature) {
            missing_codes[static_cast<size_t>(feature)] = static_cast<uint16_t>(edges.missing_bin_id_per_feat[feature]);
        }
        const size_t size = static_cast<size_t>(N) * static_cast<size_t>(P);
        if (edges.missing_bin_id <= std::numeric_limits<uint8_t>::max()) {
            std::vector<uint8_t> codes(size);
            bin_into_(X, N, mode, node_id, edges, codes.data());
            return std::make_shared<QuantizedDataset>(
                QuantizedDataset::from_u8(N, P, std::move(codes), static_cast<uint8_t>(edges.missing_bin_id),
                                          std::move(missing_codes), edges.feature_types));
        }
        std::vector<uint16_t> codes(size);
        bin_into_(X, N, mode, node_id, edges, codes.data());
        return std::make_shared<QuantizedDataset>(
            QuantizedDataset::from_u16(N, P, std::move(codes), static_cast<uint16_t>(edges.missing_bin_id),
                                       std::move(missing_codes), edges.feature_types));
    }

    // EXISTING SIGNATURE: prebin into an existing buffer
    int prebin_into(const double* X, int N, int P, const std::string& mode, uint16_t* out_codes,
                    int node_id = -1) const {
        if (!out_codes)
            throw std::invalid_argument("out_codes is null");
        const EdgeSet& edges = validate_prebin_(X, N, P, mode);
        bin_into_(X, N, mode, node_id, edges, out_codes);
        return edges.missing_bin_id;
    }

    // ----------------------------------------------------------------------------
    // EXISTING + NEW Queries (overloaded for backward compatibility)
    // ----------------------------------------------------------------------------
    const EdgeSet* get_edgeset_(const std::string& mode) const {
        auto it = modes_.find(mode);
        return (it == modes_.end()) ? nullptr : &it->second;
    }

    // EXISTING: Get max finite bins across all features (backward
    // compatibility)
    int finite_bins(const std::string& mode) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es)
            throw std::invalid_argument("mode not registered");
        return es->finite_bins;
    }

    // NEW: Get finite bins for specific feature
    int finite_bins(const std::string& mode, int feat) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es)
            throw std::invalid_argument("mode not registered");
        if (feat < 0 || feat >= P_)
            throw std::out_of_range("feature index");
        return es->finite_bins_per_feat[feat];
    }

    // EXISTING: Get global missing bin ID (backward compatibility)
    int missing_bin_id(const std::string& mode) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es)
            throw std::invalid_argument("mode not registered");
        return es->missing_bin_id;
    }

    // NEW: Get missing bin ID for specific feature
    int missing_bin_id(const std::string& mode, int feat) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es)
            throw std::invalid_argument("mode not registered");
        if (feat < 0 || feat >= P_)
            throw std::out_of_range("feature index");
        return es->missing_bin_id_per_feat[feat];
    }

    // EXISTING: Get max total bins (backward compatibility)
    int total_bins(const std::string& mode) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es)
            throw std::invalid_argument("mode not registered");
        return es->finite_bins + 1; // include reserved missing
    }

    // NEW: Get total bins for specific feature
    int total_bins(const std::string& mode, int feat) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es)
            throw std::invalid_argument("mode not registered");
        if (feat < 0 || feat >= P_)
            throw std::out_of_range("feature index");
        return es->finite_bins_per_feat[feat] + 1; // include missing
    }

    // NEW: Get all finite bin counts for a mode
    const std::vector<int>& finite_bins_per_feat(const std::string& mode) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es)
            throw std::invalid_argument("mode not registered");
        return es->finite_bins_per_feat;
    }

    // NEW: Get all missing bin IDs for a mode
    const std::vector<int>& missing_bin_ids_per_feat(const std::string& mode) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es)
            throw std::invalid_argument("mode not registered");
        return es->missing_bin_id_per_feat;
    }

    // EXISTING
    int P() const {
        return P_;
    }

private:
    struct EffectiveBins {
        std::vector<const std::vector<double>*> edges;
        std::vector<uint8_t> uniform;
        std::vector<double> lo;
        std::vector<double> inverse_width;
        std::vector<int> finite_bins;
        std::vector<int> missing_codes;
    };

    const EdgeSet& validate_prebin_(const double* X, int N, int P, const std::string& mode) const {
        if (!X)
            throw std::invalid_argument("X is null");
        if (N < 0)
            throw std::invalid_argument("X rows must be non-negative");
        if (P != P_)
            throw std::invalid_argument("X columns != P");
        const EdgeSet* edges = get_edgeset_(mode);
        if (!edges)
            throw std::invalid_argument("mode not registered");
        return *edges;
    }

    EffectiveBins effective_bins_(const std::string& mode, int node_id, const EdgeSet& defaults) const {
        const UniformMeta& metadata = uniform_.at(mode);
        EffectiveBins result;
        result.edges.resize(static_cast<size_t>(P_));
        result.uniform = metadata.is_uniform;
        result.lo = metadata.lo;
        result.inverse_width = metadata.invw;
        result.finite_bins = metadata.nb;
        result.missing_codes = defaults.missing_bin_id_per_feat;
        for (int feature = 0; feature < P_; ++feature) {
            result.edges[static_cast<size_t>(feature)] = &defaults.edges_per_feat[static_cast<size_t>(feature)];
            const auto* override_edges = get_override_edges_(mode, node_id, feature);
            if (!override_edges)
                continue;
            result.edges[static_cast<size_t>(feature)] = override_edges;
            compute_uniform_meta_single(
                *override_edges, result.uniform[static_cast<size_t>(feature)], result.lo[static_cast<size_t>(feature)],
                result.inverse_width[static_cast<size_t>(feature)], result.finite_bins[static_cast<size_t>(feature)]);
            result.missing_codes[static_cast<size_t>(feature)] = result.finite_bins[static_cast<size_t>(feature)];
        }
        return result;
    }

    template <class Code>
    void bin_into_(const double* X, int N, const std::string& mode, int node_id, const EdgeSet& defaults,
                   Code* output) const {
        const bool has_overrides = node_id >= 0 && has_any_override_(mode, node_id);
        const UniformMeta& default_metadata = uniform_.at(mode);
        EffectiveBins overridden;
        if (has_overrides)
            overridden = effective_bins_(mode, node_id, defaults);

        for (int row = 0; row < N; ++row) {
            const size_t base = static_cast<size_t>(row) * static_cast<size_t>(P_);
            for (int feature = 0; feature < P_; ++feature) {
                const size_t f = static_cast<size_t>(feature);
                const auto& edges = has_overrides ? *overridden.edges[f] : defaults.edges_per_feat[f];
                const int finite_bins = has_overrides ? overridden.finite_bins[f] : default_metadata.nb[f];
                const int missing_code =
                    has_overrides ? overridden.missing_codes[f] : defaults.missing_bin_id_per_feat[f];
                const bool uniform = has_overrides ? overridden.uniform[f] != 0 : default_metadata.is_uniform[f] != 0;
                const double lo = has_overrides ? overridden.lo[f] : default_metadata.lo[f];
                const double inverse_width = has_overrides ? overridden.inverse_width[f] : default_metadata.invw[f];
                const double value = X[base + f];
                int code = 0;
                if (!std::isfinite(value)) {
                    code = missing_code;
                } else if (finite_bins > 0 && uniform) {
                    code = static_cast<int>((value - lo) * inverse_width);
                    code = std::clamp(code, 0, finite_bins - 1);
                } else if (finite_bins > 0) {
                    if (value >= edges[static_cast<size_t>(finite_bins)]) {
                        code = finite_bins - 1;
                    } else if (value >= edges.front()) {
                        const auto it = std::upper_bound(edges.begin() + 1, edges.begin() + finite_bins, value);
                        code = static_cast<int>(it - edges.begin()) - 1;
                    }
                }
                output[base + f] = static_cast<Code>(code);
            }
        }
    }

    struct Key {
        int nid, feat;
        bool operator==(const Key& o) const {
            return nid == o.nid && feat == o.feat;
        }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const noexcept {
            // A simple 64-bit mix; good enough for (nid,feat).
            uint64_t x = (uint64_t)static_cast<uint32_t>(k.nid);
            uint64_t y = (uint64_t)static_cast<uint32_t>(k.feat);
            uint64_t z = (x << 32) ^ (y * 0x9E3779B97F4A7C15ull);
            z ^= (z >> 33);
            z *= 0xff51afd7ed558ccdULL;
            z ^= (z >> 33);
            z *= 0xc4ceb9fe1a85ec53ULL;
            z ^= (z >> 33);
            return (size_t)z;
        }
    };

    bool has_any_override_(const std::string& mode, int node_id) const {
        auto it = overrides_.find(mode);
        if (it == overrides_.end())
            return false;
        for (const auto& kv : it->second)
            if (kv.first.nid == node_id)
                return true;
        return false;
    }
    const std::vector<double>* get_edges_ptr_(const std::string& mode, int node_id, int feat) const {
        if (feat < 0 || feat >= P_)
            throw std::out_of_range("feature index");
        if (node_id >= 0) {
            auto it = overrides_.find(mode);
            if (it != overrides_.end()) {
                Key k{node_id, feat};
                auto jt = it->second.find(k);
                if (jt != it->second.end())
                    return &jt->second;
            }
        }
        auto it = modes_.find(mode);
        if (it == modes_.end())
            return nullptr;
        return &it->second.edges_per_feat[feat];
    }

    const std::vector<double>* get_override_edges_(const std::string& mode, int node_id, int feat) const {
        if (node_id < 0)
            return nullptr;
        auto mode_it = overrides_.find(mode);
        if (mode_it == overrides_.end())
            return nullptr;
        const auto it = mode_it->second.find(Key{node_id, feat});
        return it == mode_it->second.end() ? nullptr : &it->second;
    }

private:
    int P_ = 0;

    std::unordered_map<std::string, EdgeSet> modes_;
    std::unordered_map<std::string, std::unordered_map<Key, std::vector<double>, KeyHash>> overrides_;
    std::unordered_map<std::string, UniformMeta> uniform_;
};

} // namespace foretree
