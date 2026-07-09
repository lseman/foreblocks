// tree/include/foretree/core/data_binner.hpp
#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>  // nextafter, isfinite, abs
#include <cstdint>
#include <limits>
#include <memory>  // shared_ptr
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
            double next =
                std::nextafter(prev, std::numeric_limits<double>::infinity());
            if (next == prev)
                next = prev + std::numeric_limits<double>::epsilon();
            e[i] = next;
        }
    }
}

// Per-feature uniform bin metadata (lo, 1/width) for O(1) binning.
struct UniformMeta {
    std::vector<uint8_t> is_uniform;  // 0/1
    std::vector<double> lo;           // e[0]
    std::vector<double> invw;         // 1/mean_step
    std::vector<int> nb;              // per-feature #finite bins
};

inline UniformMeta compute_uniform_meta(
    const std::vector<std::vector<double>>& edges, double tol = 1e-9) {
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
        if (nb <= 0) continue;

        double total = 0.0;
        for (int k = 0; k < nb; ++k) total += (e[k + 1] - e[k]);
        const double mean = total / nb;
        if (mean <= 0.0) continue;

        double max_dev = 0.0;
        for (int k = 0; k < nb; ++k) {
            const double dev = std::abs((e[k + 1] - e[k]) - mean);
            if (dev > max_dev) max_dev = dev;
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
inline void compute_uniform_meta_single(const std::vector<double>& e,
                                        uint8_t& is_uniform, double& lo,
                                        double& invw, int& nb,
                                        double tol = 1e-9) {
    const int n = static_cast<int>(e.size());
    nb = std::max(0, n - 1);
    is_uniform = 0u;
    lo = 0.0;
    invw = 1.0;
    if (nb <= 0) return;

    double total = 0.0;
    for (int k = 0; k < nb; ++k) total += (e[k + 1] - e[k]);
    const double mean = total / nb;
    if (mean <= 0.0) return;

    double max_dev = 0.0;
    for (int k = 0; k < nb; ++k) {
        const double dev = std::abs((e[k + 1] - e[k]) - mean);
        if (dev > max_dev) max_dev = dev;
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
        if (P_ <= 0) throw std::invalid_argument("P must be positive");
    }

    // EXISTING SIGNATURE: Register edges for a mode with per-feature bin size
    // awareness
    void register_edges(const std::string& mode, EdgeSet e) {
        if (static_cast<int>(e.edges_per_feat.size()) != P_) {
            throw std::invalid_argument("EdgeSet.features != P");
        }
        for (auto& col : e.edges_per_feat) _strict_increasing(col);

        // NEW: Compute per-feature capacities
        e.finite_bins_per_feat.resize(P_);
        e.missing_bin_id_per_feat.resize(P_);
        int max_cap = 0;

        for (int j = 0; j < P_; ++j) {
            const int nb = static_cast<int>(e.edges_per_feat[j].size()) - 1;
            if (nb < 0)
                throw std::invalid_argument("edges must have size >= 2");

            e.finite_bins_per_feat[j] = nb;
            e.missing_bin_id_per_feat[j] =
                nb;  // missing id = number of finite bins
            max_cap = std::max(max_cap, nb);
        }

        if (max_cap < 0 || max_cap > std::numeric_limits<uint16_t>::max()) {
            throw std::invalid_argument(
                "finite_bins exceeds uint16_t capacity");
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
    void set_node_override(const std::string& mode, int node_id, int feat,
                           const std::vector<double>& edges) {
        if (feat < 0 || feat >= P_) throw std::out_of_range("feature index");
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
            throw std::invalid_argument(
                "override exceeds mode capacity (finite_bins); re-register "
                "mode with larger capacity");
        }

        overrides_[mode][{node_id, feat}] = std::move(e);
    }

    // EXISTING SIGNATURE: Prebin with per-feature awareness under the hood
    std::pair<std::shared_ptr<std::vector<uint16_t>>, int> prebin(
        const double* X, int N, int P, const std::string& mode,
        int node_id = -1) const {
        if (!X) throw std::invalid_argument("X is null");
        if (P != P_) throw std::invalid_argument("X columns != P");
        const EdgeSet* ES = get_edgeset_(mode);
        if (!ES) throw std::invalid_argument("mode not registered");

        const int miss =
            ES->missing_bin_id;  // EXISTING: Use global missing bin id
        const bool has_over =
            (node_id >= 0) && has_any_override_(mode, node_id);

        // Fast per-feature effective metadata prepared once before scanning
        // rows.
        const UniformMeta& U = uniform_.at(mode);

        std::vector<const std::vector<double>*> Eptr;
        Eptr.resize(P_, nullptr);
        std::vector<uint8_t> uni;
        uni.resize(P_, 0);
        std::vector<double> lo;
        lo.resize(P_, 0.0);
        std::vector<double> invw;
        invw.resize(P_, 1.0);
        std::vector<int> nb;
        nb.resize(P_, 0);

        // NEW: Track per-feature missing IDs for proper range checking
        std::vector<int> feat_miss;
        feat_miss.resize(P_, miss);

        for (int j = 0; j < P_; ++j) {
            const std::vector<double>* e_over =
                get_edges_ptr_(mode, node_id, j);
            if (has_over && e_over) {
                Eptr[j] = e_over;
                compute_uniform_meta_single(*e_over, uni[j], lo[j], invw[j],
                                            nb[j]);
                feat_miss[j] = nb[j];  // NEW: Per-feature missing for overrides
            } else {
                Eptr[j] = &ES->edges_per_feat[j];
                uni[j] = U.is_uniform[j];
                lo[j] = U.lo[j];
                invw[j] = U.invw[j];
                nb[j] = U.nb[j];
                feat_miss[j] =
                    ES->missing_bin_id_per_feat[j];  // NEW: Per-feature missing
            }
        }

        auto codes = std::make_shared<std::vector<uint16_t>>(
            static_cast<size_t>(N) * static_cast<size_t>(P_));
        auto* out = codes->data();

        // EXISTING: Tight inner loops; avoid repeated size_t casts.
        for (int i = 0; i < N; ++i) {
            const double* row = X + (size_t)i * (size_t)P_;
            const size_t base = (size_t)i * (size_t)P_;
            for (int j = 0; j < P_; ++j) {
                const std::vector<double>& e = *Eptr[j];
                const int nbj = nb[j];

                uint16_t code;
                const double v = row[j];
                if (!std::isfinite(v)) {
                    // NEW: Use per-feature missing ID, but constrain to global
                    // range for compatibility
                    code = static_cast<uint16_t>(std::min(feat_miss[j], miss));
                } else if (nbj <= 0) {
                    code = 0;  // degenerate feature (shouldn't happen after
                               // register, but safe)
                } else if (uni[j]) {
                    // Uniform binning: k = clamp(floor((v - lo)*invw), [0,
                    // nb-1])
                    int k = static_cast<int>((v - lo[j]) * invw[j]);
                    if (k < 0)
                        k = 0;
                    else if (k >= nbj)
                        k = nbj - 1;
                    code = static_cast<uint16_t>(k);
                } else {
                    // Non-uniform: binary search in edges.
                    // Edges are strictly increasing; e size == nb+1.
                    if (v < e.front()) {
                        code = 0;
                    } else if (v >= e[(size_t)nbj]) {
                        code = static_cast<uint16_t>(nbj - 1);
                    } else {
                        // Find first edge strictly greater than v among
                        // (e[1..nb-1]). upper_bound returns iterator to first
                        // element > v. Index-1 gives bin.
                        auto it =
                            std::upper_bound(e.begin() + 1, e.begin() + nbj, v);
                        const int bin = static_cast<int>((it - e.begin()) - 1);
                        code = static_cast<uint16_t>(bin);
                    }
                }
                out[base + (size_t)j] = code;
            }
        }
        return {codes, miss};  // EXISTING: Return global missing bin id
    }

    // EXISTING SIGNATURE: prebin into an existing buffer
    int prebin_into(const double* X, int N, int P, const std::string& mode,
                    uint16_t* out_codes, int node_id = -1) const {
        auto pair = prebin(X, N, P, mode, node_id);
        std::copy(pair.first->begin(), pair.first->end(), out_codes);
        return pair.second;
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
        if (!es) throw std::invalid_argument("mode not registered");
        return es->finite_bins;
    }

    // NEW: Get finite bins for specific feature
    int finite_bins(const std::string& mode, int feat) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es) throw std::invalid_argument("mode not registered");
        if (feat < 0 || feat >= P_) throw std::out_of_range("feature index");
        return es->finite_bins_per_feat[feat];
    }

    // EXISTING: Get global missing bin ID (backward compatibility)
    int missing_bin_id(const std::string& mode) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es) throw std::invalid_argument("mode not registered");
        return es->missing_bin_id;
    }

    // NEW: Get missing bin ID for specific feature
    int missing_bin_id(const std::string& mode, int feat) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es) throw std::invalid_argument("mode not registered");
        if (feat < 0 || feat >= P_) throw std::out_of_range("feature index");
        return es->missing_bin_id_per_feat[feat];
    }

    // EXISTING: Get max total bins (backward compatibility)
    int total_bins(const std::string& mode) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es) throw std::invalid_argument("mode not registered");
        return es->finite_bins + 1;  // include reserved missing
    }

    // NEW: Get total bins for specific feature
    int total_bins(const std::string& mode, int feat) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es) throw std::invalid_argument("mode not registered");
        if (feat < 0 || feat >= P_) throw std::out_of_range("feature index");
        return es->finite_bins_per_feat[feat] + 1;  // include missing
    }

    // NEW: Get all finite bin counts for a mode
    const std::vector<int>& finite_bins_per_feat(
        const std::string& mode) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es) throw std::invalid_argument("mode not registered");
        return es->finite_bins_per_feat;
    }

    // NEW: Get all missing bin IDs for a mode
    const std::vector<int>& missing_bin_ids_per_feat(
        const std::string& mode) const {
        const EdgeSet* es = get_edgeset_(mode);
        if (!es) throw std::invalid_argument("mode not registered");
        return es->missing_bin_id_per_feat;
    }

    // EXISTING
    int P() const { return P_; }

   private:
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
        if (it == overrides_.end()) return false;
        for (const auto& kv : it->second)
            if (kv.first.nid == node_id) return true;
        return false;
    }
    const std::vector<double>* get_edges_ptr_(const std::string& mode,
                                              int node_id, int feat) const {
        if (feat < 0 || feat >= P_) throw std::out_of_range("feature index");
        if (node_id >= 0) {
            auto it = overrides_.find(mode);
            if (it != overrides_.end()) {
                Key k{node_id, feat};
                auto jt = it->second.find(k);
                if (jt != it->second.end()) return &jt->second;
            }
        }
        auto it = modes_.find(mode);
        if (it == modes_.end()) return nullptr;
        return &it->second.edges_per_feat[feat];
    }

   private:
    int P_ = 0;

    std::unordered_map<std::string, EdgeSet> modes_;
    std::unordered_map<std::string,
                       std::unordered_map<Key, std::vector<double>, KeyHash>>
        overrides_;
    std::unordered_map<std::string, UniformMeta> uniform_;
};

}  // namespace foretree
