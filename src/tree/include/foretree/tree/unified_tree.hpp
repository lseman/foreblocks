#pragma once
#include "foretree/core/dataset.hpp"
#include "foretree/core/gradient_hist_system.hpp"
#include "foretree/core/histogram_kernel.hpp"
#include "foretree/core/parallel_executor.hpp"
#ifdef FORETREE_HAS_CUDA
#    include "foretree/gpu/cuda_histogram.hpp"
#endif
#include "foretree/split/split_finder.hpp"
#include "foretree/tree/growth_policy.hpp"
#include "foretree/tree/packed_tree_builder.hpp"
#include "foretree/tree/row_partitioner.hpp"
#include "foretree/tree/training_context.hpp"
#include "foretree/tree/tree_types.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <ranges>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace foretree {

// --------------------------------- UnifiedTree --------------------------------
class UnifiedTree {
public:
    friend struct ExactProvider;
    NeuralLeafConfig neural_leaf_cfg_;
    void set_neural_leaf_config(const NeuralLeafConfig& cfg) {
        neural_leaf_cfg_ = cfg;
    }
#ifdef FORETREE_HAS_CUDA
    void set_cuda_histogram_engine(cuda::CudaHistogramEngine* engine) {
        cuda_histogram_engine_ = engine;
    }
#endif

    explicit UnifiedTree(TreeConfig cfg = {}, const GradientHistogramSystem* ghs = nullptr,
                         std::shared_ptr<ParallelExecutor> executor = {})
        : cfg_(std::move(cfg)), ghs_(ghs), executor_(executor ? std::move(executor) : default_parallel_executor()),
          rng_(cfg_.rng_seed) {
        validate_config_();
    }

    void set_raw_matrix(const double* Xraw, const uint8_t* miss_mask_or_null) {
        Xraw_ = Xraw;
        Xmiss_ = miss_mask_or_null;
    }

    void set_raw_for_neural(const double* Xraw, const uint8_t* Xmiss) {
        Xraw_eval_ = Xraw;
        Xmiss_eval_ = Xmiss;
    }

    void fit(const std::vector<uint16_t>& Xb, int N, int P, const std::vector<double>& g,
             const std::vector<double>& h) {
        if (N <= 0 || P <= 0) {
            throw std::invalid_argument("UnifiedTree::fit: N and P must be positive");
        }
        std::vector<int> all(static_cast<size_t>(N));
        std::iota(all.begin(), all.end(), 0);
        fit_with_row_ids(Xb, N, P, g, h, all);
    }

    void fit(const QuantizedDataset& Xb, const std::vector<double>& g, const std::vector<double>& h) {
        std::vector<int> all(static_cast<size_t>(Xb.rows()));
        std::iota(all.begin(), all.end(), 0);
        fit_with_row_ids(Xb, g, h, all);
    }

    void fit_with_row_ids(const QuantizedDataset& Xb, const std::vector<double>& g, const std::vector<double>& h,
                          const std::vector<int>& root_rows) {
        fit_with_row_ids_impl_(Xb, Xb.rows(), Xb.features(), g, h, root_rows);
    }

    void fit_with_row_ids(const std::vector<uint16_t>& Xb, int N, int P, const std::vector<double>& g,
                          const std::vector<double>& h, const std::vector<int>& root_rows) {
        auto dataset = QuantizedDataset::from_u16(N, P, Xb, std::numeric_limits<uint16_t>::max());
        fit_with_row_ids_impl_(dataset, N, P, g, h, root_rows);
    }

private:
    void fit_with_row_ids_impl_(const QuantizedDataset& Xb, int N, int P, const std::vector<double>& g,
                                const std::vector<double>& h, const std::vector<int>& root_rows) {
        if (N <= 0 || P <= 0) {
            throw std::invalid_argument("UnifiedTree::fit_with_row_ids: N and P must be positive");
        }
        if (static_cast<int>(g.size()) != N || static_cast<int>(h.size()) != N) {
            throw std::invalid_argument("UnifiedTree::fit_with_row_ids: g/h size must match N");
        }
        if (root_rows.empty()) {
            throw std::invalid_argument("UnifiedTree::fit_with_row_ids: root_rows must not be empty");
        }

        const size_t n_sz = static_cast<size_t>(N);
        const size_t p_sz = static_cast<size_t>(P);
        if (n_sz > std::numeric_limits<size_t>::max() / p_sz) {
            throw std::runtime_error("UnifiedTree::fit_with_row_ids: dataset dimensions overflow");
        }
        const size_t expected_cells = n_sz * p_sz;
        if (Xb.size() != expected_cells) {
            throw std::invalid_argument("UnifiedTree::fit_with_row_ids: Xb size must equal N*P");
        }
        for (int r : root_rows) {
            if (r < 0 || r >= N) {
                throw std::invalid_argument("UnifiedTree::fit_with_row_ids: root_rows contains out-of-range index");
            }
        }

        Xb_ = &Xb;
        Xb.visit_codes([&](auto codes) {
            using Code = typename decltype(codes)::value_type;
            if constexpr (std::is_same_v<Code, uint8_t>) {
                Xb8_ = codes.data();
                Xb16_ = nullptr;
            } else {
                Xb8_ = nullptr;
                Xb16_ = codes.data();
            }
        });
        g_ = &g;
        h_ = &h;
        unit_hessian_ = std::all_of(h.begin(), h.end(), [](double value) { return value == 1.0; });
        N_ = N;
        P_ = P;
        training_context_.bind(Xb, g, h, executor_, std::max(cfg_.num_classes - 1, 1));

        initialize_bin_info_();
        validate_monotone_constraints_();
        reset_();

        nodes_.reserve(std::max(2 * cfg_.max_leaves + 5, 64));
        id2pos_.reserve(std::max(2 * cfg_.max_leaves + 5, 64));
        build_feature_pool_();

        std::vector<int> seed = root_rows;
        tree_subsample_applied_ = (cfg_.subsample_bytree < 1.0);
        apply_tree_level_row_subsample_(seed);
        index_pool_ = std::move(seed);

        initialize_caching_();

        K_ = std::max(cfg_.num_classes - 1, 1);
        Node r;
        r.id = next_id_++;
        r.K = K_;
        r.depth = 0;
        r.lo = 0;
        r.hi = static_cast<int>(index_pool_.size());
        accum_(r);
        accum_goss_weighted_(r);
        nodes_.push_back(std::move(r));
        register_pos_(nodes_.back());

        if (cfg_.growth == TreeConfig::Growth::LeafWise)
            grow_leaf_();
        else if (cfg_.growth == TreeConfig::Growth::LevelWise)
            grow_level_();
        else
            grow_oblivious_();

        for (auto& n : nodes_) {
            if (n.is_leaf) {
                const auto [GG_vec, HH_vec] = node_totals_for_leaf_(n);
                double GG = 0.0, HH = 0.0;
                for (size_t i = 0; i < GG_vec.size(); ++i) {
                    GG += GG_vec[i];
                    HH += HH_vec[i];
                }
                n.leaf_values = leaf_values_(GG, HH, n.min_constraint, n.max_constraint);
            }
        }
        pack_();
        cleanup_caching_();
        training_context_.release_dataset();
    }

public:
    std::vector<double> predict(const QuantizedDataset& Xb, const double* Xraw_opt = nullptr) const {
        if (Xb.features() != P_)
            throw std::invalid_argument("UnifiedTree::predict: P mismatch");
        std::vector<double> out(static_cast<size_t>(Xb.rows()), 0.0);
        if (!packed_)
            return out;
        Xb.visit_codes([&](auto codes) {
            using Code = typename decltype(codes)::value_type;
            for (int row = 0; row < Xb.rows(); ++row) {
                const Code* row_binned = codes.data() + static_cast<size_t>(row) * static_cast<size_t>(P_);
                out[static_cast<size_t>(row)] = predict_one_compact_(row_binned, row, Xraw_opt);
            }
        });
        return out;
    }

    double predict_one_binned(const QuantizedDataset& Xb, int row_idx, const double* Xraw_opt = nullptr) const {
        return Xb.visit_codes([&](auto codes) {
            const auto* row = codes.data() + static_cast<size_t>(row_idx) * static_cast<size_t>(P_);
            return predict_one_compact_(row, row_idx, Xraw_opt);
        });
    }

    template <class Code>
    double predict_one_binned_typed(const Code* row_binned, int row_idx, const double* Xraw_opt = nullptr) const {
        return predict_one_compact_(row_binned, row_idx, Xraw_opt);
    }

    std::vector<double> predict(const std::vector<uint16_t>& Xb, int N, int P) const {
        if (N < 0 || P <= 0) {
            throw std::invalid_argument("UnifiedTree::predict: N must be non-negative and P positive");
        }
        if (P != P_) {
            throw std::invalid_argument("UnifiedTree::predict: P mismatch");
        }
        const size_t n_sz = static_cast<size_t>(N);
        const size_t p_sz = static_cast<size_t>(P);
        if (n_sz > 0 && n_sz > std::numeric_limits<size_t>::max() / p_sz) {
            throw std::runtime_error("UnifiedTree::predict: dataset dimensions overflow");
        }
        if (Xb.size() != n_sz * p_sz) {
            throw std::invalid_argument("UnifiedTree::predict: Xb size must equal N*P");
        }

        std::vector<double> out(static_cast<size_t>(N), 0.0);
        if (!packed_)
            return out;
        for (int i = 0; i < N; ++i) {
            const uint16_t* row_binned = Xb.data() + static_cast<size_t>(i) * static_cast<size_t>(P_);
            out[static_cast<size_t>(i)] = predict_one_with_raw_opt_(row_binned, i, nullptr);
        }
        return out;
    }

    std::vector<double> predict(const std::vector<uint16_t>& Xb, int N, int P, const double* Xraw_opt) const {
        if (N < 0 || P <= 0) {
            throw std::invalid_argument("UnifiedTree::predict: N must be non-negative and P positive");
        }
        if (P != P_) {
            throw std::invalid_argument("UnifiedTree::predict: P mismatch");
        }
        const size_t n_sz = static_cast<size_t>(N);
        const size_t p_sz = static_cast<size_t>(P);
        if (n_sz > 0 && n_sz > std::numeric_limits<size_t>::max() / p_sz) {
            throw std::runtime_error("UnifiedTree::predict: dataset dimensions overflow");
        }
        if (Xb.size() != n_sz * p_sz) {
            throw std::invalid_argument("UnifiedTree::predict: Xb size must equal N*P");
        }

        std::vector<double> out(static_cast<size_t>(N), 0.0);
        if (!packed_)
            return out;

        for (int i = 0; i < N; ++i) {
            const uint16_t* row_binned = Xb.data() + static_cast<size_t>(i) * static_cast<size_t>(P_);
            out[static_cast<size_t>(i)] = predict_one_with_raw_opt_(row_binned, i, Xraw_opt);
        }
        return out;
    }

    double predict_one_binned(const uint16_t* row_binned, int row_idx, const double* Xraw_opt = nullptr) const {
        return predict_one_with_raw_opt_(row_binned, row_idx, Xraw_opt);
    }

    std::vector<double> predict_contrib(const std::vector<uint16_t>& Xb, int N, int P) const {
        return predict_contrib(Xb, N, P, nullptr);
    }

    std::vector<double> predict_contrib(const std::vector<uint16_t>& Xb, int N, int P, const double* Xraw_opt) const {
        if (N < 0 || P <= 0) {
            throw std::invalid_argument("UnifiedTree::predict_contrib: N must be non-negative and P positive");
        }
        if (P != P_) {
            throw std::invalid_argument("UnifiedTree::predict_contrib: P mismatch");
        }
        const size_t n_sz = static_cast<size_t>(N);
        const size_t p_sz = static_cast<size_t>(P);
        if (n_sz > 0 && n_sz > std::numeric_limits<size_t>::max() / p_sz) {
            throw std::runtime_error("UnifiedTree::predict_contrib: dataset dimensions overflow");
        }
        if (Xb.size() != n_sz * p_sz) {
            throw std::invalid_argument("UnifiedTree::predict_contrib: Xb size must equal N*P");
        }
        // Multiclass TreeSHAP not yet implemented
        if (cfg_.num_classes > 1) {
            throw std::runtime_error("UnifiedTree::predict_contrib: TreeSHAP not supported for multiclass yet");
        }
        if (!packed_)
            return std::vector<double>(static_cast<size_t>(N) * static_cast<size_t>(P_ + 1), 0.0);

        validate_tree_shap_support_();
        const auto split_features = tree_shap_split_features_();
        const bool use_bruteforce = split_features.size() <= kBruteforceTreeShapMaxFeatures;
        const bool has_repeated_feature = tree_shap_has_repeated_feature_();
        if (!use_bruteforce && has_repeated_feature) {
            throw std::runtime_error("UnifiedTree::predict_contrib: repeated split features require brute-force "
                                     "TreeSHAP and exceed the supported feature limit");
        }

        std::vector<double> out(static_cast<size_t>(N) * static_cast<size_t>(P_ + 1), 0.0);
        std::vector<int> feature_to_pos(static_cast<size_t>(P_), -1);
        for (size_t k = 0; k < split_features.size(); ++k) {
            feature_to_pos[static_cast<size_t>(split_features[k])] = static_cast<int>(k);
        }
        const double bias = use_bruteforce ? 0.0 : tree_expected_value_();
        for (int i = 0; i < N; ++i) {
            double* row_out = out.data() + static_cast<size_t>(i) * static_cast<size_t>(P_ + 1);
            const uint16_t* row_binned = Xb.data() + static_cast<size_t>(i) * static_cast<size_t>(P_);
            const auto raw_view = resolve_predict_raw_view_(Xraw_opt);
            if (use_bruteforce) {
                brute_force_tree_shap_row_(row_binned, i, raw_view, row_out, feature_to_pos,
                                           static_cast<int>(split_features.size()));
            } else {
                row_out[P_] = bias;
                std::vector<PathElement> path(static_cast<size_t>(depth() + 2));
                tree_shap_recursive_(root_id_, row_binned, i, raw_view, row_out, 0, path, 1.0, 1.0, -1);
            }
        }
        return out;
    }

    struct PredictRawView {
        const double* Xraw = nullptr;
        const uint8_t* Xmiss = nullptr;
    };

    struct PathElement {
        int feature_index = -1;
        double zero_fraction = 0.0;
        double one_fraction = 0.0;
        double pweight = 0.0;
    };

    static constexpr size_t kBruteforceTreeShapMaxFeatures = 12;

    inline PredictRawView resolve_predict_raw_view_(const double* Xraw_opt) const {
        if (Xraw_opt) {
            if (Xraw_opt == Xraw_eval_)
                return {Xraw_opt, Xmiss_eval_};
            if (Xraw_opt == Xraw_)
                return {Xraw_opt, Xmiss_};
            return {Xraw_opt, nullptr};
        }
        // The no-raw predict overload should stay binned-only. Exact/raw-aware
        // routing is enabled through predict(..., Xraw_opt).
        return {nullptr, nullptr};
    }

    inline double predict_feature_value_(const uint16_t* row_binned, int row_idx, int feat,
                                         const PredictRawView& raw_view) const {
        if (raw_view.Xraw) {
            const size_t raw_off = static_cast<size_t>(row_idx) * static_cast<size_t>(P_) + static_cast<size_t>(feat);
            if (raw_view.Xmiss && raw_view.Xmiss[raw_off] != 0)
                return std::numeric_limits<double>::quiet_NaN();
            return raw_view.Xraw[raw_off];
        }
        return binned_value_for_feature_(feat, row_binned[feat]);
    }

    inline bool predict_go_left_oblique_(int id, const uint16_t* row_binned, int row_idx, bool miss_left,
                                         const PredictRawView& raw_view) const {
        bool go_left = miss_left;
        const int oblique_off = packed_tree_.oblique_offsets[id];
        const int oblique_cnt = packed_tree_.oblique_counts[id];
        if (oblique_off < 0 || oblique_cnt <= 0)
            return go_left;

        double z = 0.0;
        for (int k = 0; k < oblique_cnt; ++k) {
            const int fz = packed_tree_.oblique_features[oblique_off + k];
            const double xv = predict_feature_value_(row_binned, row_idx, fz, raw_view);
            if (!std::isfinite(xv))
                return miss_left;
            z += packed_tree_.oblique_weights[oblique_off + k] * xv;
        }
        return (z <= packed_tree_.oblique_thresholds[id]);
    }

    inline bool predict_go_left_categorical_(int id, const uint16_t* row_binned, int feat, bool miss_left) const {
        const uint16_t b = row_binned[feat];
        const uint16_t feat_miss = static_cast<uint16_t>(missing_ids_per_feat_[feat]);
        if (b == feat_miss)
            return miss_left;

        const int off = packed_tree_.categorical_offsets[id];
        const int cnt = packed_tree_.categorical_counts[id];
        if (off < 0 || cnt <= 0)
            return false;
        const auto beg = packed_tree_.categorical_bins.begin() + off;
        const auto end = beg + cnt;
        return std::binary_search(beg, end, static_cast<int>(b));
    }

    inline bool predict_go_left_pair_(int id, const uint16_t* row_binned, bool miss_left) const {
        const int fa = packed_tree_.pair_features_a[id];
        const int fb = packed_tree_.pair_features_b[id];
        const uint16_t a = row_binned[fa];
        const uint16_t b = row_binned[fb];
        if (a == static_cast<uint16_t>(missing_ids_per_feat_[fa]) ||
            b == static_cast<uint16_t>(missing_ids_per_feat_[fb]))
            return miss_left;
        const int quadrant =
            (a > packed_tree_.pair_thresholds_a[id] ? 2 : 0) | (b > packed_tree_.pair_thresholds_b[id] ? 1 : 0);
        return (packed_tree_.pair_quadrant_masks[id] & (uint8_t{1} << quadrant)) != 0;
    }

    inline bool predict_go_left_axis_(int id, const uint16_t* row_binned, int row_idx, int feat, int thr,
                                      bool miss_left, const PredictRawView& raw_view) const {
        const uint16_t b = row_binned[feat];
        const uint16_t feat_miss = static_cast<uint16_t>(missing_ids_per_feat_[feat]);
        const bool is_miss = (b == feat_miss);

        if (std::isfinite(packed_tree_.split_values[id])) {
            if (raw_view.Xraw) {
                const double xv = predict_feature_value_(row_binned, row_idx, feat, raw_view);
                return std::isfinite(xv) ? (xv <= packed_tree_.split_values[id]) : miss_left;
            }
            if (!is_miss) {
                const double xv = binned_value_for_feature_(feat, b);
                return std::isfinite(xv) ? (xv <= packed_tree_.split_values[id]) : miss_left;
            }
            return miss_left;
        }

        return is_miss ? miss_left : (b <= static_cast<uint16_t>(thr));
    }

    inline double predict_one_with_raw_opt_(const uint16_t* row_binned, int row_idx, const double* Xraw_opt) const {
        const auto raw_view = resolve_predict_raw_view_(Xraw_opt);
        int id = root_id_;
        while (id >= 0 && packed_tree_.leaf_flags[id] == 0) {
            const int f = packed_tree_.features[id];
            const int t = packed_tree_.thresholds[id];
            const bool ml = (packed_tree_.missing_left[id] != 0);
            const auto kind = static_cast<splitx::SplitKind>(packed_tree_.split_kinds[id]);
            bool go_left = false;
            if (kind == splitx::SplitKind::Oblique) {
                go_left = predict_go_left_oblique_(id, row_binned, row_idx, ml, raw_view);
            } else if (kind == splitx::SplitKind::PairInteraction) {
                go_left = predict_go_left_pair_(id, row_binned, ml);
            } else if (kind == splitx::SplitKind::CategoricalPartition) {
                go_left = predict_go_left_categorical_(id, row_binned, f, ml);
            } else {
                go_left = predict_go_left_axis_(id, row_binned, row_idx, f, t, ml, raw_view);
            }
            id = go_left ? packed_tree_.left_children[id] : packed_tree_.right_children[id];
        }

        return predict_leaf_value_(id, row_idx, raw_view);
    }

    inline double predict_one_with_raw_(const uint16_t* row_binned, int row_idx) const {
        return predict_one_with_raw_opt_(row_binned, row_idx, nullptr);
    }

    bool row_has_valid_neural_inputs_(int row_idx, const PredictRawView& raw_view) const {
        if (!raw_view.Xraw)
            return false;
        const size_t row_off = static_cast<size_t>(row_idx) * static_cast<size_t>(P_);
        for (int feat = 0; feat < P_; ++feat) {
            const size_t off = row_off + static_cast<size_t>(feat);
            const double xv = raw_view.Xraw[off];
            const bool miss = raw_view.Xmiss ? (raw_view.Xmiss[off] != 0) : !std::isfinite(xv);
            if (miss || !std::isfinite(xv))
                return false;
        }
        return true;
    }

    double predict_leaf_value_(int leaf_id, int row_idx, const PredictRawView& raw_view) const {
        if (leaf_id < 0)
            return 0.0;

        const Node* leaf_node = by_id_(leaf_id);
        if (!leaf_node)
            return 0.0;

        if (!leaf_node->has_neural_leaf()) {
            if (K_ <= 1) {
                if (leaf_id >= static_cast<int>(packed_tree_.leaf_values.size()))
                    return 0.0;
                return packed_tree_.leaf_values[static_cast<size_t>(leaf_id)];
            } else {
                // Multiclass: return first class
                if (leaf_id >= static_cast<int>(packed_tree_.leaf_values.size() / K_))
                    return 0.0;
                return packed_tree_.leaf_values[static_cast<size_t>(leaf_id) * K_];
            }
        }

        if (!raw_view.Xraw) {
            throw std::runtime_error("UnifiedTree::predict: neural leaf inference requires raw features; call "
                                     "predict(..., Xraw) or use ForeForest.predict(...)");
        }

        if (!row_has_valid_neural_inputs_(row_idx, raw_view))
            return (leaf_node->leaf_values.empty()) ? 0.0 : leaf_node->leaf_values[0];

        const double* row_ptr = raw_view.Xraw + static_cast<size_t>(row_idx) * static_cast<size_t>(P_);
        return leaf_node->neural_leaf->predict_one(row_ptr);
    }

    void validate_tree_shap_support_() const {
        for (const auto& n : nodes_) {
            if (n.has_neural_leaf()) {
                throw std::runtime_error("UnifiedTree::predict_contrib: TreeSHAP does not support neural leaves");
            }
            if (!n.is_leaf && n.split_kind != splitx::SplitKind::Axis) {
                throw std::runtime_error(
                    "UnifiedTree::predict_contrib: TreeSHAP currently supports axis-aligned splits only");
            }
        }
    }

    std::vector<int> tree_shap_split_features_() const {
        std::set<int> seen;
        for (const auto& n : nodes_) {
            if (!n.is_leaf && n.split_kind == splitx::SplitKind::Axis && n.feature >= 0) {
                seen.insert(n.feature);
            }
        }
        return std::vector<int>(seen.begin(), seen.end());
    }

    bool tree_shap_has_repeated_feature_path_(int node_id, std::vector<uint8_t>& on_path) const {
        if (node_id < 0 || node_id >= static_cast<int>(packed_tree_.leaf_flags.size()) ||
            packed_tree_.leaf_flags[static_cast<size_t>(node_id)] != 0) {
            return false;
        }

        const int feat = packed_tree_.features[static_cast<size_t>(node_id)];
        if (feat < 0 || feat >= P_)
            return false;
        if (on_path[static_cast<size_t>(feat)] != 0)
            return true;

        on_path[static_cast<size_t>(feat)] = 1;
        const bool found =
            tree_shap_has_repeated_feature_path_(packed_tree_.left_children[static_cast<size_t>(node_id)], on_path) ||
            tree_shap_has_repeated_feature_path_(packed_tree_.right_children[static_cast<size_t>(node_id)], on_path);
        on_path[static_cast<size_t>(feat)] = 0;
        return found;
    }

    bool tree_shap_has_repeated_feature_() const {
        std::vector<uint8_t> on_path(static_cast<size_t>(P_), 0);
        return tree_shap_has_repeated_feature_path_(root_id_, on_path);
    }

    inline double node_cover_(int id) const {
        if (id < 0 || id >= static_cast<int>(packed_tree_.cover.size()))
            return 0.0;
        return packed_tree_.cover[static_cast<size_t>(id)];
    }

    double tree_expected_value_from_node_(int id) const {
        if (id < 0)
            return 0.0;
        if (id >= static_cast<int>(packed_tree_.leaf_flags.size()))
            return 0.0;
        if (packed_tree_.leaf_flags[id] != 0)
            return packed_tree_.leaf_values[static_cast<size_t>(id) * K_];

        const int left_id = packed_tree_.left_children[static_cast<size_t>(id)];
        const int right_id = packed_tree_.right_children[static_cast<size_t>(id)];
        const double cover = node_cover_(id);
        const double left_cover = node_cover_(left_id);
        const double right_cover = node_cover_(right_id);
        if (cover <= 0.0) {
            return 0.5 * tree_expected_value_from_node_(left_id) + 0.5 * tree_expected_value_from_node_(right_id);
        }
        const double left_frac = left_cover / cover;
        const double right_frac = right_cover / cover;
        return left_frac * tree_expected_value_from_node_(left_id) +
               right_frac * tree_expected_value_from_node_(right_id);
    }

    double tree_expected_value_() const {
        return tree_expected_value_from_node_(root_id_);
    }

    double tree_shap_expected_value_mask_(int node_id, const uint16_t* row_binned, int row_idx,
                                          const PredictRawView& raw_view, const std::vector<int>& feature_to_pos,
                                          uint64_t mask) const {
        if (node_id < 0)
            return 0.0;
        if (packed_tree_.leaf_flags[static_cast<size_t>(node_id)] != 0)
            return packed_tree_.leaf_values[static_cast<size_t>(node_id)];

        const int split_feature = packed_tree_.features[static_cast<size_t>(node_id)];
        const int feature_pos = (split_feature >= 0 && split_feature < static_cast<int>(feature_to_pos.size()))
                                    ? feature_to_pos[static_cast<size_t>(split_feature)]
                                    : -1;

        if (feature_pos >= 0 && (mask & (uint64_t{1} << static_cast<uint64_t>(feature_pos))) != 0) {
            const bool go_left = predict_go_left_axis_(
                node_id, row_binned, row_idx, split_feature, packed_tree_.thresholds[static_cast<size_t>(node_id)],
                packed_tree_.missing_left[static_cast<size_t>(node_id)] != 0, raw_view);
            return tree_shap_expected_value_mask_(go_left ? packed_tree_.left_children[static_cast<size_t>(node_id)]
                                                          : packed_tree_.right_children[static_cast<size_t>(node_id)],
                                                  row_binned, row_idx, raw_view, feature_to_pos, mask);
        }

        const int left_id = packed_tree_.left_children[static_cast<size_t>(node_id)];
        const int right_id = packed_tree_.right_children[static_cast<size_t>(node_id)];
        const double cover = std::max(1e-12, node_cover_(node_id));
        const double left_fraction = node_cover_(left_id) / cover;
        const double right_fraction = node_cover_(right_id) / cover;
        return left_fraction *
                   tree_shap_expected_value_mask_(left_id, row_binned, row_idx, raw_view, feature_to_pos, mask) +
               right_fraction *
                   tree_shap_expected_value_mask_(right_id, row_binned, row_idx, raw_view, feature_to_pos, mask);
    }

    static double tree_shap_combination_(int n, int k) {
        if (k < 0 || k > n)
            return 0.0;
        if (k == 0 || k == n)
            return 1.0;
        k = std::min(k, n - k);
        double out = 1.0;
        for (int i = 1; i <= k; ++i) {
            out *= static_cast<double>(n - (k - i));
            out /= static_cast<double>(i);
        }
        return out;
    }

    void brute_force_tree_shap_row_(const uint16_t* row_binned, int row_idx, const PredictRawView& raw_view,
                                    double* phi, const std::vector<int>& feature_to_pos, int M) const {
        if (M <= 0) {
            phi[P_] = tree_expected_value_();
            return;
        }

        const size_t n_masks = size_t{1} << static_cast<size_t>(M);
        std::vector<double> values(n_masks, 0.0);
        for (size_t mask = 0; mask < n_masks; ++mask) {
            values[mask] = tree_shap_expected_value_mask_(root_id_, row_binned, row_idx, raw_view, feature_to_pos,
                                                          static_cast<uint64_t>(mask));
        }

        phi[P_] = values[0];
        for (int j = 0; j < P_; ++j)
            phi[j] = 0.0;

        for (int j = 0; j < P_; ++j) {
            const int pos = feature_to_pos[static_cast<size_t>(j)];
            if (pos < 0)
                continue;
            const uint64_t bit = uint64_t{1} << static_cast<uint64_t>(pos);
            double contrib = 0.0;
            for (size_t mask = 0; mask < n_masks; ++mask) {
                if ((static_cast<uint64_t>(mask) & bit) != 0)
                    continue;
                const int s = __builtin_popcountll(static_cast<unsigned long long>(mask));
                const double weight = 1.0 / (static_cast<double>(M) * tree_shap_combination_(M - 1, s));
                contrib += weight * (values[mask | bit] - values[mask]);
            }
            phi[j] = contrib;
        }
    }

    static void extend_path_(std::vector<PathElement>& path, int unique_depth, double zero_fraction,
                             double one_fraction, int feature_index) {
        path[static_cast<size_t>(unique_depth)].feature_index = feature_index;
        path[static_cast<size_t>(unique_depth)].zero_fraction = zero_fraction;
        path[static_cast<size_t>(unique_depth)].one_fraction = one_fraction;
        path[static_cast<size_t>(unique_depth)].pweight = (unique_depth == 0) ? 1.0 : 0.0;

        for (int i = unique_depth - 1; i >= 0; --i) {
            path[static_cast<size_t>(i + 1)].pweight += one_fraction * path[static_cast<size_t>(i)].pweight *
                                                        static_cast<double>(i + 1) /
                                                        static_cast<double>(unique_depth + 1);
            path[static_cast<size_t>(i)].pweight = zero_fraction * path[static_cast<size_t>(i)].pweight *
                                                   static_cast<double>(unique_depth - i) /
                                                   static_cast<double>(unique_depth + 1);
        }
    }

    static void unwind_path_(std::vector<PathElement>& path, int unique_depth, int path_index) {
        const double one_fraction = path[static_cast<size_t>(path_index)].one_fraction;
        const double zero_fraction = path[static_cast<size_t>(path_index)].zero_fraction;
        double next_one_portion = path[static_cast<size_t>(unique_depth)].pweight;

        for (int i = unique_depth - 1; i >= 0; --i) {
            if (one_fraction != 0.0) {
                const double tmp = path[static_cast<size_t>(i)].pweight;
                path[static_cast<size_t>(i)].pweight = next_one_portion * static_cast<double>(unique_depth + 1) /
                                                       (static_cast<double>(i + 1) * one_fraction);
                next_one_portion = tmp - path[static_cast<size_t>(i)].pweight * zero_fraction *
                                             static_cast<double>(unique_depth - i) /
                                             static_cast<double>(unique_depth + 1);
            } else {
                path[static_cast<size_t>(i)].pweight = path[static_cast<size_t>(i)].pweight *
                                                       static_cast<double>(unique_depth + 1) /
                                                       (zero_fraction * static_cast<double>(unique_depth - i));
            }
        }

        for (int i = path_index; i < unique_depth; ++i) {
            path[static_cast<size_t>(i)] = path[static_cast<size_t>(i + 1)];
        }
    }

    static double unwound_path_sum_(const std::vector<PathElement>& path, int unique_depth, int path_index) {
        const double one_fraction = path[static_cast<size_t>(path_index)].one_fraction;
        const double zero_fraction = path[static_cast<size_t>(path_index)].zero_fraction;
        double next_one_portion = path[static_cast<size_t>(unique_depth)].pweight;
        double total = 0.0;

        for (int i = unique_depth - 1; i >= 0; --i) {
            if (one_fraction != 0.0) {
                const double tmp = next_one_portion * static_cast<double>(unique_depth + 1) /
                                   (static_cast<double>(i + 1) * one_fraction);
                total += tmp;
                next_one_portion = path[static_cast<size_t>(i)].pweight - tmp * zero_fraction *
                                                                              static_cast<double>(unique_depth - i) /
                                                                              static_cast<double>(unique_depth + 1);
            } else {
                total += path[static_cast<size_t>(i)].pweight * static_cast<double>(unique_depth + 1) /
                         (zero_fraction * static_cast<double>(unique_depth - i));
            }
        }

        return total;
    }

    void tree_shap_recursive_(int node_id, const uint16_t* row_binned, int row_idx, const PredictRawView& raw_view,
                              double* phi, int unique_depth, std::vector<PathElement> path, double parent_zero_fraction,
                              double parent_one_fraction, int parent_feature_index) const {
        if (node_id < 0)
            return;

        extend_path_(path, unique_depth, parent_zero_fraction, parent_one_fraction, parent_feature_index);

        if (packed_tree_.leaf_flags[static_cast<size_t>(node_id)] != 0) {
            const double leaf_value = packed_tree_.leaf_values[static_cast<size_t>(node_id) * K_];
            for (int i = 1; i <= unique_depth; ++i) {
                const double weight = unwound_path_sum_(path, unique_depth, i);
                const auto& el = path[static_cast<size_t>(i)];
                if (el.feature_index >= 0 && el.feature_index < P_) {
                    phi[el.feature_index] += weight * (el.one_fraction - el.zero_fraction) * leaf_value;
                }
            }
            return;
        }

        const int split_feature = packed_tree_.features[static_cast<size_t>(node_id)];
        const int split_thr = packed_tree_.thresholds[static_cast<size_t>(node_id)];
        const bool miss_left = (packed_tree_.missing_left[static_cast<size_t>(node_id)] != 0);
        const bool hot_left =
            predict_go_left_axis_(node_id, row_binned, row_idx, split_feature, split_thr, miss_left, raw_view);

        const int hot_index = hot_left ? packed_tree_.left_children[static_cast<size_t>(node_id)]
                                       : packed_tree_.right_children[static_cast<size_t>(node_id)];
        const int cold_index = hot_left ? packed_tree_.right_children[static_cast<size_t>(node_id)]
                                        : packed_tree_.left_children[static_cast<size_t>(node_id)];

        double incoming_zero_fraction = 1.0;
        double incoming_one_fraction = 1.0;
        int path_index = 0;
        for (; path_index <= unique_depth; ++path_index) {
            if (path[static_cast<size_t>(path_index)].feature_index == split_feature)
                break;
        }
        if (path_index <= unique_depth) {
            incoming_zero_fraction = path[static_cast<size_t>(path_index)].zero_fraction;
            incoming_one_fraction = path[static_cast<size_t>(path_index)].one_fraction;
            unwind_path_(path, unique_depth, path_index);
            --unique_depth;
        }

        const double cover = std::max(1e-12, node_cover_(node_id));
        const double hot_zero_fraction = node_cover_(hot_index) / cover;
        const double cold_zero_fraction = node_cover_(cold_index) / cover;

        tree_shap_recursive_(hot_index, row_binned, row_idx, raw_view, phi, unique_depth + 1, path,
                             hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction, split_feature);
        tree_shap_recursive_(cold_index, row_binned, row_idx, raw_view, phi, unique_depth + 1, path,
                             cold_zero_fraction * incoming_zero_fraction, 0.0, split_feature);
    }

    const std::vector<double>& feature_importance_gain() const {
        return feat_gain_;
    }
    const std::vector<double>& feature_importance_cover() const {
        return feat_cover_;
    }
    const std::vector<int>& feature_importance_frequency() const {
        return feat_frequency_;
    }

    int n_nodes() const {
        return static_cast<int>(nodes_.size());
    }
    int n_leaves() const {
        int c = 0;
        for (auto& n : nodes_)
            if (n.is_leaf)
                ++c;
        return c;
    }
    int depth() const {
        int d = 0;
        for (auto& n : nodes_)
            d = std::max(d, n.depth);
        return d;
    }

    void post_prune_ccp(double ccp_alpha) {
        if (!packed_)
            throw std::runtime_error("post_prune_ccp can only be called on packed trees");
        if (nodes_.empty() || ccp_alpha <= 0.0)
            return;

        std::vector<Node*> by_id;
        by_id.reserve(nodes_.size());
        for (auto& n : nodes_) {
            if (static_cast<int>(by_id.size()) <= n.id)
                by_id.resize(n.id + 1, nullptr);
            by_id[n.id] = &n;
        }
        Node* root = by_id[root_id_];
        if (!root)
            return;

        struct Stats {
            int leaves = 0;
            int internal = 0;
            double R_sub = 0.0;
            double R_collapse = 0.0;
            double alpha_star = std::numeric_limits<double>::infinity();
        };
        std::vector<Stats> S(by_id.size());

        std::function<void(Node*)> acc = [&](Node* nd) {
            if (!nd)
                return;
            if (nd->is_leaf) {
                S[nd->id].leaves = 1;
                S[nd->id].internal = 0;
                double Gs = 0.0, Hs = 0.0;
                for (size_t c = 0; c < nd->G.size(); ++c) {
                    Gs += nd->G[c];
                    Hs += nd->H[c];
                }
                const double Rleaf = leaf_objective_optimal_(Gs, Hs);
                S[nd->id].R_sub = Rleaf;
                S[nd->id].R_collapse = Rleaf;
                S[nd->id].alpha_star = std::numeric_limits<double>::infinity();
                return;
            }
            acc(by_id[nd->left]);
            acc(by_id[nd->right]);
            const auto& L = S[nd->left];
            const auto& R = S[nd->right];
            auto& dst = S[nd->id];
            dst.leaves = L.leaves + R.leaves;
            dst.internal = L.internal + R.internal + 1;
            dst.R_sub = L.R_sub + R.R_sub - cfg_.gamma_;
            double Gs2 = 0.0, Hs2 = 0.0;
            for (size_t c = 0; c < nd->G.size(); ++c) {
                Gs2 += nd->G[c];
                Hs2 += nd->H[c];
            }
            dst.R_collapse = leaf_objective_optimal_(Gs2, Hs2);
            const int denom = std::max(dst.leaves - 1, 1);
            dst.alpha_star = (dst.R_collapse - dst.R_sub) / static_cast<double>(denom);
        };
        acc(root);

        std::function<void(Node*)> apply = [&](Node* nd) {
            if (!nd || nd->is_leaf)
                return;
            apply(by_id[nd->left]);
            apply(by_id[nd->right]);
            if (S[nd->id].alpha_star <= ccp_alpha) {
                nd->is_leaf = true;
                nd->left = -1;
                nd->right = -1;
                nd->feature = -1;
                nd->thr = -1;
                nd->split_value = std::numeric_limits<double>::quiet_NaN();
                nd->miss_left = true;
                nd->oblique_missing_left = true;
                nd->best_gain = -std::numeric_limits<double>::infinity();
                const auto [GG_vec, HH_vec] = node_totals_for_leaf_(*nd);
                double GG = 0.0, HH = 0.0;
                for (size_t i = 0; i < GG_vec.size(); ++i) {
                    GG += GG_vec[i];
                    HH += HH_vec[i];
                }
                nd->leaf_values = leaf_values_(GG, HH, nd->min_constraint, nd->max_constraint);
            }
        };
        apply(root);
        pack_();
    }

    inline bool uses_goss_() const {
        return cfg_.goss.enabled && (cfg_.goss.top_rate + cfg_.goss.other_rate < 1.0);
    }
    inline bool should_use_goss_for_node_(const Node& n) const {
        return uses_goss_() && (n.hi - n.lo) >= cfg_.goss.min_node_size;
    }
    inline bool node_uses_goss_(const Node& n) const {
        return should_use_goss_for_node_(n) && n.uses_goss && n.goss_samples_valid_;
    }

public:
    const QuantizedDataset* Xb_ = nullptr;
    const uint8_t* Xb8_ = nullptr;
    const uint16_t* Xb16_ = nullptr;
    const std::vector<double>* g_ = nullptr;
    const std::vector<double>* h_ = nullptr;
    int N_ = 0, P_ = 0;
    int K_ = 1; // num classes - 1 (1=scalar)
    bool unit_hessian_ = false;

    TreeConfig cfg_;
    const GradientHistogramSystem* ghs_ = nullptr;
    std::shared_ptr<ParallelExecutor> executor_;
#ifdef FORETREE_HAS_CUDA
    cuda::CudaHistogramEngine* cuda_histogram_engine_ = nullptr;
#endif

    std::vector<int> finite_bins_per_feat_;
    std::vector<int> missing_ids_per_feat_;
    std::vector<size_t> feature_offsets_;
    std::vector<double> bin_centers_flat_;
    size_t total_hist_size_ = 0;
    int miss_id_ = -1;

    std::vector<Node> nodes_;
    std::vector<int> id2pos_;
    int next_id_ = 0;
    bool packed_ = false;
    int root_id_ = 0;

    std::vector<int> index_pool_;
    std::vector<int> feat_pool_;
    TreeTrainingContext training_context_;
    mutable std::mt19937 rng_;
    std::vector<double> feat_gain_;
    std::vector<double> feat_cover_;  // sum of hessian (weight) at nodes using this feature
    std::vector<int> feat_frequency_; // count of splits using this feature

    PackedTree packed_tree_;

    const double* Xraw_ = nullptr;
    const uint8_t* Xmiss_ = nullptr;
    const double* Xraw_eval_ = nullptr;
    const uint8_t* Xmiss_eval_ = nullptr;

    std::unique_ptr<HistogramPool> hist_pool_;
    std::shared_ptr<HistPair> tree_histogram_;
    std::vector<int> tree_features_;
    bool tree_subsample_applied_ = false;

    // ---------------------- GOSS Helper (NEW) -----------------------
    std::pair<double, double> compute_goss_rates_(const Node& n) const {
        double a = cfg_.goss.top_rate;
        double b = cfg_.goss.other_rate;

        if (cfg_.goss.adaptive) {
            const int total = n.hi - n.lo;
            // Sum G across classes for multiclass
            double G_sum = 0.0;
            for (size_t c = 0; c < n.G.size(); ++c)
                G_sum += n.G[c];
            const double mean = G_sum / std::max(total, 1);
            double var = 0.0;
            for (int i = n.lo; i < n.hi; ++i) {
                const double gi = (*g_)[index_pool_[i]];
                const double d = gi - mean;
                var += d * d;
            }
            var /= std::max(total - 1, 1);
            const double variance_factor = var / (var + 1.0);
            a = std::min(0.5, cfg_.goss.top_rate * (1.0 + cfg_.goss.adaptive_scale * variance_factor));
            b = std::min(0.5, cfg_.goss.other_rate * (1.0 + cfg_.goss.adaptive_scale * variance_factor));
        }

        return {a, b};
    }

    std::unordered_map<int, double> build_goss_weight_map_(const Node& n) const {
        std::unordered_map<int, double> goss_weights;
        if (!n.uses_goss || !n.goss_samples_valid_)
            return goss_weights;

        for (int r : n.goss_top_indices_) {
            goss_weights[r] = 1.0;
        }
        for (int r : n.goss_rest_indices_) {
            goss_weights[r] = n.goss_rest_scale;
        }

        return goss_weights;
    }

    // ------------------------------ Utilities --------------------------------
    void validate_config_() {
        cfg_.max_depth = std::max(1, cfg_.max_depth);
        cfg_.max_leaves = std::max(1, cfg_.max_leaves);
        cfg_.min_samples_split = std::max(2, cfg_.min_samples_split);
        cfg_.min_samples_leaf = std::max(1, cfg_.min_samples_leaf);
        cfg_.min_child_weight = std::max(0.0, cfg_.min_child_weight);
        cfg_.lambda_ = std::max(0.0, cfg_.lambda_);
        cfg_.n_bins = std::max(2, cfg_.n_bins);
        cfg_.exact_cutover = std::max(1, cfg_.exact_cutover);
        cfg_.subsample_bytree = std::clamp(cfg_.subsample_bytree, 0.0, 1.0);
        cfg_.subsample_bylevel = std::clamp(cfg_.subsample_bylevel, 0.0, 1.0);
        cfg_.subsample_bynode = std::clamp(cfg_.subsample_bynode, 0.0, 1.0);
        cfg_.cache_threshold = std::max(1, cfg_.cache_threshold);
        cfg_.goss.min_node_size = std::max(2, cfg_.goss.min_node_size);
        cfg_.sgld_noise_scale = std::max(0.0, cfg_.sgld_noise_scale);
        cfg_.goss.top_rate = std::clamp(cfg_.goss.top_rate, 0.01, 1.0);
        cfg_.goss.other_rate = std::clamp(cfg_.goss.other_rate, 0.0, 1.0);
        if (cfg_.goss.top_rate + cfg_.goss.other_rate > 1.0) {
            cfg_.goss.other_rate = std::max(0.0, 1.0 - cfg_.goss.top_rate);
        }
        cfg_.max_histogram_pool_size = std::max(1, cfg_.max_histogram_pool_size);
        cfg_.categorical_max_selected_categories = std::max(2, cfg_.categorical_max_selected_categories);
        cfg_.oblique_k_features = std::max(2, cfg_.oblique_k_features);
        cfg_.oblique_ridge = std::max(1e-9, cfg_.oblique_ridge);
        cfg_.axis_vs_oblique_guard = std::max(1.0, cfg_.axis_vs_oblique_guard);
        cfg_.pair_interaction.max_features = std::max(2, cfg_.pair_interaction.max_features);
        cfg_.pair_interaction.interaction_bins = std::clamp(cfg_.pair_interaction.interaction_bins, 2, 16);
        cfg_.pair_interaction.min_node_rows = std::max(2, cfg_.pair_interaction.min_node_rows);
        cfg_.pair_interaction.axis_guard_factor = std::max(1.0, cfg_.pair_interaction.axis_guard_factor);
        for (auto& group : cfg_.interaction_constraints) {
            std::erase_if(group, [](int feature) { return feature < 0; });
            std::ranges::sort(group);
            group.erase(std::unique(group.begin(), group.end()), group.end());
        }
        std::erase_if(cfg_.interaction_constraints, [](const auto& group) { return group.empty(); });

        auto& is_cfg = cfg_.interaction_seeded_oblique;
        is_cfg.pairs = std::max(1, is_cfg.pairs);
        is_cfg.max_var_candidates = std::max(2, is_cfg.max_var_candidates);
        is_cfg.max_top_features = std::clamp(is_cfg.max_top_features, 2, is_cfg.max_var_candidates);
        is_cfg.first_i_cap = std::clamp(is_cfg.first_i_cap, 1, std::max(1, is_cfg.max_top_features - 1));
        is_cfg.second_j_cap = std::clamp(is_cfg.second_j_cap, 2, is_cfg.max_top_features);
        is_cfg.ridge = std::max(0.0, is_cfg.ridge);
        is_cfg.axis_guard_factor = std::max(1.0, is_cfg.axis_guard_factor);
    }

    SplitEngineConfig make_split_engine_config_() const {
        SplitEngineConfig ecfg;
        ecfg.enable_axis = true;
        ecfg.enable_categorical_partition = cfg_.enable_categorical_splits;
        ecfg.categorical_max_selected_categories = cfg_.categorical_max_selected_categories;
        ecfg.enable_oblique = cfg_.enable_oblique_splits;
        ecfg.oblique_mode = cfg_.oblique_mode;
        ecfg.oblique_k_features = cfg_.oblique_k_features;
        ecfg.oblique_ridge = cfg_.oblique_ridge;
        ecfg.axis_vs_oblique_guard = cfg_.axis_vs_oblique_guard;
        ecfg.iseed = cfg_.interaction_seeded_oblique;
        ecfg.enable_pair_interactions = cfg_.enable_pair_interaction_splits;
        ecfg.pair_interaction = cfg_.pair_interaction;
        return ecfg;
    }

    void validate_monotone_constraints_() {
        if (cfg_.monotone_constraints.empty())
            return;
        if (cfg_.monotone_constraints.size() != static_cast<size_t>(P_)) {
            throw std::invalid_argument("UnifiedTree::fit: monotone_constraints size must match P");
        }
        for (auto& v : cfg_.monotone_constraints) {
            v = (v > 0) ? int8_t{1} : (v < 0 ? int8_t{-1} : int8_t{0});
        }
    }

    void initialize_bin_info_() {
        if (ghs_ && P_ > 0) {
            finite_bins_per_feat_ = ghs_->all_finite_bins();
            missing_ids_per_feat_.resize(P_);
            feature_offsets_.assign(P_ + 1, 0);

            for (int j = 0; j < P_; ++j) {
                missing_ids_per_feat_[j] = ghs_->total_bins(j) - 1;
                feature_offsets_[j + 1] = feature_offsets_[j] + ghs_->total_bins(j);
            }
            total_hist_size_ = feature_offsets_[P_];
            bin_centers_flat_.assign(total_hist_size_, std::numeric_limits<double>::quiet_NaN());
            for (int j = 0; j < P_; ++j) {
                const auto& fb = ghs_->feature_bins(j);
                const int finite = finite_bins_per_feat_[j];
                for (int b = 0; b < finite; ++b) {
                    const size_t off = feature_offsets_[j] + static_cast<size_t>(b);
                    if (static_cast<size_t>(b + 1) < fb.edges.size()) {
                        bin_centers_flat_[off] = 0.5 * (fb.edges[(size_t)b] + fb.edges[(size_t)b + 1]);
                    } else {
                        bin_centers_flat_[off] = static_cast<double>(b) + 0.5;
                    }
                }
            }

            cfg_.n_bins = ghs_->finite_bins();
            miss_id_ = ghs_->missing_bin_id();
        } else {
            finite_bins_per_feat_.assign(P_, cfg_.n_bins);
            missing_ids_per_feat_.assign(P_, cfg_.n_bins);
            miss_id_ = cfg_.n_bins;

            feature_offsets_.resize(P_ + 1);
            for (int j = 0; j <= P_; ++j)
                feature_offsets_[j] = j * (cfg_.n_bins + 1);
            total_hist_size_ = static_cast<size_t>(P_) * static_cast<size_t>(cfg_.n_bins + 1);
            bin_centers_flat_.assign(total_hist_size_, std::numeric_limits<double>::quiet_NaN());
            for (int j = 0; j < P_; ++j) {
                for (int b = 0; b < cfg_.n_bins; ++b) {
                    const size_t off = feature_offsets_[j] + static_cast<size_t>(b);
                    bin_centers_flat_[off] = static_cast<double>(b) + 0.5;
                }
            }
        }
    }

    void initialize_caching_() {
        if (cfg_.cache_histograms) {
            hist_pool_ = std::make_unique<HistogramPool>(total_hist_size_, K_, cfg_.max_histogram_pool_size);
        }
        if (cfg_.cache_histograms && !cfg_.goss.enabled && !tree_subsample_applied_ &&
            static_cast<int>(index_pool_.size()) >= cfg_.cache_threshold) {
            build_tree_histogram_();
        }
    }

    void cleanup_caching_() {
        for (auto& node : nodes_) {
            node.histogram.reset();
            node.hist_features.clear();
            node.hist_valid = false;
        }
        tree_histogram_.reset();
        if (hist_pool_) {
            hist_pool_->clear();
            hist_pool_.reset();
        }
        tree_features_.clear();
    }

    std::shared_ptr<HistPair> acquire_histogram_() const {
        if (!hist_pool_) {
            auto histogram = std::make_shared<HistPair>();
            histogram->resize(total_hist_size_, K_);
            histogram->clear();
            return histogram;
        }
        HistPair* histogram = hist_pool_->get().release();
        return std::shared_ptr<HistPair>(histogram, [pool = hist_pool_.get()](HistPair* value) {
            pool->return_histogram(std::unique_ptr<HistPair>(value));
        });
    }

    void accumulate_hist_bin_(HistPair& hist, int r, int f) const {
        uint16_t b = code_at_(r, f);
        if (b >= static_cast<uint16_t>(missing_ids_per_feat_[f]))
            b = static_cast<uint16_t>(missing_ids_per_feat_[f]);
        const size_t off = feature_offsets_[f] + static_cast<size_t>(b);

        if (off < hist.G.size()) {
            hist.G[off] += (*g_)[r];
            if (!unit_hessian_)
                hist.H[off] += (*h_)[r];
        }
        if (off < hist.C.size()) {
            hist.C[off] += 1;
        }
    }

    void build_tree_histogram_() {
        tree_features_ = feat_pool_;
        if (tree_features_.empty()) {
            tree_features_.resize(P_);
            std::iota(tree_features_.begin(), tree_features_.end(), 0);
        }

        tree_histogram_ = std::make_shared<HistPair>();
        tree_histogram_->resize(total_hist_size_, K_);
        tree_histogram_->clear();

        const int row_count = static_cast<int>(index_pool_.size());
        const int feature_count = static_cast<int>(tree_features_.size());
        const int grain = row_count * feature_count >= 32768 ? 1 : feature_count;
        executor_->parallel_for(0, feature_count, grain, [&](int feature_begin, int feature_end) {
            for (int feature_pos = feature_begin; feature_pos < feature_end; ++feature_pos) {
                const int feature = tree_features_[static_cast<size_t>(feature_pos)];
                for (int row : index_pool_)
                    accumulate_hist_bin_(*tree_histogram_, row, feature);
                if (unit_hessian_) {
                    const size_t begin = feature_offsets_[feature];
                    const size_t end = begin + static_cast<size_t>(missing_ids_per_feat_[feature]) + 1;
                    for (size_t offset = begin; offset < end; ++offset)
                        tree_histogram_->H[offset] = static_cast<double>(tree_histogram_->C[offset]);
                }
            }
        });
    }

    void reset_() {
        nodes_.clear();
        id2pos_.clear();
        index_pool_.clear();
        next_id_ = 0;
        packed_ = false;
        root_id_ = 0;
        feat_gain_.assign(P_, 0.0);
        feat_cover_.assign(P_, 0.0);
        feat_frequency_.assign(P_, 0);
        feat_pool_.clear();
        tree_subsample_applied_ = false;
        packed_tree_.split_kinds.clear();
        packed_tree_.split_values.clear();
        packed_tree_.cover.clear();
        packed_tree_.categorical_offsets.clear();
        packed_tree_.categorical_counts.clear();
        packed_tree_.categorical_bins.clear();
        packed_tree_.pair_features_a.clear();
        packed_tree_.pair_features_b.clear();
        packed_tree_.pair_thresholds_a.clear();
        packed_tree_.pair_thresholds_b.clear();
        packed_tree_.pair_quadrant_masks.clear();
        packed_tree_.oblique_offsets.clear();
        packed_tree_.oblique_counts.clear();
        packed_tree_.oblique_features.clear();
        packed_tree_.oblique_weights.clear();
        packed_tree_.oblique_thresholds.clear();
    }

    inline void register_pos_(const Node& n) {
        if (static_cast<int>(id2pos_.size()) <= n.id)
            id2pos_.resize(n.id + 1, -1);
        id2pos_[n.id] = static_cast<int>(nodes_.size()) - 1;
    }

    inline Node* by_id_(int id) {
        if (id < 0 || id >= static_cast<int>(id2pos_.size()))
            return nullptr;
        const int pos = id2pos_[id];
        if (pos < 0 || pos >= static_cast<int>(nodes_.size()))
            return nullptr;
        return &nodes_[pos];
    }

    inline const Node* by_id_(int id) const {
        if (id < 0 || id >= static_cast<int>(id2pos_.size()))
            return nullptr;
        const int pos = id2pos_[id];
        if (pos < 0 || pos >= static_cast<int>(nodes_.size()))
            return nullptr;
        return &nodes_[pos];
    }

    // -------------------------- Accumulators / GOSS --------------------------
    void accum_(Node& n) {
        const int K_ = std::max(n.K, 1);
        n.G.assign(static_cast<size_t>(K_), 0.0);
        n.H.assign(static_cast<size_t>(K_), 0.0);
        std::normal_distribution<double> noise(0.0, cfg_.sgld_noise_scale);
        for (int i = n.lo; i < n.hi; ++i) {
            const int r = index_pool_[i];
            double g = (*g_)[r];
            if (cfg_.sgld_enabled)
                g += noise(rng_);
            double h = (*h_)[r];
            for (int c = 0; c < K_; ++c) {
                n.G[static_cast<size_t>(c)] += g;
                n.H[static_cast<size_t>(c)] += h;
            }
        }
        n.C = n.hi - n.lo;
    }

    inline void set_unweighted_node_totals_(Node& n) const {
        n.uses_goss = false;
        n.goss_weighted_G = n.G;
        n.goss_weighted_H = n.H;
        n.goss_rest_scale = 1.0;
        n.goss_samples_valid_ = false;
        n.goss_top_indices_.clear();
        n.goss_rest_indices_.clear();
    }

    // Sum gradients across K classes for multiclass
    std::pair<std::vector<double>, std::vector<double>> sum_grad_hess_for_rows_(const std::vector<int>& rows) {
        const int K_ = 1; // scalar input: same gradient per class
        std::vector<double> G_out(static_cast<size_t>(K_), 0.0);
        std::vector<double> H_out(static_cast<size_t>(K_), 0.0);
        std::normal_distribution<double> noise(0.0, cfg_.sgld_noise_scale);
        for (int r : rows) {
            double g = (*g_)[r];
            if (cfg_.sgld_enabled)
                g += noise(rng_);
            double h = (*h_)[r];
            for (int c = 0; c < K_; ++c) {
                G_out[static_cast<size_t>(c)] += g;
                H_out[static_cast<size_t>(c)] += h;
            }
        }
        return {G_out, H_out};
    }

    bool select_goss_rows_(Node& n) {
        n.goss_top_indices_.clear();
        n.goss_rest_indices_.clear();
        n.goss_rest_scale = 1.0;
        n.goss_samples_valid_ = false;

        const int total = n.hi - n.lo;
        if (total <= 0)
            return false;

        auto [a, b] = compute_goss_rates_(n);
        const int k_top = std::clamp(static_cast<int>(std::round(a * total)), 1, total);
        const int k_rest = std::clamp(static_cast<int>(std::round(b * total)), 0, total - k_top);

        std::vector<std::pair<double, int>> ranked;
        ranked.reserve(total);
        std::normal_distribution<double> noise(0.0, cfg_.sgld_noise_scale);
        for (int i = n.lo; i < n.hi; ++i) {
            const int r = index_pool_[i];
            double g = (*g_)[r];
            if (cfg_.sgld_enabled)
                g += noise(rng_);
            ranked.emplace_back(std::abs(g), r);
        }
        std::ranges::sort(ranked, std::greater<>{}, &std::pair<double, int>::first);

        n.goss_top_indices_.reserve(k_top);
        n.goss_rest_indices_.reserve(k_rest);
        for (int i = 0; i < k_top; ++i) {
            n.goss_top_indices_.push_back(ranked[static_cast<size_t>(i)].second);
        }

        if (k_rest > 0) {
            if (cfg_.goss.use_random_rest) {
                std::vector<int> rest_pool;
                rest_pool.reserve(total - k_top);
                for (int i = k_top; i < total; ++i)
                    rest_pool.push_back(i);
                std::ranges::shuffle(rest_pool, rng_);
                for (int i = 0; i < k_rest; ++i) {
                    n.goss_rest_indices_.push_back(
                        ranked[static_cast<size_t>(rest_pool[static_cast<size_t>(i)])].second);
                }
            } else {
                const int rest_end = std::min(total, k_top + k_rest);
                for (int i = k_top; i < rest_end; ++i) {
                    n.goss_rest_indices_.push_back(ranked[static_cast<size_t>(i)].second);
                }
            }
        }

        n.goss_rest_scale = (1.0 - a) / std::max(b, 1e-15);
        n.goss_samples_valid_ = true;
        return true;
    }

    void accum_goss_weighted_(Node& n) {
        accum_goss_weighted_(n, should_use_goss_for_node_(n));
    }

    void accum_goss_weighted_(Node& n, bool use_goss) {
        n.uses_goss = use_goss;
        if (!n.uses_goss) {
            set_unweighted_node_totals_(n);
            return;
        }

        if (!select_goss_rows_(n)) {
            set_unweighted_node_totals_(n);
            return;
        }
        const auto [Gt, Ht] = sum_grad_hess_for_rows_(n.goss_top_indices_);
        const auto [Gr, Hr] = sum_grad_hess_for_rows_(n.goss_rest_indices_);

        n.goss_weighted_G.resize(Gt.size());
        n.goss_weighted_H.resize(Ht.size());
        for (size_t c = 0; c < Gt.size(); ++c) {
            n.goss_weighted_G[c] = Gt[c] + n.goss_rest_scale * Gr[c];
            n.goss_weighted_H[c] = cfg_.goss.scale_hessian ? (Ht[c] + n.goss_rest_scale * Hr[c]) : (Ht[c] + Hr[c]);
        }
    }

    // -------------------------- Scoring / priority ---------------------------
    double leaf_value_scalar_(double G, double H, double min_constraint = -std::numeric_limits<double>::infinity(),
                              double max_constraint = std::numeric_limits<double>::infinity()) const {
        double v = 0.0;
        if (H + cfg_.lambda_ > 0.0) {
            v = -splitx::soft(G, cfg_.alpha_) / (H + cfg_.lambda_);
        }
        double step = v;
        if (cfg_.max_delta_step > 0.0)
            step = std::clamp(step, -cfg_.max_delta_step, cfg_.max_delta_step);
        return std::clamp(step, min_constraint, max_constraint);
    }

    // Multiclass: compute K leaf values (one per class). For K=1, same as leaf_value_scalar_.
    std::vector<double> leaf_values_(double G, double H, double min_constraint, double max_constraint) const {
        std::vector<double> out;
        const int K_ = std::max(cfg_.num_classes - 1, 1);
        out.resize(static_cast<size_t>(K_));
        for (int c = 0; c < K_; ++c) {
            double v = 0.0;
            if (H + cfg_.lambda_ > 0.0) {
                v = -splitx::soft(G, cfg_.alpha_) / (H + cfg_.lambda_);
            }
            double step = v;
            if (cfg_.max_delta_step > 0.0)
                step = std::clamp(step, -cfg_.max_delta_step, cfg_.max_delta_step);
            out[static_cast<size_t>(c)] = std::clamp(step, min_constraint, max_constraint);
        }
        return out;
    }

    double leaf_objective_optimal_(double G, double H) const {
        return -0.5 * splitx::soft(G, cfg_.alpha_) * splitx::soft(G, cfg_.alpha_) / (H + cfg_.lambda_);
    }

    // Returns per-class G/H sums for multiclass, or {scalar, scalar} for K=1
    inline std::pair<std::vector<double>, std::vector<double>> node_totals_for_leaf_(const Node& n) const {
        const int K_ = std::max(n.K, 1);
        std::vector<double> G_out(static_cast<size_t>(K_), 0.0);
        std::vector<double> H_out(static_cast<size_t>(K_), 0.0);
        for (int c = 0; c < K_; ++c) {
            G_out[static_cast<size_t>(c)] =
                node_uses_goss_(n) ? n.goss_weighted_G[static_cast<size_t>(c)] : n.G[static_cast<size_t>(c)];
            H_out[static_cast<size_t>(c)] =
                node_uses_goss_(n) ? n.goss_weighted_H[static_cast<size_t>(c)] : n.H[static_cast<size_t>(c)];
        }
        return {G_out, H_out};
    }

    // Sum node G/H across K classes
    inline std::pair<double, double> node_totals_summed_(const Node& n) const {
        double G = 0.0, H = 0.0;
        const int K_ = std::max(n.K, 1);
        for (int c = 0; c < K_; ++c) {
            G += node_uses_goss_(n) ? n.goss_weighted_G[static_cast<size_t>(c)] : n.G[static_cast<size_t>(c)];
            H += node_uses_goss_(n) ? n.goss_weighted_H[static_cast<size_t>(c)] : n.H[static_cast<size_t>(c)];
        }
        return {G, H};
    }

    double priority_(double gain, const Node& nd) const {
        double pr = gain;
        if (cfg_.on_tree.enabled)
            pr -= cfg_.on_tree.ccp_alpha;
        if (cfg_.leaf_depth_penalty > 0.0)
            pr /= (1.0 + cfg_.leaf_depth_penalty * static_cast<double>(nd.depth));
        if (cfg_.leaf_hess_boost > 0.0) {
            // Use total hessian across classes for multiclass
            double H_sum = 0.0;
            for (size_t c = 0; c < nd.H.size(); ++c)
                H_sum += nd.H[c];
            pr *= (1.0 + cfg_.leaf_hess_boost * std::max(0.0, H_sum));
        }
        return pr;
    }

    // --------------------------- Feature sampling ----------------------------
    void build_feature_pool_() {
        std::vector<int> all(P_);
        std::iota(all.begin(), all.end(), 0);

        if (cfg_.feature_bagging_k > 0) {
            const int k = std::min(std::max(1, cfg_.feature_bagging_k), P_);
            feat_pool_ = sample_k_(all, k, cfg_.feature_bagging_with_replacement);
            return;
        }

        const int pct = std::clamp(cfg_.colsample_bytree_percent, 1, 100);
        if (pct >= 100) {
            feat_pool_ = std::move(all);
            return;
        }
        const int k = std::max(1, P_ * pct / 100);
        feat_pool_ = sample_k_(all, k, false);
    }

    std::vector<int> sample_k_(const std::vector<int>& pool, int k, bool with_replacement) const {
        if (k <= 0)
            return {};
        if (static_cast<int>(pool.size()) <= k && !with_replacement)
            return pool;

        std::vector<int> out;
        out.reserve(k);
        if (with_replacement) {
            std::uniform_int_distribution<int> J(0, static_cast<int>(pool.size()) - 1);
            for (int i = 0; i < k; ++i)
                out.push_back(pool[J(rng_)]);
            std::ranges::sort(out);
            out.erase(std::unique(out.begin(), out.end()), out.end());
        } else {
            out = pool;
            std::ranges::shuffle(out, rng_);
            out.resize(k);
            std::ranges::sort(out);
        }
        return out;
    }

    std::vector<int> select_features_() const {
        std::vector<int> pool = feat_pool_;
        if (pool.empty()) {
            pool.resize(P_);
            std::iota(pool.begin(), pool.end(), 0);
        }

        const int base = (cfg_.growth == TreeConfig::Growth::LeafWise ? cfg_.colsample_bynode_percent
                                                                      : cfg_.colsample_bylevel_percent);
        const int pct = std::clamp(base, 1, 100);
        if (pct >= 100)
            return pool;

        const int k = std::max(1, static_cast<int>(pool.size()) * pct / 100);
        return sample_k_(pool, k, false);
    }

    bool interaction_set_allowed_(const std::vector<int>& path, const std::vector<int>& proposed) const {
        if (cfg_.interaction_constraints.empty())
            return true;
        return std::ranges::any_of(cfg_.interaction_constraints, [&](const auto& group) {
            auto contains = [&](int feature) { return std::ranges::find(group, feature) != group.end(); };
            return std::ranges::all_of(path, contains) && std::ranges::all_of(proposed, contains);
        });
    }

    void filter_interaction_features_(std::vector<int>& features, const Node& node) const {
        if (cfg_.interaction_constraints.empty())
            return;
        std::erase_if(features, [&](int feature) {
            return !interaction_set_allowed_(node.path_features, std::vector<int>{feature});
        });
    }

    void apply_tree_level_row_subsample_(std::vector<int>& rows) {
        const double rate = cfg_.subsample_bytree;
        if (rate >= 1.0 || rows.empty())
            return;

        std::vector<int> out;
        out.reserve(static_cast<size_t>(std::ceil(rate * rows.size())));

        if (cfg_.subsample_importance_scale) {
            apply_importance_weighted_subsample_(rows, rate, out);
        } else if (!cfg_.subsample_with_replacement) {
            std::uniform_real_distribution<double> U(0.0, 1.0);
            for (int r : rows)
                if (U(rng_) < rate)
                    out.push_back(r);
        } else {
            const int k = std::max(1, static_cast<int>(std::round(rate * rows.size())));
            std::uniform_int_distribution<int> J(0, static_cast<int>(rows.size()) - 1);
            for (int i = 0; i < k; ++i)
                out.push_back(rows[J(rng_)]);
            std::ranges::sort(out);
            out.erase(std::unique(out.begin(), out.end()), out.end());
        }

        if (!out.empty())
            rows.swap(out);
    }

    void apply_importance_weighted_subsample_(const std::vector<int>& rows, double rate, std::vector<int>& out) {
        std::vector<double> weights;
        weights.reserve(rows.size());
        for (int r : rows)
            weights.push_back(std::abs((*g_)[r]) + 1e-10);

        const int k = std::max(1, static_cast<int>(std::round(rate * rows.size())));
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        for (int i = 0; i < k; ++i)
            out.push_back(rows[dist(rng_)]);
        std::ranges::sort(out);
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }

    // ------------------------------ Split API --------------------------------
    SplitHyper make_hyper_() const {
        SplitHyper hyp;
        hyp.lambda_ = cfg_.lambda_;
        hyp.alpha_ = cfg_.alpha_;
        hyp.gamma_ = cfg_.gamma_;
        hyp.min_samples_leaf_ = cfg_.min_samples_leaf;
        hyp.min_child_weight_ = cfg_.min_child_weight;

        switch (cfg_.missing_policy) {
            case TreeConfig::MissingPolicy::Learn:
                hyp.missing_policy = 0;
                break;
            case TreeConfig::MissingPolicy::AlwaysLeft:
                hyp.missing_policy = 1;
                break;
            case TreeConfig::MissingPolicy::AlwaysRight:
                hyp.missing_policy = 2;
                break;
        }
        hyp.leaf_gain_eps = cfg_.leaf_gain_eps;
        hyp.allow_zero_gain = cfg_.allow_zero_gain;
        return hyp;
    }

    const std::vector<int8_t>* maybe_monotone_() const {
        return (cfg_.monotone_constraints.size() == static_cast<size_t>(P_)) ? &cfg_.monotone_constraints : nullptr;
    }

    bool use_exact_for_(const Node& nd) const {
        if (!Xraw_)
            return false;
        if (cfg_.split_mode == TreeConfig::SplitMode::Exact)
            return true;
        if (cfg_.split_mode == TreeConfig::SplitMode::Hybrid)
            return nd.C <= cfg_.exact_cutover;
        return false;
    }

    // -------------------- Histogram Builder (NEW Helper) ---------------------
    class HistogramBuilder {
        const UnifiedTree& T;
        const std::vector<int>& index_pool;

        template <class RowAt>
        void dispatch_cpu_(int row_count, RowAt&& row_at, const std::vector<int>& feats, HistPair& hist) const {
            T.Xb_->visit_feature_major_codes([&](auto codes) {
                dispatch_feature_major_histogram(
                    T.unit_hessian_, codes, T.N_, row_count, std::forward<RowAt>(row_at), std::span<const int>(feats),
                    std::span<const size_t>(T.feature_offsets_), std::span<const int>(T.missing_ids_per_feat_),
                    std::span<const double>(*T.g_), std::span<const double>(*T.h_),
                    HistogramOutputView{hist.G, hist.H, hist.C}, *T.executor_);
            });
        }

    public:
        HistogramBuilder(const UnifiedTree& tree, const std::vector<int>& arena) : T(tree), index_pool(arena) {}

        void accumulate_bin(HistPair& hist, int r, int f) const {
            T.accumulate_hist_bin_(hist, r, f);
        }

#ifdef FORETREE_HAS_CUDA
        bool build_cuda(const std::vector<int>& rows, const std::vector<int>& feats, HistPair& hist) const {
            if (!T.cuda_histogram_engine_ || T.K_ != 1 || rows.empty() ||
                static_cast<int64_t>(rows.size()) * static_cast<int64_t>(feats.size()) < T.cfg_.cuda_min_histogram_work)
                return false;
            std::vector<uint32_t> device_rows;
            device_rows.reserve(rows.size());
            for (int row : rows)
                device_rows.push_back(static_cast<uint32_t>(row));
            const auto device_histogram = T.cuda_histogram_engine_->build_histogram(device_rows);
            for (int feature : feats) {
                const size_t begin = T.feature_offsets_[static_cast<size_t>(feature)];
                const size_t count = static_cast<size_t>(T.missing_ids_per_feat_[static_cast<size_t>(feature)] + 1);
                for (size_t index = begin; index < begin + count; ++index) {
                    hist.G[index] = static_cast<double>(device_histogram.gradients[index]);
                    hist.H[index] = static_cast<double>(device_histogram.hessians[index]);
                    hist.C[index] = static_cast<int>(device_histogram.counts[index]);
                }
            }
            return true;
        }
#endif

        void build_for_rows(const std::vector<int>& rows, const std::vector<int>& feats, HistPair& hist) const {
#ifdef FORETREE_HAS_CUDA
            if (build_cuda(rows, feats, hist))
                return;
#endif
            dispatch_cpu_(static_cast<int>(rows.size()), [&](int position) { return rows[position]; }, feats, hist);
        }

        void build_for_range(int lo, int hi, const std::vector<int>& feats, HistPair& hist) const {
#ifdef FORETREE_HAS_CUDA
            if (T.cuda_histogram_engine_) {
                std::vector<int> rows;
                rows.reserve(static_cast<size_t>(hi - lo));
                for (int index = lo; index < hi; ++index)
                    rows.push_back(index_pool[static_cast<size_t>(index)]);
                if (build_cuda(rows, feats, hist))
                    return;
            }
#endif
            dispatch_cpu_(
                hi - lo, [&](int position) { return index_pool[static_cast<size_t>(lo + position)]; }, feats, hist);
        }
    };

    // ------------------------- Split Providers (Refactored) ------------------
    struct HistogramProvider {
        const UnifiedTree& T;
        const std::vector<int>& index_pool;

        explicit HistogramProvider(const UnifiedTree& tree, const std::vector<int>& arena)
            : T(tree), index_pool(arena) {}

        std::shared_ptr<HistPair> build_histogram(const Node& nd, const std::vector<int>& feats) const {
            const bool goss_here = T.node_uses_goss_(nd);

            if (nd.hist_valid && nd.histogram && nd.hist_features == feats && nd.hist_goss_weighted == goss_here)
                return nd.histogram;

            std::shared_ptr<HistPair> hist;
            const bool is_full_root = nd.lo == 0 && nd.hi == static_cast<int>(T.index_pool_.size());
            if (!goss_here && is_full_root && T.tree_histogram_ && feats == T.tree_features_ &&
                !T.tree_subsample_applied_) {
                hist = T.tree_histogram_;
            } else {
                hist = T.acquire_histogram_();
            }

            if (!is_full_root && !goss_here && T.tree_histogram_ && feats == T.tree_features_ &&
                nd.C >= T.cfg_.cache_threshold && !T.tree_subsample_applied_) {
                derive_from_tree_histogram_(nd, feats, *hist);
            } else if (hist != T.tree_histogram_) {
                build_from_rows_(nd, feats, *hist, goss_here);
            }

            hist->goss_weighted = goss_here;

            if (T.cfg_.cache_histograms) {
                nd.histogram = hist;
                nd.hist_features = feats;
                nd.hist_valid = true;
                nd.hist_goss_weighted = goss_here;
            }
            return hist;
        }

    private:
        void derive_from_tree_histogram_(const Node& nd, const std::vector<int>& feats, HistPair& hist) const {
            HistogramBuilder builder(T, T.index_pool_);
            const int total_rows = static_cast<int>(T.index_pool_.size());
            const int node_rows = nd.hi - nd.lo;

            // Build directly from node rows when the node is smaller than its complement.
            if (node_rows * 2 <= total_rows) {
                hist.clear();
                builder.build_for_range(nd.lo, nd.hi, feats, hist);
                return;
            }

            auto complement = std::make_unique<HistPair>();
            complement->resize(T.total_hist_size_);
            complement->clear();
            builder.build_for_range(0, nd.lo, feats, *complement);
            builder.build_for_range(nd.hi, total_rows, feats, *complement);

            hist.G = T.tree_histogram_->G;
            hist.H = T.tree_histogram_->H;
            hist.C = T.tree_histogram_->C;
            hist.subtract(*complement);
        }

        void build_from_rows_(const Node& nd, const std::vector<int>& feats, HistPair& hist, bool use_goss) const {
            const std::vector<int>& work_rows = subsample_for_node_(nd);
            if (use_goss) {
                build_with_goss_(nd, feats, hist);
            } else if (!work_rows.empty()) {
                HistogramBuilder(T, index_pool).build_for_rows(work_rows, feats, hist);
            } else {
                HistogramBuilder(T, index_pool).build_for_range(nd.lo, nd.hi, feats, hist);
            }
        }

        const std::vector<int>& subsample_for_node_(const Node& nd) const {
            auto& out = T.training_context_.arena->histogram_rows;
            out.clear();
            double rate = T.cfg_.subsample_bynode;
            if (T.cfg_.growth == TreeConfig::Growth::LevelWise && T.cfg_.subsample_bylevel < 1.0)
                rate = T.cfg_.subsample_bylevel;
            if (rate >= 1.0)
                return out;

            out.reserve(static_cast<size_t>(nd.hi - nd.lo));
            std::uniform_real_distribution<double> U(0.0, 1.0);
            for (int i = nd.lo; i < nd.hi; ++i) {
                const int r = index_pool[static_cast<size_t>(i)];
                if (U(T.rng_) < rate)
                    out.push_back(r);
            }
            return out;
        }

        void build_with_goss_(const Node& nd, const std::vector<int>& feats, HistPair& hist) const {
            const int total = nd.hi - nd.lo;
            if (total < T.cfg_.goss.min_node_size || !nd.goss_samples_valid_ || nd.goss_top_indices_.empty()) {
                HistogramBuilder(T, index_pool).build_for_range(nd.lo, nd.hi, feats, hist);
                return;
            }

            auto top_hist = T.hist_pool_ ? T.hist_pool_->get() : std::make_unique<HistPair>();
            auto rest_hist = T.hist_pool_ ? T.hist_pool_->get() : std::make_unique<HistPair>();
            int K_ = std::max(T.K_, 1);
            top_hist->resize(T.total_hist_size_, K_);
            top_hist->clear();
            rest_hist->resize(T.total_hist_size_, K_);
            rest_hist->clear();

            HistogramBuilder builder(T, index_pool);
            builder.build_for_rows(nd.goss_top_indices_, feats, *top_hist);
            if (!nd.goss_rest_indices_.empty())
                builder.build_for_rows(nd.goss_rest_indices_, feats, *rest_hist);

            for (size_t i = 0; i < hist.G.size(); ++i) {
                hist.G[i] = top_hist->G[i] + nd.goss_rest_scale * rest_hist->G[i];
                hist.H[i] = T.cfg_.goss.scale_hessian ? (top_hist->H[i] + nd.goss_rest_scale * rest_hist->H[i])
                                                      : (top_hist->H[i] + rest_hist->H[i]);
                hist.C[i] = top_hist->C[i] + rest_hist->C[i];
            }
            if (T.hist_pool_) {
                T.hist_pool_->return_histogram(std::move(top_hist));
                T.hist_pool_->return_histogram(std::move(rest_hist));
            }
        }

    public:
        foretree::splitx::Candidate best_split(const Node& nd, const SplitHyper& hyp, const std::vector<int8_t>* mono) {
            auto feats = nd.hist_features.empty() ? T.select_features_() : nd.hist_features;
            T.filter_interaction_features_(feats, nd);
            auto hist = build_histogram(nd, feats);

            int K_ = std::max(static_cast<int>(hist->K), 1);
            std::vector<double> Gtot_vec(static_cast<size_t>(K_), 0.0);
            std::vector<double> Htot_vec(static_cast<size_t>(K_), 0.0);
            for (int c = 0; c < K_; ++c) {
                Gtot_vec[static_cast<size_t>(c)] =
                    nd.uses_goss ? nd.goss_weighted_G[static_cast<size_t>(c)] : nd.G[static_cast<size_t>(c)];
                Htot_vec[static_cast<size_t>(c)] =
                    nd.uses_goss ? nd.goss_weighted_H[static_cast<size_t>(c)] : nd.H[static_cast<size_t>(c)];
            }
            double Gtot = 0.0, Htot = 0.0;
            for (int c = 0; c < K_; ++c) {
                Gtot += Gtot_vec[static_cast<size_t>(c)];
                Htot += Htot_vec[static_cast<size_t>(c)];
            }

            foretree::splitx::SplitContext ctx;
            ctx.G = &hist->G;
            ctx.H = &hist->H;
            ctx.C = &hist->C;
            ctx.P = T.P_;
            ctx.B = 0;
            ctx.K = K_;
            ctx.total_hist_size = T.total_hist_size_;
            ctx.Gp = Gtot;
            ctx.Hp = Htot;
            ctx.Cp = nd.C;
            ctx.monotone = mono;
            ctx.active_features = &feats;
            ctx.feature_types = T.Xb_->feature_types().data();
            ctx.hyp = hyp;
            ctx.variable_bins = true;
            ctx.feature_offsets = T.feature_offsets_.data();
            ctx.finite_bins_per_feat = T.finite_bins_per_feat_.data();
            ctx.missing_ids_per_feat = T.missing_ids_per_feat_.data();
            ctx.row_g = T.g_->data();
            ctx.row_h = T.h_->data();
            ctx.Xb = {T.Xb8_, T.Xb16_};
            ctx.row_index = T.index_pool_.data() + nd.lo;
            ctx.N = nd.hi - nd.lo;
            ctx.bin_centers = T.bin_centers_flat_.data();
            ctx.Bz = std::max(16, std::min(512, T.cfg_.n_bins));

#ifdef FORETREE_HAS_CUDA
            struct JointCudaState {
                cuda::CudaHistogramEngine* engine;
                const int* rows;
                int row_count;
            } joint_cuda_state{T.cuda_histogram_engine_, ctx.row_index, ctx.N};
            if (joint_cuda_state.engine && !nd.uses_goss) {
                ctx.joint_histogram_state = &joint_cuda_state;
                ctx.joint_histogram_builder = [](void* opaque, const int* interleaved_pairs, int pair_count,
                                                 int reduced_bins, splitx::JointHistogramBatch& output) -> bool {
                    auto& state = *static_cast<JointCudaState*>(opaque);
                    static thread_local std::vector<cuda::FeaturePair> pairs;
                    static thread_local std::vector<uint32_t> rows;
                    pairs.resize(static_cast<size_t>(pair_count));
                    for (int pair = 0; pair < pair_count; ++pair) {
                        pairs[static_cast<size_t>(pair)] =
                            cuda::FeaturePair{static_cast<uint32_t>(interleaved_pairs[2 * pair]),
                                              static_cast<uint32_t>(interleaved_pairs[2 * pair + 1])};
                    }
                    rows.resize(static_cast<size_t>(state.row_count));
                    std::transform(state.rows, state.rows + state.row_count, rows.begin(),
                                   [](int row) { return static_cast<uint32_t>(row); });
                    const auto result = state.engine->build_joint_histograms(pairs, reduced_bins, rows);
                    output.reduced_bins = result.reduced_bins;
                    output.gradients.assign(result.gradients.begin(), result.gradients.end());
                    output.hessians.assign(result.hessians.begin(), result.hessians.end());
                    output.counts.assign(result.counts.begin(), result.counts.end());
                    return result.pair_count == static_cast<size_t>(pair_count);
                };
            }
#endif

            auto ecfg = T.make_split_engine_config_();
            // Pair histograms currently consume row-level gradients directly;
            // disable them for GOSS nodes until row sampling weights are part
            // of SplitContext, otherwise their gain would use inconsistent
            // parent and child statistics.
            if (nd.uses_goss)
                ecfg.enable_pair_interactions = false;

            auto result = Splitter::best_split(ctx, SplitEngine::Histogram, ecfg);
            return result;
        }
    };

    struct ExactProvider {
        const UnifiedTree& T;
        const std::vector<int>& index_pool;

        explicit ExactProvider(const UnifiedTree& tree, const std::vector<int>& arena) : T(tree), index_pool(arena) {}

        void build_missing_aggregates_(const Node& nd, std::vector<double>& Gmiss, std::vector<double>& Hmiss,
                                       std::vector<int>& Cmiss) const {
            Gmiss.assign(static_cast<size_t>(T.P_), 0.0);
            Hmiss.assign(static_cast<size_t>(T.P_), 0.0);
            Cmiss.assign(T.P_, 0);

            if (!T.Xraw_)
                return;

            const bool has_mask = (T.Xmiss_ != nullptr);
            for (int i = nd.lo; i < nd.hi; ++i) {
                const int r = index_pool[i];
                const size_t row = static_cast<size_t>(r) * static_cast<size_t>(T.P_);
                for (int f = 0; f < T.P_; ++f) {
                    const bool miss = has_mask ? (T.Xmiss_[row + static_cast<size_t>(f)] != 0)
                                               : !std::isfinite(T.Xraw_[row + static_cast<size_t>(f)]);
                    if (miss) {
                        Gmiss[static_cast<size_t>(f)] += (*T.g_)[r];
                        Hmiss[static_cast<size_t>(f)] += (*T.h_)[r];
                        Cmiss[f] += 1;
                    }
                }
            }
        }

        foretree::splitx::Candidate best_split(const Node& nd, const SplitHyper& hyp, const std::vector<int8_t>* mono) {
            auto features = T.select_features_();
            T.filter_interaction_features_(features, nd);
            int K_ = std::max(static_cast<int>(nd.K), 1);
            std::vector<double> Gtot_vec(static_cast<size_t>(K_), 0.0);
            std::vector<double> Htot_vec(static_cast<size_t>(K_), 0.0);
            for (int c = 0; c < K_; ++c) {
                Gtot_vec[static_cast<size_t>(c)] =
                    nd.uses_goss ? nd.goss_weighted_G[static_cast<size_t>(c)] : nd.G[static_cast<size_t>(c)];
                Htot_vec[static_cast<size_t>(c)] =
                    nd.uses_goss ? nd.goss_weighted_H[static_cast<size_t>(c)] : nd.H[static_cast<size_t>(c)];
            }
            double Gtot = 0.0, Htot = 0.0;
            for (int c = 0; c < K_; ++c) {
                Gtot += Gtot_vec[static_cast<size_t>(c)];
                Htot += Htot_vec[static_cast<size_t>(c)];
            }

            foretree::splitx::SplitContext ctx;
            ctx.P = T.P_;
            ctx.B = 0;
            ctx.K = K_;
            ctx.Gp = Gtot;
            ctx.Hp = Htot;
            ctx.Cp = nd.C;
            ctx.monotone = mono;
            ctx.active_features = &features;
            ctx.feature_types = T.Xb_->feature_types().data();
            ctx.hyp = hyp;
            ctx.row_g = T.g_->data();
            ctx.row_h = T.h_->data();

            auto& scratch = T.training_context_.arena->exact;
            auto& Gmiss = scratch.missing_gradients;
            auto& Hmiss = scratch.missing_hessians;
            auto& Cmiss = scratch.missing_counts;
            build_missing_aggregates_(nd, Gmiss, Hmiss, Cmiss);

            bool any_missing = std::ranges::any_of(Cmiss, [](int c) { return c > 0; });
            ctx.has_missing = any_missing;
            ctx.Gmiss = any_missing ? Gmiss.data() : nullptr;
            ctx.Hmiss = any_missing ? Hmiss.data() : nullptr;
            ctx.Cmiss = any_missing ? Cmiss.data() : nullptr;

            auto& rows = scratch.rows;
            rows.clear();
            rows.reserve(static_cast<size_t>(nd.hi - nd.lo));
            for (int i = nd.lo; i < nd.hi; ++i)
                rows.push_back(index_pool[i]);

            auto ecfg = T.make_split_engine_config_();
            ecfg.enable_categorical_partition = false;

            const bool needs_local_oblique_data = ecfg.enable_oblique && ecfg.oblique_mode != ObliqueMode::Off;
            if (!needs_local_oblique_data) {
                return Splitter::best_split(ctx, SplitEngine::Exact, ecfg, T.Xraw_, T.P_, rows.data(),
                                            static_cast<int>(rows.size()), hyp.missing_policy, T.Xmiss_);
            }

            const int n_local = static_cast<int>(rows.size());
            if (n_local <= 1) {
                return Splitter::best_split(ctx, SplitEngine::Exact, ecfg, T.Xraw_, T.P_, rows.data(), n_local,
                                            hyp.missing_policy, T.Xmiss_);
            }

            auto& local_X = scratch.local_matrix;
            auto& local_g = scratch.local_gradients;
            auto& local_h = scratch.local_hessians;
            auto& local_miss = scratch.local_missing;
            const size_t local_cells = static_cast<size_t>(n_local) * static_cast<size_t>(T.P_);
            local_X.resize(local_cells);
            local_g.resize(static_cast<size_t>(n_local));
            local_h.resize(static_cast<size_t>(n_local));
            local_miss.clear();
            if (T.Xmiss_) {
                local_miss.resize(local_cells);
            }

            auto& col_storage = scratch.column_storage;
            auto& xcols = scratch.columns;
            auto& local_idx = scratch.local_indices;
            col_storage.resize(local_cells);
            xcols.resize(static_cast<size_t>(T.P_));
            local_idx.resize(static_cast<size_t>(n_local));
            std::iota(local_idx.begin(), local_idx.end(), 0);

            for (int i = 0; i < n_local; ++i) {
                const int r = rows[static_cast<size_t>(i)];
                const size_t row_local = static_cast<size_t>(i) * static_cast<size_t>(T.P_);
                const size_t row_global = static_cast<size_t>(r) * static_cast<size_t>(T.P_);
                for (int f = 0; f < T.P_; ++f) {
                    const size_t idx_local = row_local + static_cast<size_t>(f);
                    const size_t idx_global = row_global + static_cast<size_t>(f);
                    const double x = T.Xraw_[idx_global];
                    local_X[idx_local] = x;
                    col_storage[static_cast<size_t>(f) * static_cast<size_t>(n_local) + static_cast<size_t>(i)] = x;
                    if (T.Xmiss_)
                        local_miss[idx_local] = T.Xmiss_[idx_global];
                }
                local_g[static_cast<size_t>(i)] = (*T.g_)[r];
                local_h[static_cast<size_t>(i)] = (*T.h_)[r];
            }

            for (int f = 0; f < T.P_; ++f) {
                xcols[static_cast<size_t>(f)] =
                    col_storage.data() + static_cast<size_t>(f) * static_cast<size_t>(n_local);
            }

            ctx.row_g = local_g.data();
            ctx.row_h = local_h.data();
            ctx.N = n_local;
            ctx.Xcols = xcols.data();

            // Recompute missing aggregates in local indexing when we switch to local row_g/row_h.
            auto& local_Gmiss = scratch.local_missing_gradients;
            auto& local_Hmiss = scratch.local_missing_hessians;
            auto& local_Cmiss = scratch.local_missing_counts;
            local_Gmiss.assign(static_cast<size_t>(T.P_), 0.0);
            local_Hmiss.assign(static_cast<size_t>(T.P_), 0.0);
            local_Cmiss.assign(static_cast<size_t>(T.P_), 0);
            for (int i = 0; i < n_local; ++i) {
                const size_t row = static_cast<size_t>(i) * static_cast<size_t>(T.P_);
                for (int f = 0; f < T.P_; ++f) {
                    const size_t idx = row + static_cast<size_t>(f);
                    const bool miss = T.Xmiss_ ? (local_miss[idx] != 0) : !std::isfinite(local_X[idx]);
                    if (miss) {
                        local_Gmiss[static_cast<size_t>(f)] += local_g[static_cast<size_t>(i)];
                        local_Hmiss[static_cast<size_t>(f)] += local_h[static_cast<size_t>(i)];
                        local_Cmiss[static_cast<size_t>(f)] += 1;
                    }
                }
            }
            const bool any_local_missing = std::ranges::any_of(local_Cmiss, [](int c) { return c > 0; });
            ctx.has_missing = any_local_missing;
            ctx.Gmiss = any_local_missing ? local_Gmiss.data() : nullptr;
            ctx.Hmiss = any_local_missing ? local_Hmiss.data() : nullptr;
            ctx.Cmiss = any_local_missing ? local_Cmiss.data() : nullptr;

            return Splitter::best_split(ctx, SplitEngine::Exact, ecfg, local_X.data(), T.P_, local_idx.data(), n_local,
                                        hyp.missing_policy, T.Xmiss_ ? local_miss.data() : nullptr);
        }
    };

    // Template method for split evaluation
    template <class Provider> bool eval_with_provider_(Node& nd, foretree::splitx::Candidate& out) {
        if (nd.C < cfg_.min_samples_split || nd.depth >= cfg_.max_depth)
            return false;

        Provider prov(*this, index_pool_);
        const auto* mono = maybe_monotone_();
        const SplitHyper hyp = make_hyper_();

        auto cand = prov.best_split(nd, hyp, mono);
        if (!std::isfinite(cand.gain))
            return false;
        std::vector<int> proposed_features;
        if (cand.kind == splitx::SplitKind::Oblique)
            proposed_features = cand.oblique_features;
        else if (cand.kind == splitx::SplitKind::PairInteraction)
            proposed_features = {cand.pair_feature_a, cand.pair_feature_b};
        else
            proposed_features = {cand.feat};
        if (!interaction_set_allowed_(nd.path_features, proposed_features))
            return false;
        if (cand.kind == splitx::SplitKind::Axis) {
            if (cand.feat < 0)
                return false;
            if (!std::isfinite(cand.split_value) && cand.thr < 0)
                return false;
        } else if (cand.kind == splitx::SplitKind::CategoricalPartition) {
            if (cand.feat < 0)
                return false;
            if (cand.categorical_left_bins.empty())
                return false;
        } else if (cand.kind == splitx::SplitKind::Oblique) {
            if (cand.oblique_features.empty())
                return false;
            if (cand.oblique_features.size() != cand.oblique_weights.size())
                return false;
            if (!std::isfinite(cand.oblique_threshold))
                return false;
        } else if (cand.kind == splitx::SplitKind::PairInteraction) {
            if (cand.pair_feature_a < 0 || cand.pair_feature_b < 0 || cand.pair_feature_a == cand.pair_feature_b)
                return false;
            if (cand.pair_threshold_a < 0 || cand.pair_threshold_b < 0 || cand.pair_quadrant_mask == 0 ||
                cand.pair_quadrant_mask == 15)
                return false;
        } else {
            return false;
        }

        out = cand;
        return true;
    }

    bool eval_node_split_(Node& nd, foretree::splitx::Candidate& out) {
        if (use_exact_for_(nd)) {
            return eval_with_provider_<ExactProvider>(nd, out);
        } else {
            return eval_with_provider_<HistogramProvider>(nd, out);
        }
    }

    inline uint16_t code_at_(int row, int feature) const noexcept {
        const size_t offset = static_cast<size_t>(row) * static_cast<size_t>(P_) + static_cast<size_t>(feature);
        return Xb8_ ? static_cast<uint16_t>(Xb8_[offset]) : Xb16_[offset];
    }

    template <class Code>
    inline double predict_one_compact_(const Code* row_binned, int row_idx, const double* Xraw_opt) const {
        PredictRawView raw_view = resolve_predict_raw_view_(Xraw_opt);
        int id = root_id_;
        while (!packed_tree_.leaf_flags[static_cast<size_t>(id)]) {
            const int feature = packed_tree_.features[static_cast<size_t>(id)];
            const bool missing_left = packed_tree_.missing_left[static_cast<size_t>(id)] != 0;
            bool go_left = false;
            if (packed_tree_.split_kinds[static_cast<size_t>(id)] == static_cast<uint8_t>(splitx::SplitKind::Oblique)) {
                bool missing = false;
                double projection = 0.0;
                const int offset = packed_tree_.oblique_offsets[static_cast<size_t>(id)];
                const int count = packed_tree_.oblique_counts[static_cast<size_t>(id)];
                for (int i = 0; i < count; ++i) {
                    const int f = packed_tree_.oblique_features[static_cast<size_t>(offset + i)];
                    double value =
                        raw_view.Xraw
                            ? raw_view
                                  .Xraw[static_cast<size_t>(row_idx) * static_cast<size_t>(P_) + static_cast<size_t>(f)]
                            : binned_value_for_feature_(f, static_cast<uint16_t>(row_binned[f]));
                    if (!std::isfinite(value)) {
                        missing = true;
                        break;
                    }
                    projection += packed_tree_.oblique_weights[static_cast<size_t>(offset + i)] * value;
                }
                go_left =
                    missing ? missing_left : projection <= packed_tree_.oblique_thresholds[static_cast<size_t>(id)];
            } else if (packed_tree_.split_kinds[static_cast<size_t>(id)] ==
                       static_cast<uint8_t>(splitx::SplitKind::PairInteraction)) {
                const int fa = packed_tree_.pair_features_a[static_cast<size_t>(id)];
                const int fb = packed_tree_.pair_features_b[static_cast<size_t>(id)];
                const uint16_t a = static_cast<uint16_t>(row_binned[fa]);
                const uint16_t b = static_cast<uint16_t>(row_binned[fb]);
                if (a == static_cast<uint16_t>(missing_ids_per_feat_[fa]) ||
                    b == static_cast<uint16_t>(missing_ids_per_feat_[fb])) {
                    go_left = missing_left;
                } else {
                    const int quadrant = (a > packed_tree_.pair_thresholds_a[static_cast<size_t>(id)] ? 2 : 0) |
                                         (b > packed_tree_.pair_thresholds_b[static_cast<size_t>(id)] ? 1 : 0);
                    go_left =
                        (packed_tree_.pair_quadrant_masks[static_cast<size_t>(id)] & (uint8_t{1} << quadrant)) != 0;
                }
            } else {
                const uint16_t code = static_cast<uint16_t>(row_binned[feature]);
                const size_t raw_offset =
                    static_cast<size_t>(row_idx) * static_cast<size_t>(P_) + static_cast<size_t>(feature);
                const bool exact_axis = raw_view.Xraw &&
                                        packed_tree_.split_kinds[static_cast<size_t>(id)] ==
                                            static_cast<uint8_t>(splitx::SplitKind::Axis) &&
                                        std::isfinite(packed_tree_.split_values[static_cast<size_t>(id)]);
                if (exact_axis) {
                    const double value = raw_view.Xraw[raw_offset];
                    const bool missing = raw_view.Xmiss ? raw_view.Xmiss[raw_offset] != 0 : !std::isfinite(value);
                    go_left = missing ? missing_left : value <= packed_tree_.split_values[static_cast<size_t>(id)];
                } else if (code == static_cast<uint16_t>(missing_ids_per_feat_[feature])) {
                    go_left = missing_left;
                } else if (packed_tree_.split_kinds[static_cast<size_t>(id)] ==
                           static_cast<uint8_t>(splitx::SplitKind::CategoricalPartition)) {
                    const int offset = packed_tree_.categorical_offsets[static_cast<size_t>(id)];
                    const int count = packed_tree_.categorical_counts[static_cast<size_t>(id)];
                    go_left = std::binary_search(packed_tree_.categorical_bins.begin() + offset,
                                                 packed_tree_.categorical_bins.begin() + offset + count,
                                                 static_cast<int>(code));
                } else {
                    go_left = code <= static_cast<uint16_t>(packed_tree_.thresholds[static_cast<size_t>(id)]);
                }
            }
            id = go_left ? packed_tree_.left_children[static_cast<size_t>(id)]
                         : packed_tree_.right_children[static_cast<size_t>(id)];
        }
        return packed_tree_.leaf_values[static_cast<size_t>(id) * static_cast<size_t>(K_)];
    }

    double binned_value_for_feature_(int feat, uint16_t code) const {
        if (feat < 0 || feat >= P_)
            return std::numeric_limits<double>::quiet_NaN();
        const int miss = missing_ids_per_feat_[feat];
        if (code == static_cast<uint16_t>(miss))
            return std::numeric_limits<double>::quiet_NaN();
        const int finite = finite_bins_per_feat_[feat];
        if (code >= static_cast<uint16_t>(finite))
            return std::numeric_limits<double>::quiet_NaN();
        const size_t off = feature_offsets_[feat] + static_cast<size_t>(code);
        if (off < bin_centers_flat_.size())
            return bin_centers_flat_[off];
        return std::numeric_limits<double>::quiet_NaN();
    }

    // ------------------------------- Partition --------------------------------
    int partition_hist_(Node& nd, int feat, int thr, bool miss_left) {
        const uint16_t miss = static_cast<uint16_t>(missing_ids_per_feat_[feat]);
        return RowPartitioner::partition(index_pool_, nd.lo, nd.hi, [&](int row) {
            const uint16_t bin = code_at_(row, feat);
            return bin == miss ? miss_left : bin <= static_cast<uint16_t>(thr);
        });
    }

    int partition_hist_categorical_(Node& nd, int feat, const std::vector<int>& categorical_left_bins, bool miss_left) {
        const uint16_t miss = static_cast<uint16_t>(missing_ids_per_feat_[feat]);
        return RowPartitioner::partition(index_pool_, nd.lo, nd.hi, [&](int row) {
            const uint16_t bin = code_at_(row, feat);
            if (bin == miss)
                return miss_left;
            return std::binary_search(categorical_left_bins.begin(), categorical_left_bins.end(),
                                      static_cast<int>(bin));
        });
    }

    int partition_oblique_(Node& nd, const foretree::splitx::Candidate& sp) {
        auto go_left = [&](int r) -> bool {
            double z = 0.0;
            bool miss = false;
            for (size_t t = 0; t < sp.oblique_features.size(); ++t) {
                const int f = sp.oblique_features[t];
                double x = std::numeric_limits<double>::quiet_NaN();
                if (Xraw_) {
                    const size_t off = static_cast<size_t>(r) * static_cast<size_t>(P_) + static_cast<size_t>(f);
                    x = Xraw_[off];
                    if (Xmiss_ && Xmiss_[off] != 0)
                        x = std::numeric_limits<double>::quiet_NaN();
                } else {
                    const uint16_t code = code_at_(r, f);
                    x = binned_value_for_feature_(f, code);
                }
                if (!std::isfinite(x)) {
                    miss = true;
                    break;
                }
                z += sp.oblique_weights[t] * x;
            }
            if (miss)
                return sp.oblique_missing_left;
            return z <= sp.oblique_threshold;
        };

        return RowPartitioner::partition(index_pool_, nd.lo, nd.hi, go_left);
    }

    int partition_pair_interaction_(Node& nd, const foretree::splitx::Candidate& sp) {
        return RowPartitioner::partition(index_pool_, nd.lo, nd.hi, [&](int row) {
            const uint16_t a = code_at_(row, sp.pair_feature_a);
            const uint16_t b = code_at_(row, sp.pair_feature_b);
            if (a == static_cast<uint16_t>(missing_ids_per_feat_[sp.pair_feature_a]) ||
                b == static_cast<uint16_t>(missing_ids_per_feat_[sp.pair_feature_b]))
                return sp.pair_missing_left;
            const int quadrant = (a > sp.pair_threshold_a ? 2 : 0) | (b > sp.pair_threshold_b ? 1 : 0);
            return (sp.pair_quadrant_mask & (uint8_t{1} << quadrant)) != 0;
        });
    }

    int partition_exact_(Node& nd, int feat, double split_value, bool miss_left) {
        const size_t stride = static_cast<size_t>(P_);
        const size_t off = static_cast<size_t>(feat);
        const bool has_mask = (Xmiss_ != nullptr);
        return RowPartitioner::partition(index_pool_, nd.lo, nd.hi, [&](int row) {
            const size_t index = static_cast<size_t>(row) * stride + off;
            const double value = Xraw_[index];
            const bool missing = has_mask ? Xmiss_[index] != 0 : !std::isfinite(value);
            return missing ? miss_left : value <= split_value;
        });
    }

    void apply_split_(Node& nd, const foretree::splitx::Candidate& sp) {
        std::vector<int> categorical_left_bins;
        if (sp.kind == splitx::SplitKind::CategoricalPartition) {
            categorical_left_bins = sp.categorical_left_bins;
            std::ranges::sort(categorical_left_bins);
            categorical_left_bins.erase(std::unique(categorical_left_bins.begin(), categorical_left_bins.end()),
                                        categorical_left_bins.end());
        }

        const int mid = (sp.kind == splitx::SplitKind::CategoricalPartition)
                            ? partition_hist_categorical_(nd, sp.feat, categorical_left_bins, sp.miss_left)
                        : (sp.kind == splitx::SplitKind::Oblique) ? partition_oblique_(nd, sp)
                        : (sp.kind == splitx::SplitKind::PairInteraction)
                            ? partition_pair_interaction_(nd, sp)
                            : ((std::isfinite(sp.split_value) && Xraw_)
                                   ? partition_exact_(nd, sp.feat, sp.split_value, sp.miss_left)
                                   : partition_hist_(nd, sp.feat, sp.thr, sp.miss_left));

        Node ln, rn;
        ln.id = next_id_++;
        rn.id = next_id_++;
        ln.K = K_;
        rn.K = K_;
        ln.depth = nd.depth + 1;
        rn.depth = nd.depth + 1;
        ln.lo = nd.lo;
        ln.hi = mid;
        rn.lo = mid;
        rn.hi = nd.hi;

        ln.path_features = nd.path_features;
        if (sp.kind == splitx::SplitKind::Oblique) {
            ln.path_features.insert(ln.path_features.end(), sp.oblique_features.begin(), sp.oblique_features.end());
        } else if (sp.kind == splitx::SplitKind::PairInteraction) {
            ln.path_features.push_back(sp.pair_feature_a);
            ln.path_features.push_back(sp.pair_feature_b);
        } else {
            ln.path_features.push_back(sp.feat);
        }
        std::ranges::sort(ln.path_features);
        ln.path_features.erase(std::unique(ln.path_features.begin(), ln.path_features.end()), ln.path_features.end());
        rn.path_features = ln.path_features;

        ln.min_constraint = nd.min_constraint;
        ln.max_constraint = nd.max_constraint;
        rn.min_constraint = nd.min_constraint;
        rn.max_constraint = nd.max_constraint;

        if (sp.kind == splitx::SplitKind::Axis && sp.feat >= 0 &&
            cfg_.monotone_constraints.size() > static_cast<size_t>(sp.feat)) {
            const int8_t mono = cfg_.monotone_constraints[sp.feat];
            if (mono != 0) {
                // To compute the exact mid, we'd need the left and right weights, but they aren't computed here
                // cleanly. We'll approximate the `mid` bound directly using the parent's actual weight, which
                // guarantees safety and simplicity.
                const double node_weight = leaf_value_scalar_(nd.G[0], nd.H[0], nd.min_constraint, nd.max_constraint);
                const double mid_bound = std::clamp(node_weight, nd.min_constraint, nd.max_constraint);
                if (mono > 0) {
                    ln.max_constraint = mid_bound;
                    rn.min_constraint = mid_bound;
                } else if (mono < 0) {
                    ln.min_constraint = mid_bound;
                    rn.max_constraint = mid_bound;
                }
            }
        }

        accum_(ln);
        accum_(rn);
        accum_goss_weighted_(ln);
        accum_goss_weighted_(rn);

        nd.is_leaf = false;
        nd.feature = sp.feat;
        nd.thr = sp.thr;
        nd.split_value = sp.split_value;
        nd.miss_left = sp.miss_left;
        nd.split_kind = sp.kind;
        nd.categorical_left_bins = std::move(categorical_left_bins);
        nd.oblique_features = sp.oblique_features;
        nd.oblique_weights = sp.oblique_weights;
        nd.oblique_threshold = sp.oblique_threshold;
        nd.oblique_missing_left = sp.oblique_missing_left;
        nd.pair_feature_a = sp.pair_feature_a;
        nd.pair_feature_b = sp.pair_feature_b;
        nd.pair_threshold_a = sp.pair_threshold_a;
        nd.pair_threshold_b = sp.pair_threshold_b;
        nd.pair_quadrant_mask = sp.pair_quadrant_mask;
        nd.pair_missing_left = sp.pair_missing_left;
        nd.left = ln.id;
        nd.right = rn.id;
        nd.best_gain = sp.gain;
        ln.sibling = rn.id;
        rn.sibling = ln.id;

        nd.goss_samples_valid_ = false;
        nd.goss_top_indices_.clear();
        nd.goss_rest_indices_.clear();
        nd.goss_rest_scale = 1.0;

        if (sp.feat >= 0 && sp.feat < static_cast<int>(feat_gain_.size()) && std::isfinite(sp.gain)) {
            feat_gain_[sp.feat] += sp.gain;
            feat_frequency_[sp.feat]++;
            // Cover = sum of left+right hessian (total node hessian before split)
            double node_h = 0.0;
            for (size_t c = 0; c < nd.H.size(); ++c)
                node_h += nd.H[c];
            feat_cover_[sp.feat] += node_h;
        }

        if (cfg_.cache_histograms && cfg_.use_sibling_subtract && !node_uses_goss_(nd) && !node_uses_goss_(ln) &&
            !node_uses_goss_(rn)) {
            const auto features = nd.hist_features.empty() ? select_features_() : nd.hist_features;
            HistogramProvider prov(*this, index_pool_);

            auto parent_hist = prov.build_histogram(nd, features);
            Node& smaller = (ln.C <= rn.C) ? ln : rn;
            Node& larger = (ln.C <= rn.C) ? rn : ln;
            auto small_hist = prov.build_histogram(smaller, features);

            if (cfg_.cache_histograms) {
                larger.histogram = acquire_histogram_();
                *larger.histogram = *parent_hist;
                larger.histogram->subtract(*small_hist);
                larger.hist_features = features;
                larger.hist_valid = true;
                larger.hist_goss_weighted = false;
                smaller.histogram = std::move(small_hist);
                smaller.hist_features = features;
                smaller.hist_valid = true;
                smaller.hist_goss_weighted = false;
            }
            nd.histogram.reset();
            nd.hist_valid = false;
            if (nd.depth == 0)
                tree_histogram_.reset();
        }

        nodes_.push_back(std::move(ln));
        register_pos_(nodes_.back());
        nodes_.push_back(std::move(rn));
        register_pos_(nodes_.back());
    }

    inline double parent_leaf_objective_(const Node& nd) const {
        const auto [G_vec, H_vec] = node_totals_for_leaf_(nd);
        double G = 0.0, H = 0.0;
        for (size_t i = 0; i < G_vec.size(); ++i) {
            G += G_vec[i];
            H += H_vec[i];
        }
        return leaf_objective_optimal_(G, H);
    }

    inline bool accept_split_(const Node& nd, const foretree::splitx::Candidate& sp) const {
        if (!cfg_.on_tree.enabled)
            return true;

        double g = sp.gain;
        if (cfg_.on_tree.ccp_alpha > 0.0)
            g -= cfg_.on_tree.ccp_alpha;

        const double min_abs = std::max(cfg_.on_tree.min_gain, cfg_.on_tree.min_impurity_decrease);
        if (min_abs > 0.0 && g < (min_abs - cfg_.on_tree.eps))
            return false;

        if (cfg_.on_tree.min_gain_rel > 0.0) {
            const double base = std::abs(parent_leaf_objective_(nd));
            const double rel_thresh = cfg_.on_tree.min_gain_rel * base;
            if (g < (rel_thresh - cfg_.on_tree.eps))
                return false;
        }
        return (g > cfg_.on_tree.eps);
    }

    inline void finalize_leaf_(Node& n) {
        const auto [G_vec, H_vec] = node_totals_for_leaf_(n);
        double G = 0.0, H = 0.0;
        for (size_t i = 0; i < G_vec.size(); ++i) {
            G += G_vec[i];
            H += H_vec[i];
        }
        if (should_use_neural_leaf_(n))
            create_neural_leaf_(n, G, H);
        else
            n.leaf_values = leaf_values_(G, H, n.min_constraint, n.max_constraint);
        n.histogram.reset();
        n.hist_features.clear();
        n.hist_valid = false;
    }

    // ------------------------------- Growth -----------------------------------
    void grow_leaf_() {
        GrowthPolicy::leaf_wise(
            nodes_[0].id, cfg_.max_leaves, [&](int id, auto& split) { return eval_node_split_(*by_id_(id), split); },
            [&](int id, const auto& split) { return accept_split_(*by_id_(id), split); },
            [&](int id, const auto& split) { apply_split_(*by_id_(id), split); },
            [&](int id) { finalize_leaf_(*by_id_(id)); },
            [&](int id) {
                const Node* n = by_id_(id);
                return std::pair{n->left, n->right};
            },
            [&](int id, const auto& split) { return priority_(split.gain, *by_id_(id)); });
    }

    void grow_level_() {
        GrowthPolicy::level_wise(
            nodes_[0].id, cfg_.max_depth, [&](int id, auto& split) { return eval_node_split_(*by_id_(id), split); },
            [&](int id, const auto& split) { return accept_split_(*by_id_(id), split); },
            [&](int id, const auto& split) { apply_split_(*by_id_(id), split); },
            [&](int id) { finalize_leaf_(*by_id_(id)); },
            [&](int id) {
                const Node* n = by_id_(id);
                return std::pair{n->left, n->right};
            });
    }

    void grow_oblivious_() {
        GrowthPolicy::oblivious(
            nodes_[0].id, cfg_.max_depth, [&](int id, auto& split) { return eval_node_split_(*by_id_(id), split); },
            [&](int id, const auto& split) { return accept_split_(*by_id_(id), split); },
            [&](int id, const auto& split) { apply_split_(*by_id_(id), split); },
            [&](int id) { finalize_leaf_(*by_id_(id)); },
            [&](int id) {
                const Node* n = by_id_(id);
                return std::pair{n->left, n->right};
            });
    }

    // ------------------------- Neural leaf selection --------------------------
    bool should_use_neural_leaf_(const Node& n) const {
        if (!cfg_.neural_leaf.enabled)
            return false;
        if (n.C < cfg_.neural_leaf.min_samples)
            return false;
        if (n.depth < cfg_.neural_leaf.max_depth_start)
            return false;
        if (!Xraw_eval_)
            return false;

        const double r_abs = compute_residual_complexity_(n);
        const double r2 = r_abs * r_abs;
        return (r2 > cfg_.neural_leaf.complexity_threshold);
    }

    double compute_residual_complexity_(const Node& n) const {
        std::vector<double> residuals;
        residuals.reserve(n.C);
        for (int i = n.lo; i < n.hi; ++i) {
            const int r = index_pool_[i];
            const double residual = -(*g_)[r] / ((*h_)[r] + cfg_.lambda_);
            residuals.push_back(residual);
        }
        return compute_feature_correlation_strength_(residuals, n);
    }

    void create_neural_leaf_(Node& n, double GG, double HH) {
        n.leaf_values = leaf_values_(GG, HH, n.min_constraint, n.max_constraint);

        if (!cfg_.neural_leaf.enabled)
            return;
        if (!Xraw_eval_)
            return;
        if (n.C < cfg_.neural_leaf.min_samples)
            return;

        // Build GOSS weight map using helper
        auto goss_weights = build_goss_weight_map_(n);

        // Gather clean rows and weights
        std::vector<int> rows;
        std::vector<double> weights;
        rows.reserve(n.C);
        weights.reserve(n.C);

        for (int i = n.lo; i < n.hi; ++i) {
            const int r = index_pool_[i];

            // Skip if GOSS active and sample not selected
            if (n.uses_goss && goss_weights.find(r) == goss_weights.end()) {
                continue;
            }

            // Check for missing values
            bool ok = true;
            for (int j = 0; j < P_; ++j) {
                const double v = Xraw_eval_[(size_t)r * (size_t)P_ + (size_t)j];
                const bool miss =
                    Xmiss_eval_ ? (Xmiss_eval_[(size_t)r * (size_t)P_ + (size_t)j] != 0) : !std::isfinite(v);
                if (miss) {
                    ok = false;
                    break;
                }
            }

            if (ok) {
                rows.push_back(r);
                weights.push_back(n.uses_goss ? goss_weights[r] : 1.0);
            }
        }

        if ((int)rows.size() < cfg_.neural_leaf.min_samples) {
            n.neural_leaf.reset();
            return;
        }

        // Materialize contiguous buffers
        const int M = (int)rows.size();
        std::vector<double> Xbuf((size_t)M * (size_t)P_);
        std::vector<double> ybuf((size_t)M);
        std::vector<double> wbuf((size_t)M);

        for (int t = 0; t < M; ++t) {
            const int r = rows[(size_t)t];
            const double* src = Xraw_eval_ + (size_t)r * (size_t)P_;
            std::copy_n(src, (size_t)P_, Xbuf.data() + (size_t)t * (size_t)P_);
            ybuf[(size_t)t] = -(*g_)[r] / ((*h_)[r] + cfg_.lambda_);
            wbuf[(size_t)t] = weights[(size_t)t];
        }

        // Train with weights
        neural_leaf_cfg_.input_dim = P_;
        auto candidate = std::make_unique<NeuralLeafPredictor>(neural_leaf_cfg_);
        candidate->fit(Xbuf.data(), M, ybuf.data(), wbuf.data());

        if (candidate->valid()) {
            n.neural_leaf = std::move(candidate);
        } else {
            n.neural_leaf.reset();
        }
    }

    double compute_feature_correlation_strength_(const std::vector<double>& residuals, const Node& n) const {
        if (!Xraw_eval_ || residuals.empty() || n.C < 3)
            return 0.0;
        double max_abs_r = 0.0;

        for (int feat = 0; feat < P_; ++feat) {
            std::vector<double> xs, ys;
            xs.reserve(n.C);
            ys.reserve(n.C);

            for (int i = n.lo, k = 0; i < n.hi; ++i, ++k) {
                const int r = index_pool_[i];
                const double val = Xraw_eval_[(size_t)r * (size_t)P_ + (size_t)feat];
                const bool miss =
                    Xmiss_eval_ ? (Xmiss_eval_[(size_t)r * (size_t)P_ + (size_t)feat] != 0) : !std::isfinite(val);
                if (miss)
                    continue;
                xs.push_back(val);
                ys.push_back(residuals[(size_t)k]);
            }

            if (xs.size() < 3)
                continue;

            auto mean = [](const std::vector<double>& v) {
                return std::accumulate(v.begin(), v.end(), 0.0) / std::max<size_t>(1, v.size());
            };
            const double mx = mean(xs);
            const double my = mean(ys);

            double Sxx = 0.0, Syy = 0.0, Sxy = 0.0;
            for (size_t j = 0; j < xs.size(); ++j) {
                const double dx = xs[j] - mx;
                const double dy = ys[j] - my;
                Sxx += dx * dx;
                Syy += dy * dy;
                Sxy += dx * dy;
            }
            if (Sxx <= 0.0 || Syy <= 0.0)
                continue;

            const double r = Sxy / std::sqrt(Sxx * Syy);
            max_abs_r = std::max(max_abs_r, std::abs(r));
        }
        return max_abs_r;
    }

    // ---------------------------- Packing / predict ---------------------------
    void pack_() {
        K_ = std::max(cfg_.num_classes - 1, 1);
        root_id_ = PackedTreeBuilder::build(nodes_, K_,
                                            PackedTreeArrays{packed_tree_.features,
                                                             packed_tree_.thresholds,
                                                             packed_tree_.split_values,
                                                             packed_tree_.split_kinds,
                                                             packed_tree_.missing_left,
                                                             packed_tree_.left_children,
                                                             packed_tree_.right_children,
                                                             packed_tree_.leaf_flags,
                                                             packed_tree_.cover,
                                                             packed_tree_.categorical_offsets,
                                                             packed_tree_.categorical_counts,
                                                             packed_tree_.categorical_bins,
                                                             packed_tree_.pair_features_a,
                                                             packed_tree_.pair_features_b,
                                                             packed_tree_.pair_thresholds_a,
                                                             packed_tree_.pair_thresholds_b,
                                                             packed_tree_.pair_quadrant_masks,
                                                             packed_tree_.oblique_offsets,
                                                             packed_tree_.oblique_counts,
                                                             packed_tree_.oblique_features,
                                                             packed_tree_.oblique_weights,
                                                             packed_tree_.oblique_thresholds,
                                                             packed_tree_.leaf_values});
        packed_tree_.root = root_id_;
        packed_tree_.outputs = K_;
        packed_ = true;
    }
};
} // namespace foretree
