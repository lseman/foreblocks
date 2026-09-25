// foreforest_pybind.cpp
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <memory>    // std::make_shared
#include <stdexcept> // std::invalid_argument
#include <string>
#include <vector>

#include "foretree/core/histogram_primitives.hpp"
#include "foretree/ensemble/forest.hpp"
#include "foretree/ensemble/isolation_forest.hpp"
#include "foretree/split/split_engine.hpp"
#include "foretree/split/split_finder.hpp"
#include "foretree/tree/tree_types.hpp"
#include "foretree/tree/packed_tree.hpp"

namespace nb = nanobind;
using namespace nanobind::literals;

using foretree::ForeForest;
using foretree::ForeForestConfig;
using foretree::HistogramConfig;
using foretree::IsolationForest;
using foretree::IsolationForestConfig;
using foretree::InteractionSeededConfig;
using foretree::ObliqueMode;
using foretree::PairInteractionConfig;
using foretree::TreeConfig;

using CDoubleArray = nb::ndarray<nb::numpy, double, nb::c_contig>;
using CByteArray = nb::ndarray<nb::numpy, uint8_t, nb::c_contig>;

// Hand a std::vector to numpy without copying it again: the vector is moved to
// the heap and the capsule owns it, so the array stays valid for its lifetime.
// (Holding the owner in a local shared_ptr with a no-op capsule deleter frees
// the storage on return and leaves numpy reading freed memory.)
template <typename T>
static nb::ndarray<nb::numpy, T, nb::c_contig> ndarray_from_storage(std::vector<T> data,
                                                                     std::initializer_list<size_t> shape) {
    auto* owner = new std::vector<T>(std::move(data));
    nb::capsule cap(owner, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::ndarray<nb::numpy, T, nb::c_contig>(owner->data(), shape, std::move(cap));
}

// ---- Small helpers ----------------------------------------------------------
template <typename Array> static inline void ensure_2d(const Array& a, const char* name) {
    if (a.ndim() != 2)
        throw std::invalid_argument(std::string(name) + ": expected 2D array");
}
template <typename Array> static inline void ensure_1d(const Array& a, const char* name) {
    if (a.ndim() != 1)
        throw std::invalid_argument(std::string(name) + ": expected 1D array");
}

struct RawMatrixView {
    const double* Xraw = nullptr;
    const uint8_t* mask = nullptr;
};

static inline RawMatrixView parse_raw_matrix_view(const CDoubleArray& Xraw, const nb::object& miss_mask,
                                                  CByteArray& mask_buf) {
    ensure_2d(Xraw, "Xraw");
    RawMatrixView view{};
    view.Xraw = Xraw.data();

    if (miss_mask.is_none())
        return view;

    mask_buf = nb::cast<CByteArray>(miss_mask);
    ensure_2d(mask_buf, "miss_mask");
    if (mask_buf.shape(0) != Xraw.shape(0) || mask_buf.shape(1) != Xraw.shape(1))
        throw std::invalid_argument("miss_mask: shape must match Xraw");
    view.mask = mask_buf.data();
    return view;
}

NB_MODULE(foreforest, m) {
    m.doc() = "ForeForest — scalar-output bagging/GBDT with DART, nanobind bindings";

    // --------------------- HistogramConfig ---------------------
    nb::class_<HistogramConfig>(m, "HistogramConfig")
        .def(nb::init<>())
        .def_rw("method", &HistogramConfig::method)
        .def_rw("max_bins", &HistogramConfig::max_bins)
        .def_rw("use_missing_bin", &HistogramConfig::use_missing_bin)
        .def_rw("coarse_bins", &HistogramConfig::coarse_bins)
        .def_rw("density_aware", &HistogramConfig::density_aware)
        .def_rw("min_bins", &HistogramConfig::min_bins)
        .def_rw("target_bins", &HistogramConfig::target_bins)
        .def_rw("adaptive_binning", &HistogramConfig::adaptive_binning)
        .def_rw("importance_threshold", &HistogramConfig::importance_threshold)
        .def_rw("complexity_threshold", &HistogramConfig::complexity_threshold)
        .def_rw("use_feature_importance", &HistogramConfig::use_feature_importance)
        .def_rw("feature_importance_weights", &HistogramConfig::feature_importance_weights)
        .def_rw("subsample_ratio", &HistogramConfig::subsample_ratio)
        .def_rw("min_sketch_size", &HistogramConfig::min_sketch_size)
        .def_rw("use_parallel", &HistogramConfig::use_parallel)
        .def_rw("max_workers", &HistogramConfig::max_workers)
        .def_rw("rng_seed", &HistogramConfig::rng_seed)
        .def_rw("eps", &HistogramConfig::eps)
        .def("total_bins", &HistogramConfig::total_bins)
        .def("missing_bin_id", &HistogramConfig::missing_bin_id);

    // --------------------- TreeConfig + enums ------------------
    nb::class_<TreeConfig> pyTreeCfg(m, "TreeConfig");
    pyTreeCfg.def(nb::init<>())
        .def_rw("max_depth", &TreeConfig::max_depth)
        .def_rw("max_leaves", &TreeConfig::max_leaves)
        .def_rw("min_samples_split", &TreeConfig::min_samples_split)
        .def_rw("min_samples_leaf", &TreeConfig::min_samples_leaf)
        .def_rw("min_child_weight", &TreeConfig::min_child_weight)
        .def_rw("lambda_", &TreeConfig::lambda_)
        .def_rw("alpha_", &TreeConfig::alpha_)
        .def_rw("gamma_", &TreeConfig::gamma_)
        .def_rw("max_delta_step", &TreeConfig::max_delta_step)
        .def_rw("n_bins", &TreeConfig::n_bins)
        .def_rw("leaf_gain_eps", &TreeConfig::leaf_gain_eps)
        .def_rw("allow_zero_gain", &TreeConfig::allow_zero_gain)
        .def_rw("leaf_depth_penalty", &TreeConfig::leaf_depth_penalty)
        .def_rw("leaf_hess_boost", &TreeConfig::leaf_hess_boost)
        .def_rw("feature_bagging_k", &TreeConfig::feature_bagging_k)
        .def_rw("feature_bagging_with_replacement", &TreeConfig::feature_bagging_with_replacement)
        .def_rw("colsample_bytree_percent", &TreeConfig::colsample_bytree_percent)
        .def_rw("colsample_bylevel_percent", &TreeConfig::colsample_bylevel_percent)
        .def_rw("colsample_bynode_percent", &TreeConfig::colsample_bynode_percent)
        .def_rw("use_sibling_subtract", &TreeConfig::use_sibling_subtract)
        .def_rw("cuda_min_histogram_work", &TreeConfig::cuda_min_histogram_work)
        .def_rw("monotone_constraints", &TreeConfig::monotone_constraints)
        .def_rw("exact_cutover", &TreeConfig::exact_cutover)
        .def_rw("enable_categorical_splits", &TreeConfig::enable_categorical_splits)
        .def_rw("categorical_max_selected_categories", &TreeConfig::categorical_max_selected_categories)
        .def_rw("enable_oblique_splits", &TreeConfig::enable_oblique_splits)
        .def_rw("oblique_mode", &TreeConfig::oblique_mode)
        .def_rw("oblique_k_features", &TreeConfig::oblique_k_features)
        .def_rw("oblique_newton_steps", &TreeConfig::oblique_newton_steps)
        .def_rw("oblique_l1", &TreeConfig::oblique_l1)
        .def_rw("oblique_ridge", &TreeConfig::oblique_ridge)
        .def_rw("axis_vs_oblique_guard", &TreeConfig::axis_vs_oblique_guard)
        .def_rw("interaction_seeded_oblique", &TreeConfig::interaction_seeded_oblique)
        .def_rw("enable_pair_interaction_splits", &TreeConfig::enable_pair_interaction_splits)
        .def_rw("pair_interaction", &TreeConfig::pair_interaction)
        .def_rw("interaction_constraints", &TreeConfig::interaction_constraints)
        .def_rw("subsample_bytree", &TreeConfig::subsample_bytree)
        .def_rw("subsample_bylevel", &TreeConfig::subsample_bylevel)
        .def_rw("subsample_bynode", &TreeConfig::subsample_bynode)
        .def_rw("subsample_with_replacement", &TreeConfig::subsample_with_replacement)
        .def_rw("subsample_importance_scale", &TreeConfig::subsample_importance_scale)
        .def_rw("growth", &TreeConfig::growth)
        .def_rw("missing_policy", &TreeConfig::missing_policy)
        .def_rw("split_mode", &TreeConfig::split_mode);

    nb::enum_<TreeConfig::Growth>(m, "Growth")
        .value("LeafWise", TreeConfig::Growth::LeafWise)
        .value("LevelWise", TreeConfig::Growth::LevelWise)
        .value("Oblivious", TreeConfig::Growth::Oblivious);

    nb::enum_<TreeConfig::MissingPolicy>(m, "MissingPolicy")
        .value("Learn", TreeConfig::MissingPolicy::Learn)
        .value("AlwaysLeft", TreeConfig::MissingPolicy::AlwaysLeft)
        .value("AlwaysRight", TreeConfig::MissingPolicy::AlwaysRight);

    nb::enum_<TreeConfig::SplitMode>(m, "SplitMode")
        .value("Histogram", TreeConfig::SplitMode::Histogram)
        .value("Exact", TreeConfig::SplitMode::Exact)
        .value("Hybrid", TreeConfig::SplitMode::Hybrid);

    nb::enum_<ObliqueMode>(m, "ObliqueMode")
        .value("Off", ObliqueMode::Off)
        .value("Full", ObliqueMode::Full)
        .value("Auto", ObliqueMode::Auto);

    nb::class_<InteractionSeededConfig>(m, "InteractionSeededConfig")
        .def(nb::init<>())
        .def_rw("pairs", &InteractionSeededConfig::pairs)
        .def_rw("max_top_features", &InteractionSeededConfig::max_top_features)
        .def_rw("max_var_candidates", &InteractionSeededConfig::max_var_candidates)
        .def_rw("first_i_cap", &InteractionSeededConfig::first_i_cap)
        .def_rw("second_j_cap", &InteractionSeededConfig::second_j_cap)
        .def_rw("ridge", &InteractionSeededConfig::ridge)
        .def_rw("axis_guard_factor", &InteractionSeededConfig::axis_guard_factor)
        .def_rw("use_axis_guard", &InteractionSeededConfig::use_axis_guard);

    nb::class_<PairInteractionConfig>(m, "PairInteractionConfig")
        .def(nb::init<>())
        .def_rw("max_features", &PairInteractionConfig::max_features)
        .def_rw("interaction_bins", &PairInteractionConfig::interaction_bins)
        .def_rw("min_node_rows", &PairInteractionConfig::min_node_rows)
        .def_rw("complexity_penalty", &PairInteractionConfig::complexity_penalty)
        .def_rw("axis_guard_factor", &PairInteractionConfig::axis_guard_factor);

    // GOSS nested struct + member on TreeConfig
    nb::class_<TreeConfig::GOSS>(m, "GOSS")
        .def(nb::init<>())
        .def_rw("enabled", &TreeConfig::GOSS::enabled)
        .def_rw("top_rate", &TreeConfig::GOSS::top_rate)
        .def_rw("other_rate", &TreeConfig::GOSS::other_rate)
        .def_rw("scale_hessian", &TreeConfig::GOSS::scale_hessian)
        .def_rw("min_node_size", &TreeConfig::GOSS::min_node_size)
        .def_rw("use_random_rest", &TreeConfig::GOSS::use_random_rest)
        .def_rw("adaptive", &TreeConfig::GOSS::adaptive)
        .def_rw("adaptive_scale", &TreeConfig::GOSS::adaptive_scale);
    pyTreeCfg.def_rw("goss", &TreeConfig::goss);

    nb::class_<TreeConfig::NeuralLeaf>(m, "NeuralLeaf")
        .def(nb::init<>())
        .def_rw("enabled", &TreeConfig::NeuralLeaf::enabled);
    pyTreeCfg.def_rw("neural_cfg", &TreeConfig::neural_leaf);

    // --------------------- ForeForestConfig + enums -------------
    nb::class_<ForeForestConfig> pyFFCfg(m, "ForeForestConfig");
    pyFFCfg.def(nb::init<>())
        .def_rw("mode", &ForeForestConfig::mode)
        .def_rw("device", &ForeForestConfig::device)
        .def_rw("enable_cuda_backend", &ForeForestConfig::enable_cuda_backend)
        .def_rw("objective", &ForeForestConfig::objective)
        .def_rw("n_estimators", &ForeForestConfig::n_estimators)
        .def_rw("learning_rate", &ForeForestConfig::learning_rate)
        .def_rw("track_train_metric", &ForeForestConfig::track_train_metric)
        .def_rw("quantized_gradients", &ForeForestConfig::quantized_gradients)
        .def_rw("quantized_gradient_bits", &ForeForestConfig::quantized_gradient_bits)
        .def_rw("rng_seed", &ForeForestConfig::rng_seed)
        .def_rw("focal_gamma", &ForeForestConfig::focal_gamma)
        .def_rw("huber_delta", &ForeForestConfig::huber_delta)
        .def_rw("quantile_tau", &ForeForestConfig::quantile_tau)
        .def_rw("num_classes", &ForeForestConfig::num_classes)
        .def_rw("scale_pos_weight", &ForeForestConfig::scale_pos_weight)
        .def_rw("class_weight", &ForeForestConfig::class_weight)
        .def_rw("custom_class_weights", &ForeForestConfig::custom_class_weights)
        .def_rw("colsample_bytree", &ForeForestConfig::colsample_bytree)
        .def_rw("colsample_bynode", &ForeForestConfig::colsample_bynode)
        .def_rw("hist_cfg", &ForeForestConfig::hist_cfg)
        .def_rw("tree_cfg", &ForeForestConfig::tree_cfg)
        .def_rw("rf_row_subsample", &ForeForestConfig::rf_row_subsample)
        .def_rw("rf_bootstrap", &ForeForestConfig::rf_bootstrap)
        .def_rw("rf_parallel", &ForeForestConfig::rf_parallel)
        .def_rw("efb_enabled", &ForeForestConfig::efb_enabled)
        .def_rw("efb_sparse_threshold", &ForeForestConfig::efb_sparse_threshold)
        .def_rw("efb_min_nonzero", &ForeForestConfig::efb_min_nonzero)
        .def_rw("efb_max_conflict_rate", &ForeForestConfig::efb_max_conflict_rate)
        .def_rw("ordered_categorical_enabled", &ForeForestConfig::ordered_categorical_enabled)
        .def_rw("categorical_features", &ForeForestConfig::categorical_features)
        .def_rw("ordered_categorical_permutations", &ForeForestConfig::ordered_categorical_permutations)
        .def_rw("ordered_categorical_prior", &ForeForestConfig::ordered_categorical_prior)
        .def_rw("ordered_categorical_prior_weight", &ForeForestConfig::ordered_categorical_prior_weight)
        .def_rw("ordered_boosting_enabled", &ForeForestConfig::ordered_boosting_enabled)
        .def_rw("ordered_boosting_min_prefix", &ForeForestConfig::ordered_boosting_min_prefix)
        .def_rw("gbdt_row_subsample", &ForeForestConfig::gbdt_row_subsample)
        .def_rw("gbdt_use_subsample", &ForeForestConfig::gbdt_use_subsample)
        .def_rw("fw_use_subsample", &ForeForestConfig::fw_use_subsample)
        .def_rw("fw_row_subsample", &ForeForestConfig::fw_row_subsample)
        .def_rw("fw_nu", &ForeForestConfig::fw_nu)
        .def_rw("fw_line_search_points", &ForeForestConfig::fw_line_search_points)
        .def_rw("fw_alpha_max", &ForeForestConfig::fw_alpha_max)
        .def_rw("fw_alpha_tol", &ForeForestConfig::fw_alpha_tol)
        .def_rw("early_stopping_enabled", &ForeForestConfig::early_stopping_enabled)
        .def_rw("early_stopping_rounds", &ForeForestConfig::early_stopping_rounds)
        .def_rw("early_stopping_min_delta", &ForeForestConfig::early_stopping_min_delta)
        .def_rw("dart_enabled", &ForeForestConfig::dart_enabled)
        .def_rw("dart_drop_rate", &ForeForestConfig::dart_drop_rate)
        .def_rw("dart_max_drop", &ForeForestConfig::dart_max_drop)
        .def_rw("dart_normalize", &ForeForestConfig::dart_normalize);

    nb::enum_<ForeForestConfig::Mode>(m, "Mode")
        .value("Bagging", ForeForestConfig::Mode::Bagging)
        .value("GBDT", ForeForestConfig::Mode::GBDT)
        .value("FWBoost", ForeForestConfig::Mode::FWBoost);

    nb::enum_<ForeForestConfig::Device>(m, "Device")
        .value("CPU", ForeForestConfig::Device::CPU)
        .value("CUDA", ForeForestConfig::Device::CUDA)
        .value("Auto", ForeForestConfig::Device::Auto);

    nb::enum_<ForeForestConfig::Objective>(m, "Objective")
        .value("SquaredError", ForeForestConfig::Objective::SquaredError)
        .value("BinaryLogloss", ForeForestConfig::Objective::BinaryLogloss)
        .value("BinaryFocalLoss", ForeForestConfig::Objective::BinaryFocalLoss)
        .value("HuberError", ForeForestConfig::Objective::HuberError)
        .value("QuantileError", ForeForestConfig::Objective::QuantileError);

    nb::enum_<ClassWeight>(m, "ClassWeight")
        .value("None", ClassWeight::None)
        .value("Balanced", ClassWeight::Balanced)
        .value("Auto", ClassWeight::Auto);

    // --------------------- ForeForest (Python-facing) ----------
    nb::class_<ForeForest>(m, "ForeForest")
        .def(nb::init<ForeForestConfig>(), nb::arg("config"))

        // set_raw_matrix: float64 (N x P) + optional uint8 mask (N x P)
        .def(
            "set_raw_matrix",
            [](ForeForest& self, CDoubleArray Xraw, nb::object miss_mask /* None or array_t<uint8> */) {
                CByteArray mask;
                const auto raw = parse_raw_matrix_view(Xraw, miss_mask, mask);
                self.set_raw_matrix(raw.Xraw, raw.mask);
            },
            nb::arg("Xraw"), nb::arg("miss_mask") = nb::none(),
            nb::keep_alive<1, 2>(), // keep Xraw alive as long as self
            nb::keep_alive<1, 3>()  // keep miss_mask alive as long as self
            )

        .def(
            "set_raw_for_neural",
            [](ForeForest& self, CDoubleArray Xraw, nb::object miss_mask /* None or array_t<uint8> */) {
                CByteArray mask;
                const auto raw = parse_raw_matrix_view(Xraw, miss_mask, mask);
                self.set_raw_for_neural(raw.Xraw, raw.mask);
            },
            nb::arg("Xraw"), nb::arg("miss_mask") = nb::none(),
            nb::keep_alive<1, 2>(), // keep Xraw alive as long as self
            nb::keep_alive<1, 3>()  // keep miss_mask alive as long as self
            )

        // fit_complete: X float64 (N x P), y float64 (N) for scalar targets
        .def(
            "fit_complete",
            [](ForeForest& self, CDoubleArray X, CDoubleArray y, nb::object X_valid, nb::object y_valid) {
                ensure_2d(X, "X");
                ensure_1d(y, "y");
                const ssize_t N = X.shape(0);
                const ssize_t P = X.shape(1);
                if (y.shape(0) != N)
                    throw std::invalid_argument("y length must equal X.shape[0]");
                const bool has_X_valid = !X_valid.is_none();
                const bool has_y_valid = !y_valid.is_none();
                if (has_X_valid != has_y_valid)
                    throw std::invalid_argument("X_valid and y_valid must be both provided or both "
                                                "None");
                if (!has_X_valid) {
                    self.fit_complete(X.data(), static_cast<int>(N), static_cast<int>(P), y.data());
                    return;
                }

                CDoubleArray Xv = nb::cast<CDoubleArray>(X_valid);
                CDoubleArray yv = nb::cast<CDoubleArray>(y_valid);
                ensure_2d(Xv, "X_valid");
                ensure_1d(yv, "y_valid");
                const ssize_t Nv = Xv.shape(0);
                const ssize_t Pv = Xv.shape(1);
                if (Pv != P)
                    throw std::invalid_argument("X_valid.shape[1] must equal X.shape[1]");
                if (yv.shape(0) != Nv)
                    throw std::invalid_argument("y_valid length must equal X_valid.shape[0]");

                self.fit_complete(X.data(), static_cast<int>(N), static_cast<int>(P), y.data(), Xv.data(),
                                  static_cast<int>(Nv), static_cast<int>(Pv), yv.data());
            },
            nb::arg("X"), nb::arg("y"), nb::arg("X_valid") = nb::none(), nb::arg("y_valid") = nb::none(),
            "Fit a scalar-output forest. `y` and optional `y_valid` must be "
            "1-D arrays of length N.")

        // predict: X float64 (N x P) -> float64 (N) or (N, K)
        .def(
            "predict",
            [](const ForeForest& self, const nb::ndarray<nb::numpy, double, nb::c_contig>& X) {
                ensure_2d(X, "X");
                const ssize_t N = X.shape(0);
                const ssize_t P = X.shape(1);
                std::vector<double> out = self.predict(X.data(), static_cast<int>(N), static_cast<int>(P));
                int K = std::max(self.num_classes() - 1, 1);
                if (K <= 1) {
                    return ndarray_from_storage(std::move(out), {N});
                } else {
                    return ndarray_from_storage(std::move(out), {N, static_cast<ssize_t>(K)});
                }
            },
            nb::arg("X"),
            "Predict. Returns (N,) for scalar, (N,) for binary, (N, K) for "
            "multiclass.")

        .def(
            "predict_margin",
            [](const ForeForest& self, const nb::ndarray<nb::numpy, double, nb::c_contig>& X) {
                ensure_2d(X, "X");
                const ssize_t N = X.shape(0);
                const ssize_t P = X.shape(1);
                std::vector<double> out = self.predict_margin(X.data(), static_cast<int>(N), static_cast<int>(P));
                return ndarray_from_storage(std::move(out), {N});
            },
            nb::arg("X"),
            "Predict raw scalar margins, one per row. "
            "Forest prediction uses raw `X`, so neural-leaf inference is "
            "applied automatically when enabled.")

        .def(
            "predict_contrib",
            [](const ForeForest& self, const nb::ndarray<nb::numpy, double, nb::c_contig>& X) {
                ensure_2d(X, "X");
                const ssize_t N = X.shape(0);
                const ssize_t P = X.shape(1);
                std::vector<double> out = self.predict_contrib(X.data(), static_cast<int>(N), static_cast<int>(P));
                int K = std::max(self.num_classes() - 1, 1);
                if (K <= 1) {
                    return ndarray_from_storage(std::move(out), {N, P + 1});
                } else {
                    return ndarray_from_storage(std::move(out), {N, static_cast<ssize_t>(K) * (P + 1)});
                }
            },
            nb::arg("X"),
            "Predict TreeSHAP contributions. Returns (N, P+1) for "
            "scalar/binary, (N, K*(P+1)) for multiclass.")

        .def("feature_importance_gain",
             [](const ForeForest& self) {
                 std::vector<double> v = self.feature_importance_gain();
                 const auto n = static_cast<ssize_t>(v.size());
                 return ndarray_from_storage(std::move(v), {n});
             })
        .def("feature_importance_cover",
             [](const ForeForest& self) {
                 std::vector<double> v = self.feature_importance_cover();
                 const auto n = static_cast<ssize_t>(v.size());
                 return ndarray_from_storage(std::move(v), {n});
             })
        .def("feature_importance_frequency",
             [](const ForeForest& self) {
                 std::vector<int> v = self.feature_importance_frequency();
                 const auto n = static_cast<ssize_t>(v.size());
                 return ndarray_from_storage(std::move(v), {n});
             })
        .def("train_metric_history",
             [](const ForeForest& self) {
                 const std::vector<double>& v = self.train_metric_history();
                 const auto n = static_cast<ssize_t>(v.size());
                 return ndarray_from_storage(std::move(v), {n});
             })
        .def("valid_metric_history",
             [](const ForeForest& self) {
                 const std::vector<double>& v = self.valid_metric_history();
                 const auto n = static_cast<ssize_t>(v.size());
                 return ndarray_from_storage(std::move(v), {n});
             })
        .def("best_iteration", &ForeForest::best_iteration)
        .def("best_score", &ForeForest::best_score)
        .def("early_stopped", &ForeForest::early_stopped)
        .def("eval_metric_name", &ForeForest::eval_metric_name)

        .def("size", &ForeForest::size)
        .def("clear", &ForeForest::clear)
        .def("num_classes", &ForeForest::num_classes)
        .def(
            "get_packed_tree",
            [](const ForeForest& self, int idx) -> nb::tuple {
                const auto& t = self.get_packed_tree(idx);
                // Zero-copy: return views into PackedTree memory.
                // The numpy arrays reference internal PackedTree storage.
                // Users must not access these arrays after the ForeForest is destroyed.
                auto ia = [](const int* d, size_t s) {
                    auto* mut_d = const_cast<int*>(d);
                    return nb::ndarray<nb::numpy, int, nb::c_contig>(mut_d, {s}, nb::none());
                };
                auto da = [](const double* d, size_t s) {
                    auto* mut_d = const_cast<double*>(d);
                    return nb::ndarray<nb::numpy, double, nb::c_contig>(mut_d, {s}, nb::none());
                };
                auto ua = [](const uint8_t* d, size_t s) {
                    auto* mut_d = const_cast<uint8_t*>(d);
                    return nb::ndarray<nb::numpy, uint8_t, nb::c_contig>(mut_d, {s}, nb::none());
                };
                return nb::make_tuple(
                    ia(t.features.data(), t.features.size()),
                    ia(t.thresholds.data(), t.thresholds.size()),
                    da(t.split_values.data(), t.split_values.size()),
                    ua(t.split_kinds.data(), t.split_kinds.size()),
                    ua(t.missing_left.data(), t.missing_left.size()),
                    ia(t.left_children.data(), t.left_children.size()),
                    ia(t.right_children.data(), t.right_children.size()),
                    ua(t.leaf_flags.data(), t.leaf_flags.size()),
                    da(t.cover.data(), t.cover.size()),
                    ia(t.categorical_offsets.data(), t.categorical_offsets.size()),
                    ia(t.categorical_counts.data(), t.categorical_counts.size()),
                    ia(t.categorical_bins.data(), t.categorical_bins.size()),
                    ia(t.pair_features_a.data(), t.pair_features_a.size()),
                    ia(t.pair_features_b.data(), t.pair_features_b.size()),
                    ia(t.pair_thresholds_a.data(), t.pair_thresholds_a.size()),
                    ia(t.pair_thresholds_b.data(), t.pair_thresholds_b.size()),
                    ua(t.pair_quadrant_masks.data(), t.pair_quadrant_masks.size()),
                    ia(t.oblique_offsets.data(), t.oblique_offsets.size()),
                    ia(t.oblique_counts.data(), t.oblique_counts.size()),
                    ia(t.oblique_features.data(), t.oblique_features.size()),
                    da(t.oblique_weights.data(), t.oblique_weights.size()),
                    da(t.oblique_thresholds.data(), t.oblique_thresholds.size()),
                    da(t.leaf_values.data(), t.leaf_values.size())
                );
            }, nb::arg("index"),
            "Return packed tree data as a tuple of numpy arrays (zero-copy views).");

    // --------------------- IsolationForest ---------------------
    nb::class_<IsolationForestConfig>(m, "IsolationForestConfig")
        .def(nb::init<>())
        .def_rw("n_estimators", &IsolationForestConfig::n_estimators)
        .def_rw("max_samples", &IsolationForestConfig::max_samples)
        .def_rw("max_features", &IsolationForestConfig::max_features)
        .def_rw("max_depth", &IsolationForestConfig::max_depth)
        .def_rw("extension_level", &IsolationForestConfig::extension_level)
        .def_rw("bootstrap", &IsolationForestConfig::bootstrap)
        .def_rw("contamination", &IsolationForestConfig::contamination)
        .def_rw("rng_seed", &IsolationForestConfig::rng_seed);

    // X: float64 (N x P) -> float64 (N); runs without the GIL.
    auto iso_scores = [](std::vector<double> (IsolationForest::*method)(const double*, int, int) const) {
        return [method](const IsolationForest& self, const CDoubleArray& X) {
            ensure_2d(X, "X");
            const auto N = static_cast<int>(X.shape(0));
            const auto P = static_cast<int>(X.shape(1));
            std::vector<double> out;
            {
                nb::gil_scoped_release release;
                out = (self.*method)(X.data(), N, P);
            }
            return ndarray_from_storage(std::move(out), {static_cast<size_t>(N)});
        };
    };

    nb::class_<IsolationForest>(m, "IsolationForest",
                                "Isolation Forest (Liu et al., 2008); extension_level > 0 gives the "
                                "Extended Isolation Forest (Hariri et al., 2019).")
        .def(nb::init<IsolationForestConfig>(), nb::arg("config"))
        .def(
            "__init__",
            [](IsolationForest* self, int n_estimators, int max_samples, double max_features, int max_depth,
               int extension_level, bool bootstrap, double contamination, uint64_t random_state) {
                IsolationForestConfig cfg;
                cfg.n_estimators = n_estimators;
                cfg.max_samples = max_samples;
                cfg.max_features = max_features;
                cfg.max_depth = max_depth;
                cfg.extension_level = extension_level;
                cfg.bootstrap = bootstrap;
                cfg.contamination = contamination;
                cfg.rng_seed = random_state;
                new (self) IsolationForest(cfg);
            },
            nb::kw_only(), nb::arg("n_estimators") = 200, nb::arg("max_samples") = 256,
            nb::arg("max_features") = 1.0, nb::arg("max_depth") = -1, nb::arg("extension_level") = 0,
            nb::arg("bootstrap") = false, nb::arg("contamination") = -1.0, nb::arg("random_state") = 42,
            "contamination < 0 means 'auto' (threshold at anomaly score 0.5); extension_level = -1 "
            "uses hyperplanes over all features.")
        .def(
            "fit",
            [](IsolationForest& self, const CDoubleArray& X) -> IsolationForest& {
                ensure_2d(X, "X");
                const auto N = static_cast<int>(X.shape(0));
                const auto P = static_cast<int>(X.shape(1));
                {
                    nb::gil_scoped_release release;
                    self.fit(X.data(), N, P);
                }
                return self;
            },
            nb::arg("X"), nb::rv_policy::reference, "Fit on float64 X (N x P); NaN marks missing values.")
        .def("score_samples", iso_scores(&IsolationForest::score_samples), nb::arg("X"),
             "Negated anomaly score: higher is more normal (scikit-learn convention).")
        .def("anomaly_score", iso_scores(&IsolationForest::anomaly_score), nb::arg("X"),
             "Anomaly score s(x) = 2^(-E[h(x)] / c(psi)) in (0, 1]; higher is more anomalous.")
        .def("decision_function", iso_scores(&IsolationForest::decision_function), nb::arg("X"),
             "score_samples(X) - offset: negative for outliers.")
        .def("mean_path_length", iso_scores(&IsolationForest::mean_path_length), nb::arg("X"),
             "Average isolation depth E[h(x)] over the trees.")
        .def(
            "predict",
            [](const IsolationForest& self, const CDoubleArray& X) {
                ensure_2d(X, "X");
                const auto N = static_cast<int>(X.shape(0));
                const auto P = static_cast<int>(X.shape(1));
                std::vector<int> out;
                {
                    nb::gil_scoped_release release;
                    out = self.predict(X.data(), N, P);
                }
                return ndarray_from_storage(std::move(out), {static_cast<size_t>(N)});
            },
            nb::arg("X"), "+1 for inliers, -1 for outliers.")
        .def_prop_ro("offset", &IsolationForest::offset)
        .def_prop_ro("n_trees", &IsolationForest::n_trees)
        .def_prop_ro("max_samples", &IsolationForest::max_samples)
        .def_prop_ro("max_depth", &IsolationForest::max_depth)
        .def_prop_ro("fitted", &IsolationForest::fitted)
        .def_static("average_path_length", &IsolationForest::average_path_length, nb::arg("n"));
}
