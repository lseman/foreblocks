// foreforest_pybind.cpp
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <cstring>   // std::memcpy
#include <stdexcept> // std::invalid_argument
#include <string>
#include <vector>

#include "foretree/core/histogram_primitives.hpp"
#include "foretree/ensemble/forest.hpp"
#include "foretree/split/split_engine.hpp"
#include "foretree/split/split_finder.hpp"
#include "foretree/tree/tree_types.hpp"
#include "foretree/tree/packed_tree.hpp"

namespace nb = nanobind;
using namespace nanobind::literals;

using foretree::ForeForest;
using foretree::ForeForestConfig;
using foretree::HistogramConfig;
using foretree::InteractionSeededConfig;
using foretree::ObliqueMode;
using foretree::PairInteractionConfig;
using foretree::TreeConfig;

using CDoubleArray = nb::ndarray<nb::numpy, double, nb::c_contig>;
using CByteArray = nb::ndarray<nb::numpy, uint8_t, nb::c_contig>;

template <typename T> static nb::ndarray<nb::numpy, T, nb::c_contig> make_array(std::initializer_list<size_t> shape) {
    size_t size = 1;
    for (size_t extent : shape)
        size *= extent;
    T* data = new T[size];
    nb::capsule owner(data, [](void* pointer) noexcept { delete[] static_cast<T*>(pointer); });
    return nb::ndarray<nb::numpy, T, nb::c_contig>(data, shape, owner);
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
                    auto arr = make_array<double>({N});
                    if (!out.empty()) {
                        std::memcpy(arr.data(), out.data(), sizeof(double) * static_cast<size_t>(N));
                    }
                    return arr;
                } else {
                    auto arr = make_array<double>({N, static_cast<ssize_t>(K)});
                    if (!out.empty()) {
                        std::memcpy(arr.data(), out.data(), sizeof(double) * out.size());
                    }
                    return arr;
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
                auto arr = make_array<double>({N});
                if (!out.empty()) {
                    std::memcpy(arr.data(), out.data(), sizeof(double) * static_cast<size_t>(N));
                }
                return arr;
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
                    auto arr = make_array<double>({N, P + 1});
                    if (!out.empty()) {
                        std::memcpy(arr.data(), out.data(), sizeof(double) * out.size());
                    }
                    return arr;
                } else {
                    auto arr = make_array<double>({N, static_cast<ssize_t>(K) * (P + 1)});
                    if (!out.empty()) {
                        std::memcpy(arr.data(), out.data(), sizeof(double) * out.size());
                    }
                    return arr;
                }
            },
            nb::arg("X"),
            "Predict TreeSHAP contributions. Returns (N, P+1) for "
            "scalar/binary, (N, K*(P+1)) for multiclass.")

        .def("feature_importance_gain",
             [](const ForeForest& self) {
                 std::vector<double> v = self.feature_importance_gain();
                 auto arr = make_array<double>({static_cast<ssize_t>(v.size())});
                 if (!v.empty())
                     std::memcpy(arr.data(), v.data(), sizeof(double) * v.size());
                 return arr;
             })
        .def("feature_importance_cover",
             [](const ForeForest& self) {
                 std::vector<double> v = self.feature_importance_cover();
                 auto arr = make_array<double>({static_cast<ssize_t>(v.size())});
                 if (!v.empty())
                     std::memcpy(arr.data(), v.data(), sizeof(double) * v.size());
                 return arr;
             })
        .def("feature_importance_frequency",
             [](const ForeForest& self) {
                 std::vector<int> v = self.feature_importance_frequency();
                 auto arr = make_array<int>({static_cast<ssize_t>(v.size())});
                 if (!v.empty())
                     std::memcpy(arr.data(), v.data(), sizeof(int) * v.size());
                 return arr;
             })
        .def("train_metric_history",
             [](const ForeForest& self) {
                 const std::vector<double>& v = self.train_metric_history();
                 auto arr = make_array<double>({static_cast<ssize_t>(v.size())});
                 if (!v.empty())
                     std::memcpy(arr.data(), v.data(), sizeof(double) * v.size());
                 return arr;
             })
        .def("valid_metric_history",
             [](const ForeForest& self) {
                 const std::vector<double>& v = self.valid_metric_history();
                 auto arr = make_array<double>({static_cast<ssize_t>(v.size())});
                 if (!v.empty())
                     std::memcpy(arr.data(), v.data(), sizeof(double) * v.size());
                 return arr;
             })
        .def("best_iteration", &ForeForest::best_iteration)
        .def("best_score", &ForeForest::best_score)
        .def("early_stopped", &ForeForest::early_stopped)
        .def("eval_metric_name", &ForeForest::eval_metric_name)

        .def("size", &ForeForest::size)
        .def("clear", &ForeForest::clear)
        .def("num_classes", &ForeForest::num_classes)
        .def(
            "get_packed_tree", [](const ForeForest& self, int idx) -> nb::tuple {
                const auto& t = self.get_packed_tree(idx);
                auto ia = [](const int* d, size_t s) {
                    auto* b = new int[s];
                    std::memcpy(b, d, s * sizeof(int));
                    auto c = nb::capsule(b, [](void* p) noexcept { delete[] static_cast<int*>(p); });
                    return nb::ndarray<nb::numpy, int, nb::c_contig>(b, {s}, std::move(c));
                };
                auto da = [](const double* d, size_t s) {
                    auto* b = new double[s];
                    std::memcpy(b, d, s * sizeof(double));
                    auto c = nb::capsule(b, [](void* p) noexcept { delete[] static_cast<double*>(p); });
                    return nb::ndarray<nb::numpy, double, nb::c_contig>(b, {s}, std::move(c));
                };
                auto ua = [](const uint8_t* d, size_t s) {
                    auto* b = new uint8_t[s];
                    std::memcpy(b, d, s * sizeof(uint8_t));
                    auto c = nb::capsule(b, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
                    return nb::ndarray<nb::numpy, uint8_t, nb::c_contig>(b, {s}, std::move(c));
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
            "Return packed tree data as a tuple of numpy arrays.");
}
