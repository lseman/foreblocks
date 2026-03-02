// foreforest_pybind.cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>   // std::memcpy
#include <stdexcept> // std::invalid_argument
#include <string>
#include <vector>

#include "../include/foretree/ensemble.hpp"

namespace py = pybind11;

using foretree::ForeForest;
using foretree::ForeForestConfig;
using foretree::HistogramConfig;
using foretree::InteractionSeededConfig;
using foretree::ObliqueMode;
using foretree::TreeConfig;

using CDoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using CByteArray   = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

// ---- Small helpers ----------------------------------------------------------
// Accept base py::array to avoid template-flag mismatches (c_style/forcecast).
static inline void ensure_2d(const py::array &a, const char *name) {
    if (a.ndim() != 2) throw std::invalid_argument(std::string(name) + ": expected 2D array");
}
static inline void ensure_1d(const py::array &a, const char *name) {
    if (a.ndim() != 1) throw std::invalid_argument(std::string(name) + ": expected 1D array");
}

struct RawMatrixView {
    const double  *Xraw = nullptr;
    const uint8_t *mask = nullptr;
};

static inline RawMatrixView parse_raw_matrix_view(const CDoubleArray &Xraw, const py::object &miss_mask,
                                                  CByteArray &mask_buf) {
    ensure_2d(Xraw, "Xraw");
    RawMatrixView view{};
    view.Xraw = Xraw.data();

    if (miss_mask.is_none()) return view;

    mask_buf = miss_mask.cast<CByteArray>();
    ensure_2d(mask_buf, "miss_mask");
    if (mask_buf.shape(0) != Xraw.shape(0) || mask_buf.shape(1) != Xraw.shape(1))
        throw std::invalid_argument("miss_mask: shape must match Xraw");
    view.mask = mask_buf.data();
    return view;
}

PYBIND11_MODULE(foreforest, m) {
    m.doc() = "ForeForest — scalar-output bagging/GBDT with DART, pybind11 bindings";

    // --------------------- HistogramConfig ---------------------
    py::class_<HistogramConfig>(m, "HistogramConfig")
        .def(py::init<>())
        .def_readwrite("method", &HistogramConfig::method)
        .def_readwrite("max_bins", &HistogramConfig::max_bins)
        .def_readwrite("use_missing_bin", &HistogramConfig::use_missing_bin)
        .def_readwrite("coarse_bins", &HistogramConfig::coarse_bins)
        .def_readwrite("density_aware", &HistogramConfig::density_aware)
        .def_readwrite("min_bins", &HistogramConfig::min_bins)
        .def_readwrite("target_bins", &HistogramConfig::target_bins)
        .def_readwrite("adaptive_binning", &HistogramConfig::adaptive_binning)
        .def_readwrite("importance_threshold", &HistogramConfig::importance_threshold)
        .def_readwrite("complexity_threshold", &HistogramConfig::complexity_threshold)
        .def_readwrite("use_feature_importance", &HistogramConfig::use_feature_importance)
        .def_readwrite("feature_importance_weights", &HistogramConfig::feature_importance_weights)
        .def_readwrite("subsample_ratio", &HistogramConfig::subsample_ratio)
        .def_readwrite("min_sketch_size", &HistogramConfig::min_sketch_size)
        .def_readwrite("use_parallel", &HistogramConfig::use_parallel)
        .def_readwrite("max_workers", &HistogramConfig::max_workers)
        .def_readwrite("rng_seed", &HistogramConfig::rng_seed)
        .def_readwrite("eps", &HistogramConfig::eps)
        .def("total_bins", &HistogramConfig::total_bins)
        .def("missing_bin_id", &HistogramConfig::missing_bin_id);

    // --------------------- TreeConfig + enums ------------------
    py::class_<TreeConfig> pyTreeCfg(m, "TreeConfig");
    pyTreeCfg.def(py::init<>())
        .def_readwrite("max_depth", &TreeConfig::max_depth)
        .def_readwrite("max_leaves", &TreeConfig::max_leaves)
        .def_readwrite("min_samples_split", &TreeConfig::min_samples_split)
        .def_readwrite("min_samples_leaf", &TreeConfig::min_samples_leaf)
        .def_readwrite("min_child_weight", &TreeConfig::min_child_weight)
        .def_readwrite("lambda_", &TreeConfig::lambda_)
        .def_readwrite("alpha_", &TreeConfig::alpha_)
        .def_readwrite("gamma_", &TreeConfig::gamma_)
        .def_readwrite("max_delta_step", &TreeConfig::max_delta_step)
        .def_readwrite("n_bins", &TreeConfig::n_bins)
        .def_readwrite("leaf_gain_eps", &TreeConfig::leaf_gain_eps)
        .def_readwrite("allow_zero_gain", &TreeConfig::allow_zero_gain)
        .def_readwrite("leaf_depth_penalty", &TreeConfig::leaf_depth_penalty)
        .def_readwrite("leaf_hess_boost", &TreeConfig::leaf_hess_boost)
        .def_readwrite("feature_bagging_k", &TreeConfig::feature_bagging_k)
        .def_readwrite("feature_bagging_with_replacement", &TreeConfig::feature_bagging_with_replacement)
        .def_readwrite("colsample_bytree_percent", &TreeConfig::colsample_bytree_percent)
        .def_readwrite("colsample_bylevel_percent", &TreeConfig::colsample_bylevel_percent)
        .def_readwrite("colsample_bynode_percent", &TreeConfig::colsample_bynode_percent)
        .def_readwrite("use_sibling_subtract", &TreeConfig::use_sibling_subtract)
        .def_readwrite("monotone_constraints", &TreeConfig::monotone_constraints)
        .def_readwrite("exact_cutover", &TreeConfig::exact_cutover)
        .def_readwrite("enable_kway_splits", &TreeConfig::enable_kway_splits)
        .def_readwrite("kway_max_groups", &TreeConfig::kway_max_groups)
        .def_readwrite("enable_oblique_splits", &TreeConfig::enable_oblique_splits)
        .def_readwrite("oblique_mode", &TreeConfig::oblique_mode)
        .def_readwrite("oblique_k_features", &TreeConfig::oblique_k_features)
        .def_readwrite("oblique_newton_steps", &TreeConfig::oblique_newton_steps)
        .def_readwrite("oblique_l1", &TreeConfig::oblique_l1)
        .def_readwrite("oblique_ridge", &TreeConfig::oblique_ridge)
        .def_readwrite("axis_vs_oblique_guard", &TreeConfig::axis_vs_oblique_guard)
        .def_readwrite("interaction_seeded_oblique", &TreeConfig::interaction_seeded_oblique)
        .def_readwrite("subsample_bytree", &TreeConfig::subsample_bytree)
        .def_readwrite("subsample_bylevel", &TreeConfig::subsample_bylevel)
        .def_readwrite("subsample_bynode", &TreeConfig::subsample_bynode)
        .def_readwrite("subsample_with_replacement", &TreeConfig::subsample_with_replacement)
        .def_readwrite("subsample_importance_scale", &TreeConfig::subsample_importance_scale)
        .def_readwrite("growth", &TreeConfig::growth)
        .def_readwrite("missing_policy", &TreeConfig::missing_policy)
        .def_readwrite("split_mode", &TreeConfig::split_mode);

    py::enum_<TreeConfig::Growth>(m, "Growth")
        .value("LeafWise", TreeConfig::Growth::LeafWise)
        .value("LevelWise", TreeConfig::Growth::LevelWise)
        .value("Oblivious", TreeConfig::Growth::Oblivious);

    py::enum_<TreeConfig::MissingPolicy>(m, "MissingPolicy")
        .value("Learn", TreeConfig::MissingPolicy::Learn)
        .value("AlwaysLeft", TreeConfig::MissingPolicy::AlwaysLeft)
        .value("AlwaysRight", TreeConfig::MissingPolicy::AlwaysRight);

    py::enum_<TreeConfig::SplitMode>(m, "SplitMode")
        .value("Histogram", TreeConfig::SplitMode::Histogram)
        .value("Exact", TreeConfig::SplitMode::Exact)
        .value("Hybrid", TreeConfig::SplitMode::Hybrid);

    py::enum_<ObliqueMode>(m, "ObliqueMode")
        .value("Off", ObliqueMode::Off)
        .value("Full", ObliqueMode::Full)
        .value("InteractionSeeded", ObliqueMode::InteractionSeeded)
        .value("Auto", ObliqueMode::Auto);

    py::class_<InteractionSeededConfig>(m, "InteractionSeededConfig")
        .def(py::init<>())
        .def_readwrite("pairs", &InteractionSeededConfig::pairs)
        .def_readwrite("max_top_features", &InteractionSeededConfig::max_top_features)
        .def_readwrite("max_var_candidates", &InteractionSeededConfig::max_var_candidates)
        .def_readwrite("first_i_cap", &InteractionSeededConfig::first_i_cap)
        .def_readwrite("second_j_cap", &InteractionSeededConfig::second_j_cap)
        .def_readwrite("ridge", &InteractionSeededConfig::ridge)
        .def_readwrite("axis_guard_factor", &InteractionSeededConfig::axis_guard_factor)
        .def_readwrite("use_axis_guard", &InteractionSeededConfig::use_axis_guard);

    // GOSS nested struct + member on TreeConfig
    py::class_<TreeConfig::GOSS>(m, "GOSS")
        .def(py::init<>())
        .def_readwrite("enabled", &TreeConfig::GOSS::enabled)
        .def_readwrite("top_rate", &TreeConfig::GOSS::top_rate)
        .def_readwrite("other_rate", &TreeConfig::GOSS::other_rate)
        .def_readwrite("scale_hessian", &TreeConfig::GOSS::scale_hessian)
        .def_readwrite("min_node_size", &TreeConfig::GOSS::min_node_size)
        .def_readwrite("use_random_rest", &TreeConfig::GOSS::use_random_rest)
        .def_readwrite("adaptive", &TreeConfig::GOSS::adaptive)
        .def_readwrite("adaptive_scale", &TreeConfig::GOSS::adaptive_scale);
    pyTreeCfg.def_readwrite("goss", &TreeConfig::goss);

    py::class_<TreeConfig::NeuralLeaf>(m, "NeuralLeaf")
        .def(py::init<>())
        .def_readwrite("enabled", &TreeConfig::NeuralLeaf::enabled);
    pyTreeCfg.def_readwrite("neural_cfg", &TreeConfig::neural_leaf);

    // --------------------- ForeForestConfig + enums -------------
    py::class_<ForeForestConfig> pyFFCfg(m, "ForeForestConfig");
    pyFFCfg.def(py::init<>())
        .def_readwrite("mode", &ForeForestConfig::mode)
        .def_readwrite("objective", &ForeForestConfig::objective)
        .def_readwrite("n_estimators", &ForeForestConfig::n_estimators)
        .def_readwrite("learning_rate", &ForeForestConfig::learning_rate)
        .def_readwrite("rng_seed", &ForeForestConfig::rng_seed)
        .def_readwrite("focal_gamma", &ForeForestConfig::focal_gamma)
        .def_readwrite("huber_delta", &ForeForestConfig::huber_delta)
        .def_readwrite("colsample_bytree", &ForeForestConfig::colsample_bytree)
        .def_readwrite("colsample_bynode", &ForeForestConfig::colsample_bynode)
        // .def_readwrite("use_raw_matrix_for_exact", &ForeForestConfig::use_raw_matrix_for_exact)
        .def_readwrite("hist_cfg", &ForeForestConfig::hist_cfg)
        .def_readwrite("tree_cfg", &ForeForestConfig::tree_cfg)
        .def_readwrite("rf_row_subsample", &ForeForestConfig::rf_row_subsample)
        .def_readwrite("rf_bootstrap", &ForeForestConfig::rf_bootstrap)
        .def_readwrite("rf_parallel", &ForeForestConfig::rf_parallel)
        .def_readwrite("efb_enabled", &ForeForestConfig::efb_enabled)
        .def_readwrite("efb_sparse_threshold", &ForeForestConfig::efb_sparse_threshold)
        .def_readwrite("efb_min_nonzero", &ForeForestConfig::efb_min_nonzero)
        .def_readwrite("efb_max_conflict_rate", &ForeForestConfig::efb_max_conflict_rate)
        .def_readwrite("gbdt_row_subsample", &ForeForestConfig::gbdt_row_subsample)
        .def_readwrite("gbdt_use_subsample", &ForeForestConfig::gbdt_use_subsample)
        .def_readwrite("fw_use_subsample", &ForeForestConfig::fw_use_subsample)
        .def_readwrite("fw_row_subsample", &ForeForestConfig::fw_row_subsample)
        .def_readwrite("fw_nu", &ForeForestConfig::fw_nu)
        .def_readwrite("fw_line_search_points", &ForeForestConfig::fw_line_search_points)
        .def_readwrite("fw_alpha_max", &ForeForestConfig::fw_alpha_max)
        .def_readwrite("fw_alpha_tol", &ForeForestConfig::fw_alpha_tol)
        .def_readwrite("early_stopping_enabled", &ForeForestConfig::early_stopping_enabled)
        .def_readwrite("early_stopping_rounds", &ForeForestConfig::early_stopping_rounds)
        .def_readwrite("early_stopping_min_delta", &ForeForestConfig::early_stopping_min_delta)
        .def_readwrite("dart_enabled", &ForeForestConfig::dart_enabled)
        .def_readwrite("dart_drop_rate", &ForeForestConfig::dart_drop_rate)
        .def_readwrite("dart_max_drop", &ForeForestConfig::dart_max_drop)
        .def_readwrite("dart_normalize", &ForeForestConfig::dart_normalize);

    py::enum_<ForeForestConfig::Mode>(m, "Mode")
        .value("Bagging", ForeForestConfig::Mode::Bagging)
        .value("GBDT", ForeForestConfig::Mode::GBDT)
        .value("FWBoost", ForeForestConfig::Mode::FWBoost);

    py::enum_<ForeForestConfig::Objective>(m, "Objective")
        .value("SquaredError", ForeForestConfig::Objective::SquaredError)
        .value("BinaryLogloss", ForeForestConfig::Objective::BinaryLogloss)
        .value("BinaryFocalLoss", ForeForestConfig::Objective::BinaryFocalLoss)
        .value("HuberError", ForeForestConfig::Objective::HuberError);

    // --------------------- ForeForest (Python-facing) ----------
    py::class_<ForeForest>(m, "ForeForest")
        .def(py::init<ForeForestConfig>(), py::arg("config"))

        // set_raw_matrix: float64 (N x P) + optional uint8 mask (N x P)
        .def(
            "set_raw_matrix",
            [](ForeForest &self, CDoubleArray Xraw, py::object miss_mask /* None or array_t<uint8> */) {
                CByteArray mask;
                const auto raw = parse_raw_matrix_view(Xraw, miss_mask, mask);
                // ForeForest stores/uses the raw pointer & its own
                // strides/views internally.
                self.set_raw_matrix(raw.Xraw, raw.mask);
            },
            py::arg("Xraw"), py::arg("miss_mask") = py::none(),
            py::keep_alive<1, 2>(), // keep Xraw alive as long as self
            py::keep_alive<1, 3>()  // keep miss_mask alive as long as self
            )

        .def(
            "set_raw_for_neural",
            [](ForeForest &self, CDoubleArray Xraw, py::object miss_mask /* None or array_t<uint8> */) {
                CByteArray mask;
                const auto raw = parse_raw_matrix_view(Xraw, miss_mask, mask);
                // Store pointers for neural leaves
                self.set_raw_for_neural(raw.Xraw, raw.mask);
            },
            py::arg("Xraw"), py::arg("miss_mask") = py::none(),
            py::keep_alive<1, 2>(), // keep Xraw alive as long as self
            py::keep_alive<1, 3>()  // keep miss_mask alive as long as self
            )

        // fit_complete: X float64 (N x P), y float64 (N) for scalar targets
        .def(
            "fit_complete",
            [](ForeForest &self, CDoubleArray X, CDoubleArray y, py::object X_valid, py::object y_valid) {
                ensure_2d(X, "X");
                ensure_1d(y, "y");
                const py::ssize_t N = X.shape(0);
                const py::ssize_t P = X.shape(1);
                if (y.shape(0) != N) throw std::invalid_argument("y length must equal X.shape[0]");
                const bool has_X_valid = !X_valid.is_none();
                const bool has_y_valid = !y_valid.is_none();
                if (has_X_valid != has_y_valid)
                    throw std::invalid_argument("X_valid and y_valid must be both provided or both None");
                if (!has_X_valid) {
                    self.fit_complete(X.data(), static_cast<int>(N), static_cast<int>(P), y.data());
                    return;
                }

                CDoubleArray Xv = X_valid.cast<CDoubleArray>();
                CDoubleArray yv = y_valid.cast<CDoubleArray>();
                ensure_2d(Xv, "X_valid");
                ensure_1d(yv, "y_valid");
                const py::ssize_t Nv = Xv.shape(0);
                const py::ssize_t Pv = Xv.shape(1);
                if (Pv != P) throw std::invalid_argument("X_valid.shape[1] must equal X.shape[1]");
                if (yv.shape(0) != Nv) throw std::invalid_argument("y_valid length must equal X_valid.shape[0]");

                self.fit_complete(X.data(), static_cast<int>(N), static_cast<int>(P), y.data(), Xv.data(),
                                  static_cast<int>(Nv), static_cast<int>(Pv), yv.data());
            },
            py::arg("X"), py::arg("y"), py::arg("X_valid") = py::none(), py::arg("y_valid") = py::none(),
            "Fit a scalar-output forest. `y` and optional `y_valid` must be 1-D arrays of length N.")

        // predict: X float64 (N x P) -> float64 (N)
        .def(
            "predict",
            [](const ForeForest &self, py::array_t<double, py::array::c_style | py::array::forcecast> X) {
                ensure_2d(X, "X");
                const py::ssize_t   N   = X.shape(0);
                const py::ssize_t   P   = X.shape(1);
                std::vector<double> out = self.predict(X.data(), static_cast<int>(N), static_cast<int>(P));
                py::array_t<double> arr({N});
                if (!out.empty()) {
                    std::memcpy(arr.mutable_data(), out.data(), sizeof(double) * static_cast<size_t>(N));
                }
                return arr;
            },
            py::arg("X"),
            "Predict one scalar value per row. "
            "Forest prediction uses raw `X`, so neural-leaf inference is applied automatically when enabled.")

        .def(
            "predict_margin",
            [](const ForeForest &self, py::array_t<double, py::array::c_style | py::array::forcecast> X) {
                ensure_2d(X, "X");
                const py::ssize_t   N   = X.shape(0);
                const py::ssize_t   P   = X.shape(1);
                std::vector<double> out = self.predict_margin(X.data(), static_cast<int>(N), static_cast<int>(P));
                py::array_t<double> arr({N});
                if (!out.empty()) {
                    std::memcpy(arr.mutable_data(), out.data(), sizeof(double) * static_cast<size_t>(N));
                }
                return arr;
            },
            py::arg("X"),
            "Predict raw scalar margins, one per row. "
            "Forest prediction uses raw `X`, so neural-leaf inference is applied automatically when enabled.")

        .def(
            "predict_contrib",
            [](const ForeForest &self, py::array_t<double, py::array::c_style | py::array::forcecast> X) {
                ensure_2d(X, "X");
                const py::ssize_t N = X.shape(0);
                const py::ssize_t P = X.shape(1);
                std::vector<double> out = self.predict_contrib(X.data(), static_cast<int>(N), static_cast<int>(P));
                py::array_t<double> arr({N, P + 1});
                if (!out.empty()) {
                    std::memcpy(arr.mutable_data(), out.data(), sizeof(double) * out.size());
                }
                return arr;
            },
            py::arg("X"),
            "Predict TreeSHAP contributions on the raw-margin scale. The last column is the bias term.")

        .def("feature_importance_gain",
             [](const ForeForest &self) {
                 std::vector<double> v = self.feature_importance_gain();
                 py::array_t<double> arr({static_cast<py::ssize_t>(v.size())});
                 if (!v.empty()) std::memcpy(arr.mutable_data(), v.data(), sizeof(double) * v.size());
                 return arr;
             })
        .def("train_metric_history",
             [](const ForeForest &self) {
                 const std::vector<double> &v = self.train_metric_history();
                 py::array_t<double>        arr({static_cast<py::ssize_t>(v.size())});
                 if (!v.empty()) std::memcpy(arr.mutable_data(), v.data(), sizeof(double) * v.size());
                 return arr;
             })
        .def("valid_metric_history",
             [](const ForeForest &self) {
                 const std::vector<double> &v = self.valid_metric_history();
                 py::array_t<double>        arr({static_cast<py::ssize_t>(v.size())});
                 if (!v.empty()) std::memcpy(arr.mutable_data(), v.data(), sizeof(double) * v.size());
                 return arr;
             })
        .def("best_iteration", &ForeForest::best_iteration)
        .def("best_score", &ForeForest::best_score)
        .def("early_stopped", &ForeForest::early_stopped)
        .def("eval_metric_name", &ForeForest::eval_metric_name)

        .def("size", &ForeForest::size)
        .def("clear", &ForeForest::clear);
}
