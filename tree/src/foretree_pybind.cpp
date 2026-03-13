// src/foretree_pybind.cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>  // memcpy

// headers (updated paths)
#include "../include/foretree/core.hpp"
#include "../include/foretree/tree.hpp"

namespace py = pybind11;

using foretree::DataBinner;
using foretree::EdgeSet;
using foretree::FeatureBins;
using foretree::GradientHistogramSystem;
using foretree::HistogramConfig;
using foretree::InteractionSeededConfig;
using foretree::ObliqueMode;
using foretree::TreeConfig;
using foretree::UnifiedTree;

template <typename T>
static std::vector<T> arr_to_vec_1d(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& a) {
    auto buf = a.request();
    if (buf.ndim != 1) throw std::invalid_argument("Expected a 1D array");
    const T* ptr = static_cast<const T*>(buf.ptr);
    return std::vector<T>(ptr, ptr + buf.shape[0]);
}

template <typename T>
static std::vector<T> arr_to_vec_any(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& a) {
    auto buf = a.request();
    size_t n = 1;
    for (auto s : buf.shape) n *= static_cast<size_t>(s);
    const T* ptr = static_cast<const T*>(buf.ptr);
    return std::vector<T>(ptr, ptr + n);
}

PYBIND11_MODULE(foretree, m) {
    m.doc() = "Foretree: gradient-aware binning + unified tree (pybind11)";

    // Updated pybind11 binding code for DataBinner
    // Replace the existing DataBinner binding section with this:

    py::class_<EdgeSet>(m, "EdgeSet")
        .def(py::init<>())
        .def_readwrite("edges_per_feat", &EdgeSet::edges_per_feat)
        .def_readwrite("finite_bins", &EdgeSet::finite_bins)
        .def_readwrite("missing_bin_id", &EdgeSet::missing_bin_id)
        // New fields (read-only from Python for safety)
        .def_readonly("finite_bins_per_feat", &EdgeSet::finite_bins_per_feat)
        .def_readonly("missing_bin_id_per_feat",
                      &EdgeSet::missing_bin_id_per_feat);

    py::class_<DataBinner>(m, "DataBinner")
        .def(py::init<int>(), py::arg("P"))
        .def(
            "register_edges",
            [](DataBinner& db, const std::string& mode,
               const std::vector<std::vector<double>>& edges_per_feat) {
                EdgeSet es;
                es.edges_per_feat = edges_per_feat;
                // The new fields will be automatically computed in
                // register_edges()
                db.register_edges(mode, es);
            },
            py::arg("mode"), py::arg("edges_per_feat"))
        .def("register_edges",
             static_cast<void (DataBinner::*)(const std::string&, EdgeSet)>(
                 &DataBinner::register_edges),
             py::arg("mode"), py::arg("edgeset"))
        .def("set_node_override", &DataBinner::set_node_override,
             py::arg("mode"), py::arg("node_id"), py::arg("feat"),
             py::arg("edges"))
        .def(
            "prebin",
            [](const DataBinner& db, py::array_t<double> X,
               const std::string& mode, int node_id) {
                py::buffer_info buf = X.request();
                if (buf.ndim != 2) {
                    throw std::runtime_error(
                        "Input array must be 2-dimensional");
                }
                int N = static_cast<int>(buf.shape[0]);
                int P = static_cast<int>(buf.shape[1]);

                auto result = db.prebin(static_cast<const double*>(buf.ptr), N,
                                        P, mode, node_id);

                // Convert to numpy array
                auto codes_array = py::array_t<uint16_t>(
                    {N, P},                                    // shape
                    {sizeof(uint16_t) * P, sizeof(uint16_t)},  // strides
                    result.first->data()                       // data pointer
                );

                return py::make_tuple(codes_array, result.second);
            },
            py::arg("X"), py::arg("mode"), py::arg("node_id") = -1,
            "Returns tuple of (codes_array, missing_bin_id)")
        .def(
            "prebin_into",
            [](const DataBinner& db, py::array_t<double> X,
               const std::string& mode, py::array_t<uint16_t> out_codes,
               int node_id) {
                py::buffer_info X_buf = X.request();
                py::buffer_info out_buf = out_codes.request();

                if (X_buf.ndim != 2 || out_buf.ndim != 2) {
                    throw std::runtime_error("Arrays must be 2-dimensional");
                }

                int N = static_cast<int>(X_buf.shape[0]);
                int P = static_cast<int>(X_buf.shape[1]);

                return db.prebin_into(
                    static_cast<const double*>(X_buf.ptr), N, P, mode,
                    static_cast<uint16_t*>(out_buf.ptr), node_id);
            },
            py::arg("X"), py::arg("mode"), py::arg("out_codes"),
            py::arg("node_id") = -1)

        // Existing query methods (backward compatible)
        .def("finite_bins",
             static_cast<int (DataBinner::*)(const std::string&) const>(
                 &DataBinner::finite_bins),
             py::arg("mode"))
        .def("missing_bin_id",
             static_cast<int (DataBinner::*)(const std::string&) const>(
                 &DataBinner::missing_bin_id),
             py::arg("mode"))
        .def("total_bins",
             static_cast<int (DataBinner::*)(const std::string&) const>(
                 &DataBinner::total_bins),
             py::arg("mode"))

        // New per-feature query methods
        .def("finite_bins",
             static_cast<int (DataBinner::*)(const std::string&, int) const>(
                 &DataBinner::finite_bins),
             py::arg("mode"), py::arg("feat"))
        .def("missing_bin_id",
             static_cast<int (DataBinner::*)(const std::string&, int) const>(
                 &DataBinner::missing_bin_id),
             py::arg("mode"), py::arg("feat"))
        .def("total_bins",
             static_cast<int (DataBinner::*)(const std::string&, int) const>(
                 &DataBinner::total_bins),
             py::arg("mode"), py::arg("feat"))

        // New bulk per-feature query methods
        .def("finite_bins_per_feat", &DataBinner::finite_bins_per_feat,
             py::arg("mode"), py::return_value_policy::reference_internal)
        .def("missing_bin_ids_per_feat", &DataBinner::missing_bin_ids_per_feat,
             py::arg("mode"), py::return_value_policy::reference_internal)

        .def("P", &DataBinner::P);

    // Updated pybind11 binding code for GradientHistogramSystem
    // Replace the existing GradientHistogramSystem binding section with this:

    py::class_<HistogramConfig>(m, "HistogramConfig")
        .def(py::init<>())
        .def_readwrite("method", &HistogramConfig::method)
        .def_readwrite("max_bins", &HistogramConfig::max_bins)
        .def_readwrite("use_missing_bin", &HistogramConfig::use_missing_bin)
        .def_readwrite("coarse_bins", &HistogramConfig::coarse_bins)
        .def_readwrite("density_aware", &HistogramConfig::density_aware)
        // NEW: Adaptive binning fields
        .def_readwrite("min_bins", &HistogramConfig::min_bins)
        .def_readwrite("target_bins", &HistogramConfig::target_bins)
        .def_readwrite("adaptive_binning", &HistogramConfig::adaptive_binning)
        .def_readwrite("importance_threshold",
                       &HistogramConfig::importance_threshold)
        .def_readwrite("complexity_threshold",
                       &HistogramConfig::complexity_threshold)
        .def_readwrite("use_feature_importance",
                       &HistogramConfig::use_feature_importance)
        .def_readwrite("feature_importance_weights",
                       &HistogramConfig::feature_importance_weights)
        // Other fields
        .def_readwrite("subsample_ratio", &HistogramConfig::subsample_ratio)
        .def_readwrite("min_sketch_size", &HistogramConfig::min_sketch_size)
        .def_readwrite("use_parallel", &HistogramConfig::use_parallel)
        .def_readwrite("max_workers", &HistogramConfig::max_workers)
        .def_readwrite("rng_seed", &HistogramConfig::rng_seed)
        .def_readwrite("eps", &HistogramConfig::eps)
        .def("total_bins", &HistogramConfig::total_bins)
        .def("missing_bin_id", &HistogramConfig::missing_bin_id);

    // py::class_<FeatureStats>(m, "FeatureStats")
    //     .def(py::init<>())
    //     .def_readwrite("variance", &FeatureStats::variance)
    //     .def_readwrite("gradient_variance", &FeatureStats::gradient_variance)
    //     .def_readwrite("gradient_range", &FeatureStats::gradient_range)
    //     .def_readwrite("value_range", &FeatureStats::value_range)
    //     .def_readwrite("unique_count", &FeatureStats::unique_count)
    //     .def_readwrite("complexity_score", &FeatureStats::complexity_score)
    //     .def_readwrite("importance_score", &FeatureStats::importance_score)
    //     .def_readwrite("is_categorical", &FeatureStats::is_categorical)
    //     .def_readwrite("suggested_bins", &FeatureStats::suggested_bins)
    //     .def_readwrite("allocation_reason",
    //     &FeatureStats::allocation_reason);

    py::class_<FeatureBins>(m, "FeatureBins")
        .def(py::init<>())
        .def_readwrite("edges", &FeatureBins::edges)
        .def_readwrite("is_uniform", &FeatureBins::is_uniform)
        .def_readwrite("strategy", &FeatureBins::strategy)
        .def_readwrite("lo", &FeatureBins::lo)
        .def_readwrite("width", &FeatureBins::width)
        .def_readwrite("stats", &FeatureBins::stats)
        .def("n_bins", &FeatureBins::n_bins);

    py::class_<GradientHistogramSystem,
               std::shared_ptr<GradientHistogramSystem>>(
        m, "GradientHistogramSystem")
        .def(py::init<HistogramConfig>(), py::arg("config"))
        .def(
            "fit_bins",
            [](GradientHistogramSystem& ghs, py::array_t<double> X,
               py::array_t<double> g, py::array_t<double> h) {
                py::buffer_info X_buf = X.request();
                py::buffer_info g_buf = g.request();
                py::buffer_info h_buf = h.request();

                if (X_buf.ndim != 2) {
                    throw std::runtime_error("X must be 2-dimensional");
                }
                if (g_buf.ndim != 1 || h_buf.ndim != 1) {
                    throw std::runtime_error("g and h must be 1-dimensional");
                }

                int N = static_cast<int>(X_buf.shape[0]);
                int P = static_cast<int>(X_buf.shape[1]);

                if (g_buf.shape[0] != N || h_buf.shape[0] != N) {
                    throw std::runtime_error(
                        "g and h must have same length as X rows");
                }

                ghs.fit_bins(static_cast<const double*>(X_buf.ptr), N, P,
                             static_cast<const double*>(g_buf.ptr),
                             static_cast<const double*>(h_buf.ptr));
            },
            py::arg("X"), py::arg("g"), py::arg("h"))
        .def(
            "prebin_dataset",
            [](GradientHistogramSystem& ghs, py::array_t<double> X) {
                py::buffer_info buf = X.request();
                if (buf.ndim != 2) {
                    throw std::runtime_error("X must be 2-dimensional");
                }
                int N = static_cast<int>(buf.shape[0]);
                int P = static_cast<int>(buf.shape[1]);

                auto result = ghs.prebin_dataset(
                    static_cast<const double*>(buf.ptr), N, P);

                // Convert to numpy array
                auto codes_array = py::array_t<uint16_t>(
                    {N, P},                                    // shape
                    {sizeof(uint16_t) * P, sizeof(uint16_t)},  // strides
                    result.first->data()                       // data pointer
                );

                return py::make_tuple(codes_array, result.second);
            },
            py::arg("X"), "Returns tuple of (codes_array, missing_bin_id)")
        .def(
            "prebin_matrix",
            [](const GradientHistogramSystem& ghs, py::array_t<double> X) {
                py::buffer_info buf = X.request();
                if (buf.ndim != 2) {
                    throw std::runtime_error("X must be 2-dimensional");
                }
                int N = static_cast<int>(buf.shape[0]);
                int P = static_cast<int>(buf.shape[1]);

                auto result = ghs.prebin_matrix(
                    static_cast<const double*>(buf.ptr), N, P);

                // Convert to numpy array
                auto codes_array = py::array_t<uint16_t>(
                    {N, P},                                    // shape
                    {sizeof(uint16_t) * P, sizeof(uint16_t)},  // strides
                    result.first->data()                       // data pointer
                );

                return py::make_tuple(codes_array, result.second);
            },
            py::arg("X"), "Returns tuple of (codes_array, missing_bin_id)")
        .def(
            "build_histograms_fast",
            [](const GradientHistogramSystem& ghs, py::array_t<float> g,
               py::array_t<float> h, py::array_t<int> sample_indices) {
                py::buffer_info g_buf = g.request();
                py::buffer_info h_buf = h.request();

                if (g_buf.ndim != 1 || h_buf.ndim != 1) {
                    throw std::runtime_error("g and h must be 1-dimensional");
                }

                const int* indices_ptr = nullptr;
                int n_sub = 0;
                if (sample_indices.size() > 0) {
                    py::buffer_info idx_buf = sample_indices.request();
                    if (idx_buf.ndim != 1) {
                        throw std::runtime_error(
                            "sample_indices must be 1-dimensional");
                    }
                    indices_ptr = static_cast<const int*>(idx_buf.ptr);
                    n_sub = static_cast<int>(idx_buf.shape[0]);
                }

                auto result = ghs.build_histograms_fast(
                    static_cast<const float*>(g_buf.ptr),
                    static_cast<const float*>(h_buf.ptr), indices_ptr, n_sub);

                return py::make_tuple(result.first, result.second);
            },
            py::arg("g"), py::arg("h"),
            py::arg("sample_indices") = py::array_t<int>(),
            "Returns tuple of (Hg, Hh)")
        .def(
            "build_histograms_fast_with_counts",
            [](const GradientHistogramSystem& ghs, py::array_t<float> g,
               py::array_t<float> h, py::array_t<int> sample_indices) {
                py::buffer_info g_buf = g.request();
                py::buffer_info h_buf = h.request();

                if (g_buf.ndim != 1 || h_buf.ndim != 1) {
                    throw std::runtime_error("g and h must be 1-dimensional");
                }

                const int* indices_ptr = nullptr;
                int n_sub = 0;
                if (sample_indices.size() > 0) {
                    py::buffer_info idx_buf = sample_indices.request();
                    if (idx_buf.ndim != 1) {
                        throw std::runtime_error(
                            "sample_indices must be 1-dimensional");
                    }
                    indices_ptr = static_cast<const int*>(idx_buf.ptr);
                    n_sub = static_cast<int>(idx_buf.shape[0]);
                }

                auto result = ghs.build_histograms_fast_with_counts(
                    static_cast<const float*>(g_buf.ptr),
                    static_cast<const float*>(h_buf.ptr), indices_ptr, n_sub);

                return py::make_tuple(std::get<0>(result), std::get<1>(result),
                                      std::get<2>(result));
            },
            py::arg("g"), py::arg("h"),
            py::arg("sample_indices") = py::array_t<int>(),
            "Returns tuple of (Hg, Hh, C)")
        .def(
            "extract_feature_histogram",
            [](const GradientHistogramSystem& ghs,
               const std::vector<double>& Hg, const std::vector<double>& Hh,
               const std::vector<int>& C, int feature) {
                auto result = ghs.extract_feature_histogram(Hg, Hh, C, feature);
                return py::make_tuple(std::get<0>(result), std::get<1>(result),
                                      std::get<2>(result));
            },
            py::arg("Hg"), py::arg("Hh"), py::arg("C"), py::arg("feature"),
            "Returns tuple of (feat_Hg, feat_Hh, feat_C)")
        .def("get_feature_offsets",
             &GradientHistogramSystem::get_feature_offsets)
        .def("get_bin_allocation_summary",
             &GradientHistogramSystem::get_bin_allocation_summary)

        // Accessors with explicit casting for overloaded methods
        .def("P", &GradientHistogramSystem::P)
        .def("N", &GradientHistogramSystem::N)
        .def("missing_bin_id",
             static_cast<int (GradientHistogramSystem::*)() const>(
                 &GradientHistogramSystem::missing_bin_id))
        .def("finite_bins",
             static_cast<int (GradientHistogramSystem::*)() const>(
                 &GradientHistogramSystem::finite_bins))
        .def("total_bins",
             static_cast<int (GradientHistogramSystem::*)() const>(
                 &GradientHistogramSystem::total_bins))

        // Per-feature accessors
        .def("finite_bins",
             static_cast<int (GradientHistogramSystem::*)(int) const>(
                 &GradientHistogramSystem::finite_bins),
             py::arg("feature"))
        .def("total_bins",
             static_cast<int (GradientHistogramSystem::*)(int) const>(
                 &GradientHistogramSystem::total_bins),
             py::arg("feature"))
        .def("missing_bin_id",
             static_cast<int (GradientHistogramSystem::*)(int) const>(
                 &GradientHistogramSystem::missing_bin_id),
             py::arg("feature"))

        // Vector accessors
        .def("all_finite_bins", &GradientHistogramSystem::all_finite_bins)
        .def("all_total_bins", &GradientHistogramSystem::all_total_bins)

        // Feature analysis results
        .def("feature_stats", &GradientHistogramSystem::feature_stats,
             py::arg("feature"), py::return_value_policy::reference_internal)
        .def("feature_bins", &GradientHistogramSystem::feature_bins,
             py::arg("feature"), py::return_value_policy::reference_internal)

        // Internal access
        .def("binner", &GradientHistogramSystem::binner,
             py::return_value_policy::reference_internal)
        .def(
            "codes_view",
            [](const GradientHistogramSystem& ghs) {
                auto codes_ptr = ghs.codes_view();
                if (!codes_ptr) {
                    throw std::runtime_error(
                        "No codes available. Call prebin_dataset first.");
                }

                // Convert to numpy array (N x P shape)
                int N = ghs.N();
                int P = ghs.P();
                auto codes_array = py::array_t<uint16_t>(
                    {N, P},                                    // shape
                    {sizeof(uint16_t) * P, sizeof(uint16_t)},  // strides
                    codes_ptr->data()                          // data pointer
                );

                return codes_array;
            },
            "Returns the cached binned codes as numpy array (N x P)");
    // ---------------- TreeConfig ----------------

    // Enums (exposed at module level for simplicity)
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

    // Nested GOSS struct
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

    py::class_<TreeConfig>(m, "TreeConfig")
        .def(py::init<>())
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
        .def_readwrite("growth", &TreeConfig::growth)
        .def_readwrite("leaf_gain_eps", &TreeConfig::leaf_gain_eps)
        .def_readwrite("allow_zero_gain", &TreeConfig::allow_zero_gain)
        .def_readwrite("leaf_depth_penalty", &TreeConfig::leaf_depth_penalty)
        .def_readwrite("leaf_hess_boost", &TreeConfig::leaf_hess_boost)
        .def_readwrite("feature_bagging_k", &TreeConfig::feature_bagging_k)
        .def_readwrite("feature_bagging_with_replacement",
                       &TreeConfig::feature_bagging_with_replacement)
        .def_readwrite("colsample_bytree_percent",
                       &TreeConfig::colsample_bytree_percent)
        .def_readwrite("colsample_bylevel_percent",
                       &TreeConfig::colsample_bylevel_percent)
        .def_readwrite("colsample_bynode_percent",
                       &TreeConfig::colsample_bynode_percent)
        .def_readwrite("use_sibling_subtract",
                       &TreeConfig::use_sibling_subtract)
        .def_readwrite("missing_policy", &TreeConfig::missing_policy)
        .def_readwrite("monotone_constraints",
                       &TreeConfig::monotone_constraints)
        .def_readwrite("split_mode", &TreeConfig::split_mode)
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
        .def_readwrite("goss", &TreeConfig::goss);

    // ---------------- UnifiedTree (binned API) ----------------
    py::class_<UnifiedTree>(m, "UnifiedTree", py::dynamic_attr())
        .def(py::init<>())
        // Attach a GradientHistogramSystem; keep_alive so GHS outlives the tree
        .def(py::init([](const TreeConfig& cfg,
                         std::shared_ptr<GradientHistogramSystem> ghs) {
                 return new UnifiedTree(cfg, ghs.get());
             }),
             py::arg("config"), py::arg("ghs"), py::keep_alive<1, 2>())

        // --- raw matrix (for Exact/Hybrid) ---
        .def(
            "set_raw_matrix",
            [](UnifiedTree& self,
               const py::array_t<double, py::array::c_style |
                                             py::array::forcecast>& Xraw,
               py::object mask /* None or uint8 array */) {
                auto xb = Xraw.request();
                if (xb.ndim != 2)
                    throw std::invalid_argument(
                        "Xraw must be 2D float64 (N, P)");
                const double* Xptr = static_cast<const double*>(xb.ptr);

                const uint8_t* mptr = nullptr;
                py::array_t<uint8_t> mask_arr;
                if (!mask.is_none()) {
                    mask_arr = mask.cast<py::array_t<uint8_t>>();
                    auto mb = mask_arr.request();
                    if (mb.ndim != 2)
                        throw std::invalid_argument(
                            "mask must be 2D uint8 (N, P)");
                    if (mb.shape[0] != xb.shape[0] ||
                        mb.shape[1] != xb.shape[1])
                        throw std::invalid_argument(
                            "mask shape must match Xraw");
                    mptr = static_cast<const uint8_t*>(mb.ptr);
                }

                self.set_raw_matrix(Xptr, mptr);

                // Keep arrays alive by attaching references to the tree object
                // (pybind: get Python handle for 'self' and store refs)
                py::object self_obj = py::cast(&self);
                self_obj.attr("_Xraw_ref") = Xraw;
                if (!mask.is_none()) self_obj.attr("_Xmask_ref") = mask_arr;
            },
            py::arg("Xraw"), py::arg("mask") = py::none())

        // --- training / inference ---
        .def(
            "fit_binned",
            [](UnifiedTree& self,
               const py::array_t<uint16_t,
                                 py::array::c_style | py::array::forcecast>& Xb,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>& g,
               const py::array_t<double, py::array::c_style |
                                             py::array::forcecast>& h) {
                auto xb = Xb.request();
                if (xb.ndim != 2)
                    throw std::invalid_argument("Xb must be 2D (N, P)");
                const int N = static_cast<int>(xb.shape[0]);
                const int P = static_cast<int>(xb.shape[1]);

                std::vector<uint16_t> Xv = arr_to_vec_any<uint16_t>(Xb);
                std::vector<double> gv = arr_to_vec_1d<double>(g);
                std::vector<double> hv = arr_to_vec_1d<double>(h);
                if ((int)gv.size() != N || (int)hv.size() != N)
                    throw std::invalid_argument(
                        "g and h must have length N = Xb.shape[0]");

                self.fit(Xv, N, P, gv, hv);
            },
            py::arg("Xb"), py::arg("g"), py::arg("h"),
            "Fit a scalar-output tree on binned features with 1-D gradient and hessian arrays.")

        .def(
            "predict_binned",
            [](const UnifiedTree& self,
               const py::array_t<uint16_t, py::array::c_style |
                                               py::array::forcecast>& Xb,
               py::object Xraw_opt) {
                auto xb = Xb.request();
                if (xb.ndim != 2)
                    throw std::invalid_argument("Xb must be 2D (N, P)");
                const int N = static_cast<int>(xb.shape[0]);
                const int P = static_cast<int>(xb.shape[1]);

                std::vector<uint16_t> Xv = arr_to_vec_any<uint16_t>(Xb);
                std::vector<double> pred;
                if (Xraw_opt.is_none()) {
                    pred = self.predict(Xv, N, P);
                } else {
                    auto Xraw = Xraw_opt.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                    auto xr   = Xraw.request();
                    if (xr.ndim != 2)
                        throw std::invalid_argument("Xraw must be 2D (N, P)");
                    if (xr.shape[0] != xb.shape[0] || xr.shape[1] != xb.shape[1])
                        throw std::invalid_argument("Xraw shape must match Xb shape");
                    pred = self.predict(Xv, N, P, static_cast<const double*>(xr.ptr));
                }

                if (static_cast<int>(pred.size()) != N) {
                    throw std::runtime_error("UnifiedTree::predict_binned: expected one scalar prediction per row");
                }

                py::array_t<double> out({static_cast<py::ssize_t>(N)});
                std::memcpy(out.request().ptr, pred.data(), pred.size() * sizeof(double));
                return out;
            },
            py::arg("Xb"), py::arg("Xraw") = py::none(),
            "Predict one scalar value per row from binned features. "
            "If the tree contains neural leaves, `Xraw` must be provided for correct inference.")

        .def(
            "predict_contrib_binned",
            [](const UnifiedTree& self,
               const py::array_t<uint16_t, py::array::c_style |
                                               py::array::forcecast>& Xb,
               py::object Xraw_opt) {
                auto xb = Xb.request();
                if (xb.ndim != 2)
                    throw std::invalid_argument("Xb must be 2D (N, P)");
                const int N = static_cast<int>(xb.shape[0]);
                const int P = static_cast<int>(xb.shape[1]);

                std::vector<uint16_t> Xv = arr_to_vec_any<uint16_t>(Xb);
                std::vector<double> contrib;
                if (Xraw_opt.is_none()) {
                    contrib = self.predict_contrib(Xv, N, P);
                } else {
                    auto Xraw = Xraw_opt.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                    auto xr   = Xraw.request();
                    if (xr.ndim != 2)
                        throw std::invalid_argument("Xraw must be 2D (N, P)");
                    if (xr.shape[0] != xb.shape[0] || xr.shape[1] != xb.shape[1])
                        throw std::invalid_argument("Xraw shape must match Xb shape");
                    contrib = self.predict_contrib(Xv, N, P, static_cast<const double*>(xr.ptr));
                }

                py::array_t<double> out({static_cast<py::ssize_t>(N), static_cast<py::ssize_t>(P + 1)});
                if (!contrib.empty()) {
                    std::memcpy(out.request().ptr, contrib.data(), contrib.size() * sizeof(double));
                }
                return out;
            },
            py::arg("Xb"), py::arg("Xraw") = py::none(),
            "Predict TreeSHAP contributions on the raw-margin scale. The last column is the bias term.")

        // --- introspection: properties ---
        .def_property_readonly("n_nodes", &UnifiedTree::n_nodes)
        .def_property_readonly("n_leaves", &UnifiedTree::n_leaves)
        .def_property_readonly("depth", &UnifiedTree::depth)

        // --- feature importance (method returning a Python list) ---
        .def("feature_importance_gain",
             [](const UnifiedTree& self) {
                 return self.feature_importance_gain();
             })
        .def("post_prune_ccp", &UnifiedTree::post_prune_ccp,
             py::arg("ccp_alpha"));
}
