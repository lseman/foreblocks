// src/pybind/foretree.cpp
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>

#include <cstring> // memcpy

// headers (updated paths)
#include "../../include/foretree/core.hpp"
#include "../../include/foretree/tree.hpp"
#include "../../include/foretree/tree/packed_tree.hpp"

#ifdef FORETREE_HAS_CUDA
#include "../../include/foretree/gpu/gpu_prediction.hpp"
using foretree::cuda::GpuPredictionEngine;
#endif

namespace nb = nanobind;
using namespace nanobind::literals;

using foretree::DataBinner;
using foretree::EdgeSet;
using foretree::FeatureBins;
using foretree::PackedTree;
using foretree::GradientHistogramSystem;
using foretree::HistogramConfig;
using foretree::TreeConfig;
using foretree::UnifiedTree;

template <typename Array> struct ArrayView {
    size_t ndim;
    std::vector<size_t> shape;
    decltype(std::declval<Array>().data()) ptr;
};

template <typename Array> static ArrayView<Array> request(const Array& array) {
    std::vector<size_t> shape(array.ndim());
    for (size_t i = 0; i < array.ndim(); ++i)
        shape[i] = array.shape(i);
    return {array.ndim(), std::move(shape), array.data()};
}

template <typename T> static nb::ndarray<nb::numpy, T, nb::c_contig> make_array(std::initializer_list<size_t> shape) {
    size_t size = 1;
    for (size_t extent : shape)
        size *= extent;
    T* data = new T[size];
    nb::capsule owner(data, [](void* pointer) noexcept { delete[] static_cast<T*>(pointer); });
    return nb::ndarray<nb::numpy, T, nb::c_contig>(data, shape, owner);
}

template <typename T>
static std::vector<T> arr_to_vec_1d(const nb::ndarray<nb::numpy, T, nb::c_contig>& a) {
    auto buf = request(a);
    if (buf.ndim != 1)
        throw std::invalid_argument("Expected a 1D array");
    const T* ptr = static_cast<const T*>(buf.ptr);
    return std::vector<T>(ptr, ptr + buf.shape[0]);
}

template <typename T>
static std::vector<T> arr_to_vec_any(const nb::ndarray<nb::numpy, T, nb::c_contig>& a) {
    auto buf = request(a);
    size_t n = 1;
    for (auto s : buf.shape)
        n *= static_cast<size_t>(s);
    const T* ptr = static_cast<const T*>(buf.ptr);
    return std::vector<T>(ptr, ptr + n);
}

NB_MODULE(foretree, m) {
    m.doc() = "Foretree: gradient-aware binning + unified tree (nanobind)";

    // EdgeSet
    nb::class_<EdgeSet>(m, "EdgeSet")
        .def(nb::init<>())
        .def_rw("edges_per_feat", &EdgeSet::edges_per_feat)
        .def_rw("finite_bins", &EdgeSet::finite_bins)
        .def_rw("missing_bin_id", &EdgeSet::missing_bin_id)
        .def_ro("finite_bins_per_feat", &EdgeSet::finite_bins_per_feat)
        .def_ro("missing_bin_id_per_feat", &EdgeSet::missing_bin_id_per_feat);

    // DataBinner
    nb::class_<DataBinner>(m, "DataBinner")
        .def(nb::init<int>(), nb::arg("P"))
        .def(
            "register_edges",
            [](DataBinner& db, const std::string& mode, const std::vector<std::vector<double>>& edges_per_feat) {
                EdgeSet es;
                es.edges_per_feat = edges_per_feat;
                db.register_edges(mode, es);
            },
            nb::arg("mode"), nb::arg("edges_per_feat"))
        .def("register_edges",
             static_cast<void (DataBinner::*)(const std::string&, EdgeSet)>(&DataBinner::register_edges),
             nb::arg("mode"), nb::arg("edgeset"))
        .def("set_node_override", &DataBinner::set_node_override, nb::arg("mode"), nb::arg("node_id"), nb::arg("feat"),
             nb::arg("edges"))
        .def(
            "prebin",
            [](const DataBinner& db, const nb::ndarray<nb::numpy, double, nb::c_contig>& X,
               const std::string& mode, int node_id) {
                auto buf = request(X);
                if (buf.ndim != 2) {
                    throw std::runtime_error("Input array must be 2-dimensional");
                }
                int N = static_cast<int>(buf.shape[0]);
                int P = static_cast<int>(buf.shape[1]);

                auto result = db.prebin(static_cast<const double*>(buf.ptr), N, P, mode, node_id);

                auto codes_array = make_array<uint16_t>({static_cast<size_t>(N), static_cast<size_t>(P)});
                std::memcpy(codes_array.data(), result.first->data(), result.first->size() * sizeof(uint16_t));

                return nb::make_tuple(codes_array, result.second);
            },
            nb::arg("X"), nb::arg("mode"), nb::arg("node_id") = -1, "Returns tuple of (codes_array, missing_bin_id)")
        .def(
            "prebin_into",
            [](const DataBinner& db, const nb::ndarray<nb::numpy, double, nb::c_contig>& X,
               const std::string& mode, const nb::ndarray<nb::numpy, uint16_t, nb::c_contig>& out_codes, int node_id) {
                auto X_buf = request(X);
                auto out_buf = request(out_codes);

                if (X_buf.ndim != 2 || out_buf.ndim != 2) {
                    throw std::runtime_error("Arrays must be 2-dimensional");
                }

                int N = static_cast<int>(X_buf.shape[0]);
                int P = static_cast<int>(X_buf.shape[1]);
                if (out_buf.shape[0] != X_buf.shape[0] || out_buf.shape[1] != X_buf.shape[1]) {
                    throw std::runtime_error("out_codes shape must match input X");
                }

                return db.prebin_into(static_cast<const double*>(X_buf.ptr), N, P, mode,
                                      static_cast<uint16_t*>(out_buf.ptr), node_id);
            },
            nb::arg("X"), nb::arg("mode"), nb::arg("out_codes"), nb::arg("node_id") = -1)

        .def("finite_bins", static_cast<int (DataBinner::*)(const std::string&) const>(&DataBinner::finite_bins),
             nb::arg("mode"))
        .def("missing_bin_id", static_cast<int (DataBinner::*)(const std::string&) const>(&DataBinner::missing_bin_id),
             nb::arg("mode"))
        .def("total_bins", static_cast<int (DataBinner::*)(const std::string&) const>(&DataBinner::total_bins),
             nb::arg("mode"))

        .def("finite_bins", static_cast<int (DataBinner::*)(const std::string&, int) const>(&DataBinner::finite_bins),
             nb::arg("mode"), nb::arg("feat"))
        .def("missing_bin_id",
             static_cast<int (DataBinner::*)(const std::string&, int) const>(&DataBinner::missing_bin_id),
             nb::arg("mode"), nb::arg("feat"))
        .def("total_bins", static_cast<int (DataBinner::*)(const std::string&, int) const>(&DataBinner::total_bins),
             nb::arg("mode"), nb::arg("feat"))

        .def("finite_bins_per_feat", &DataBinner::finite_bins_per_feat, nb::arg("mode"),
             nb::rv_policy::reference_internal)
        .def("missing_bin_ids_per_feat", &DataBinner::missing_bin_ids_per_feat, nb::arg("mode"),
             nb::rv_policy::reference_internal)

        .def("P", &DataBinner::P);

    // HistogramConfig
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

    // FeatureBins
    nb::class_<FeatureBins>(m, "FeatureBins")
        .def(nb::init<>())
        .def_rw("edges", &FeatureBins::edges)
        .def_rw("is_uniform", &FeatureBins::is_uniform)
        .def_rw("strategy", &FeatureBins::strategy)
        .def_rw("lo", &FeatureBins::lo)
        .def_rw("width", &FeatureBins::width)
        .def_rw("stats", &FeatureBins::stats)
        .def("n_bins", &FeatureBins::n_bins);

    // GradientHistogramSystem
    nb::class_<GradientHistogramSystem>(m, "GradientHistogramSystem")
        .def(nb::init<HistogramConfig>(), nb::arg("config"))
        .def(
            "fit_bins",
            [](GradientHistogramSystem& ghs, const nb::ndarray<nb::numpy, double, nb::c_contig>& X,
               const nb::ndarray<nb::numpy, double, nb::c_contig>& g,
               const nb::ndarray<nb::numpy, double, nb::c_contig>& h) {
                auto X_buf = request(X);
                auto g_buf = request(g);
                auto h_buf = request(h);

                if (X_buf.ndim != 2) {
                    throw std::runtime_error("X must be 2-dimensional");
                }
                if (g_buf.ndim != 1 || h_buf.ndim != 1) {
                    throw std::runtime_error("g and h must be 1-dimensional");
                }

                int N = static_cast<int>(X_buf.shape[0]);
                int P = static_cast<int>(X_buf.shape[1]);

                if (g_buf.shape[0] != N || h_buf.shape[0] != N) {
                    throw std::runtime_error("g and h must have same length as X rows");
                }

                ghs.fit_bins(static_cast<const double*>(X_buf.ptr), N, P, static_cast<const double*>(g_buf.ptr),
                             static_cast<const double*>(h_buf.ptr));
            },
            nb::arg("X"), nb::arg("g"), nb::arg("h"))
        .def(
            "prebin_dataset",
            [](GradientHistogramSystem& ghs, const nb::ndarray<nb::numpy, double, nb::c_contig>& X) {
                auto buf = request(X);
                if (buf.ndim != 2) {
                    throw std::runtime_error("X must be 2-dimensional");
                }
                int N = static_cast<int>(buf.shape[0]);
                int P = static_cast<int>(buf.shape[1]);

                auto result = ghs.prebin_dataset(static_cast<const double*>(buf.ptr), N, P);

                auto codes_array = make_array<uint16_t>({static_cast<size_t>(N), static_cast<size_t>(P)});
                std::memcpy(codes_array.data(), result.first->data(), result.first->size() * sizeof(uint16_t));

                return nb::make_tuple(codes_array, result.second);
            },
            nb::arg("X"), "Returns tuple of (codes_array, missing_bin_id)")
        .def(
            "prebin_matrix",
            [](const GradientHistogramSystem& ghs, const nb::ndarray<nb::numpy, double, nb::c_contig>& X) {
                auto buf = request(X);
                if (buf.ndim != 2) {
                    throw std::runtime_error("X must be 2-dimensional");
                }
                int N = static_cast<int>(buf.shape[0]);
                int P = static_cast<int>(buf.shape[1]);

                auto result = ghs.prebin_matrix(static_cast<const double*>(buf.ptr), N, P);

                auto codes_array = make_array<uint16_t>({static_cast<size_t>(N), static_cast<size_t>(P)});
                std::memcpy(codes_array.data(), result.first->data(), result.first->size() * sizeof(uint16_t));

                return nb::make_tuple(codes_array, result.second);
            },
            nb::arg("X"), "Returns tuple of (codes_array, missing_bin_id)")
        .def(
            "build_histograms_fast",
            [](const GradientHistogramSystem& ghs, const nb::ndarray<nb::numpy, float, nb::c_contig>& g,
               const nb::ndarray<nb::numpy, float, nb::c_contig>& h,
               const nb::ndarray<nb::numpy, int, nb::c_contig>& sample_indices) {
                auto g_buf = request(g);
                auto h_buf = request(h);

                if (g_buf.ndim != 1 || h_buf.ndim != 1) {
                    throw std::runtime_error("g and h must be 1-dimensional");
                }

                const int* indices_ptr = nullptr;
                int n_sub = 0;
                if (sample_indices.size() > 0) {
                    auto idx_buf = request(sample_indices);
                    if (idx_buf.ndim != 1) {
                        throw std::runtime_error("sample_indices must be 1-dimensional");
                    }
                    indices_ptr = static_cast<const int*>(idx_buf.ptr);
                    n_sub = static_cast<int>(idx_buf.shape[0]);
                }

                auto result = ghs.build_histograms_fast(static_cast<const float*>(g_buf.ptr),
                                                        static_cast<const float*>(h_buf.ptr), indices_ptr, n_sub);

                return nb::make_tuple(result.first, result.second);
            },
            nb::arg("g"), nb::arg("h"),
            nb::arg("sample_indices") = nb::ndarray<nb::numpy, int, nb::c_contig>(), "Returns tuple of (Hg, Hh)")
        .def(
            "build_histograms_fast_with_counts",
            [](const GradientHistogramSystem& ghs, const nb::ndarray<nb::numpy, float, nb::c_contig>& g,
               const nb::ndarray<nb::numpy, float, nb::c_contig>& h,
               const nb::ndarray<nb::numpy, int, nb::c_contig>& sample_indices) {
                auto g_buf = request(g);
                auto h_buf = request(h);

                if (g_buf.ndim != 1 || h_buf.ndim != 1) {
                    throw std::runtime_error("g and h must be 1-dimensional");
                }

                const int* indices_ptr = nullptr;
                int n_sub = 0;
                if (sample_indices.size() > 0) {
                    auto idx_buf = request(sample_indices);
                    if (idx_buf.ndim != 1) {
                        throw std::runtime_error("sample_indices must be 1-dimensional");
                    }
                    indices_ptr = static_cast<const int*>(idx_buf.ptr);
                    n_sub = static_cast<int>(idx_buf.shape[0]);
                }

                auto result = ghs.build_histograms_fast_with_counts(
                    static_cast<const float*>(g_buf.ptr), static_cast<const float*>(h_buf.ptr), indices_ptr, n_sub);

                return nb::make_tuple(std::get<0>(result), std::get<1>(result), std::get<2>(result));
            },
            nb::arg("g"), nb::arg("h"),
            nb::arg("sample_indices") = nb::ndarray<nb::numpy, int, nb::c_contig>(), "Returns tuple of (Hg, Hh, C)")
        .def(
            "extract_feature_histogram",
            [](const GradientHistogramSystem& ghs, const std::vector<double>& Hg, const std::vector<double>& Hh,
               const std::vector<int>& C, int feature) {
                auto result = ghs.extract_feature_histogram(Hg, Hh, C, feature);
                return nb::make_tuple(std::get<0>(result), std::get<1>(result), std::get<2>(result));
            },
            nb::arg("Hg"), nb::arg("Hh"), nb::arg("C"), nb::arg("feature"),
            "Returns tuple of (feat_Hg, feat_Hh, feat_C)")
        .def("get_feature_offsets", &GradientHistogramSystem::get_feature_offsets)
        .def("get_bin_allocation_summary", &GradientHistogramSystem::get_bin_allocation_summary)

        .def("P", &GradientHistogramSystem::P)
        .def("N", &GradientHistogramSystem::N)
        .def("missing_bin_id",
             static_cast<int (GradientHistogramSystem::*)() const>(&GradientHistogramSystem::missing_bin_id))
        .def("finite_bins",
             static_cast<int (GradientHistogramSystem::*)() const>(&GradientHistogramSystem::finite_bins))
        .def("total_bins", static_cast<int (GradientHistogramSystem::*)() const>(&GradientHistogramSystem::total_bins))

        .def("finite_bins",
             static_cast<int (GradientHistogramSystem::*)(int) const>(&GradientHistogramSystem::finite_bins),
             nb::arg("feature"))
        .def("total_bins",
             static_cast<int (GradientHistogramSystem::*)(int) const>(&GradientHistogramSystem::total_bins),
             nb::arg("feature"))
        .def("missing_bin_id",
             static_cast<int (GradientHistogramSystem::*)(int) const>(&GradientHistogramSystem::missing_bin_id),
             nb::arg("feature"))

        .def("all_finite_bins", &GradientHistogramSystem::all_finite_bins)
        .def("all_total_bins", &GradientHistogramSystem::all_total_bins)

        .def("feature_stats", &GradientHistogramSystem::feature_stats, nb::arg("feature"),
             nb::rv_policy::reference_internal)
        .def("feature_bins", &GradientHistogramSystem::feature_bins, nb::arg("feature"),
             nb::rv_policy::reference_internal)

        .def("binner", &GradientHistogramSystem::binner, nb::rv_policy::reference_internal)
        .def(
            "codes_view",
            [](const GradientHistogramSystem& ghs) {
                auto codes_ptr = ghs.codes_view();
                if (!codes_ptr) {
                    throw std::runtime_error("No codes available. Call prebin_dataset first.");
                }

                int N = ghs.N();
                int P = ghs.P();
                auto codes_array = make_array<uint16_t>({static_cast<size_t>(N), static_cast<size_t>(P)});
                std::memcpy(codes_array.data(), codes_ptr->data(), codes_ptr->size() * sizeof(uint16_t));

                return codes_array;
            },
            "Returns the cached binned codes as numpy array (N x P)");

    // TreeConfig enums
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

    // TreeConfig
    nb::class_<TreeConfig>(m, "TreeConfig")
        .def(nb::init<>())
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
        .def_rw("growth", &TreeConfig::growth)
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
        .def_rw("missing_policy", &TreeConfig::missing_policy)
        .def_rw("monotone_constraints", &TreeConfig::monotone_constraints)
        .def_rw("split_mode", &TreeConfig::split_mode)
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
        .def_rw("goss", &TreeConfig::goss)
        .def_rw("num_classes", &TreeConfig::num_classes);

    // PackedTree
    nb::class_<PackedTree>(m, "PackedTree")
        .def_ro("root", &PackedTree::root)
        .def_ro("outputs", &PackedTree::outputs)
        .def_ro("features", &PackedTree::features)
        .def_ro("thresholds", &PackedTree::thresholds)
        .def_ro("split_values", &PackedTree::split_values)
        .def_ro("split_kinds", &PackedTree::split_kinds)
        .def_ro("missing_left", &PackedTree::missing_left)
        .def_ro("left_children", &PackedTree::left_children)
        .def_ro("right_children", &PackedTree::right_children)
        .def_ro("leaf_flags", &PackedTree::leaf_flags)
        .def_ro("cover", &PackedTree::cover)
        .def_ro("categorical_offsets", &PackedTree::categorical_offsets)
        .def_ro("categorical_counts", &PackedTree::categorical_counts)
        .def_ro("categorical_bins", &PackedTree::categorical_bins)
        .def_ro("pair_features_a", &PackedTree::pair_features_a)
        .def_ro("pair_features_b", &PackedTree::pair_features_b)
        .def_ro("pair_thresholds_a", &PackedTree::pair_thresholds_a)
        .def_ro("pair_thresholds_b", &PackedTree::pair_thresholds_b)
        .def_ro("pair_quadrant_masks", &PackedTree::pair_quadrant_masks)
        .def_ro("oblique_offsets", &PackedTree::oblique_offsets)
        .def_ro("oblique_counts", &PackedTree::oblique_counts)
        .def_ro("oblique_features", &PackedTree::oblique_features)
        .def_ro("oblique_weights", &PackedTree::oblique_weights)
        .def_ro("oblique_thresholds", &PackedTree::oblique_thresholds)
        .def_ro("leaf_values", &PackedTree::leaf_values)
        .def_prop_ro("node_count", [](const PackedTree& self) { return self.node_count(); });

    // UnifiedTree
    nb::class_<UnifiedTree>(m, "UnifiedTree", nb::dynamic_attr())
        .def(nb::init<>())
        .def("__init__", [](UnifiedTree* self, TreeConfig config,
                            const std::shared_ptr<GradientHistogramSystem>& ghs) {
                 new (self) UnifiedTree(std::move(config), ghs.get());
             },
             nb::arg("config"), nb::arg("ghs"), nb::keep_alive<1, 2>())

        .def(
            "set_raw_matrix",
            [](UnifiedTree& self, const nb::ndarray<nb::numpy, double, nb::c_contig>& Xraw,
               nb::object mask /* None or uint8 array */) {
                auto xb = request(Xraw);
                if (xb.ndim != 2)
                    throw std::invalid_argument("Xraw must be 2D float64 (N, P)");
                const double* Xptr = static_cast<const double*>(xb.ptr);

                const uint8_t* mptr = nullptr;
                nb::ndarray<nb::numpy, uint8_t, nb::c_contig> mask_arr;
                if (!mask.is_none()) {
                    mask_arr = nb::cast<nb::ndarray<nb::numpy, uint8_t, nb::c_contig>>(mask);
                    auto mb = request(mask_arr);
                    if (mb.ndim != 2)
                        throw std::invalid_argument("mask must be 2D uint8 (N, P)");
                    if (mb.shape[0] != xb.shape[0] || mb.shape[1] != xb.shape[1])
                        throw std::invalid_argument("mask shape must match Xraw");
                    mptr = static_cast<const uint8_t*>(mb.ptr);
                }

                self.set_raw_matrix(Xptr, mptr);

                nb::object self_obj = nb::cast(&self);
                self_obj.attr("_Xraw_ref") = Xraw;
                if (!mask.is_none())
                    self_obj.attr("_Xmask_ref") = mask_arr;
            },
            nb::arg("Xraw"), nb::arg("mask") = nb::none())

        .def(
            "fit_binned",
            [](UnifiedTree& self, const nb::ndarray<nb::numpy, uint16_t, nb::c_contig>& Xb,
               const nb::ndarray<nb::numpy, double, nb::c_contig>& g,
               const nb::ndarray<nb::numpy, double, nb::c_contig>& h) {
                auto xb = request(Xb);
                if (xb.ndim != 2)
                    throw std::invalid_argument("Xb must be 2D (N, P)");
                const int N = static_cast<int>(xb.shape[0]);
                const int P = static_cast<int>(xb.shape[1]);

                std::vector<uint16_t> Xv = arr_to_vec_any<uint16_t>(Xb);
                std::vector<double> gv = arr_to_vec_1d<double>(g);
                std::vector<double> hv = arr_to_vec_1d<double>(h);
                if ((int)gv.size() != N || (int)hv.size() != N)
                    throw std::invalid_argument("g and h must have length N = Xb.shape[0]");

                self.fit(Xv, N, P, gv, hv);
            },
            nb::arg("Xb"), nb::arg("g"), nb::arg("h"),
            "Fit a scalar-output tree on binned features with 1-D gradient and hessian arrays.")

        .def(
            "predict_binned",
            [](const UnifiedTree& self, const nb::ndarray<nb::numpy, uint16_t, nb::c_contig>& Xb,
               nb::object Xraw_opt) {
                auto xb = request(Xb);
                if (xb.ndim != 2)
                    throw std::invalid_argument("Xb must be 2D (N, P)");
                const int N = static_cast<int>(xb.shape[0]);
                const int P = static_cast<int>(xb.shape[1]);

                std::vector<uint16_t> Xv = arr_to_vec_any<uint16_t>(Xb);
                std::vector<double> pred;
                if (Xraw_opt.is_none()) {
                    pred = self.predict(Xv, N, P);
                } else {
                    auto Xraw = nb::cast<nb::ndarray<nb::numpy, double, nb::c_contig>>(Xraw_opt);
                    auto xr = request(Xraw);
                    if (xr.ndim != 2)
                        throw std::invalid_argument("Xraw must be 2D (N, P)");
                    if (xr.shape[0] != xb.shape[0] || xr.shape[1] != xb.shape[1])
                        throw std::invalid_argument("Xraw shape must match Xb shape");
                    pred = self.predict(Xv, N, P, static_cast<const double*>(xr.ptr));
                }

                int K = std::max(self.cfg_.num_classes - 1, 1);
                if (static_cast<int>(pred.size()) != N * K) {
                    throw std::runtime_error("UnifiedTree::predict_binned: prediction size mismatch");
                }

                if (K <= 1) {
                    auto out = make_array<double>({N});
                    if (!pred.empty())
                        std::memcpy(out.data(), pred.data(), pred.size() * sizeof(double));
                    return out;
                } else {
                    auto out = make_array<double>({N, K});
                    if (!pred.empty())
                        std::memcpy(out.data(), pred.data(), pred.size() * sizeof(double));
                    return out;
                }
            },
            nb::arg("Xb"), nb::arg("Xraw") = nb::none(),
            "Predict from binned features. Returns (N,) for scalar or (N, K) for multiclass.")

        .def(
            "predict_contrib_binned",
            [](const UnifiedTree& self, const nb::ndarray<nb::numpy, uint16_t, nb::c_contig>& Xb,
               nb::object Xraw_opt) {
                auto xb = request(Xb);
                if (xb.ndim != 2)
                    throw std::invalid_argument("Xb must be 2D (N, P)");
                const int N = static_cast<int>(xb.shape[0]);
                const int P = static_cast<int>(xb.shape[1]);

                std::vector<uint16_t> Xv = arr_to_vec_any<uint16_t>(Xb);
                std::vector<double> contrib;
                if (Xraw_opt.is_none()) {
                    contrib = self.predict_contrib(Xv, N, P);
                } else {
                    auto Xraw = nb::cast<nb::ndarray<nb::numpy, double, nb::c_contig>>(Xraw_opt);
                    auto xr = request(Xraw);
                    if (xr.ndim != 2)
                        throw std::invalid_argument("Xraw must be 2D (N, P)");
                    if (xr.shape[0] != xb.shape[0] || xr.shape[1] != xb.shape[1])
                        throw std::invalid_argument("Xraw shape must match Xb shape");
                    contrib = self.predict_contrib(Xv, N, P, static_cast<const double*>(xr.ptr));
                }

                int K = std::max(self.cfg_.num_classes - 1, 1);
                if (K <= 1) {
                    auto out = make_array<double>({N, P + 1});
                    if (!contrib.empty())
                        std::memcpy(out.data(), contrib.data(), contrib.size() * sizeof(double));
                    return out;
                } else {
                    auto out = make_array<double>({N, K * (P + 1)});
                    if (!contrib.empty())
                        std::memcpy(out.data(), contrib.data(), contrib.size() * sizeof(double));
                    return out;
                }
            },
            nb::arg("Xb"), nb::arg("Xraw") = nb::none(),
            "Predict TreeSHAP contributions. Returns (N, P+1) for scalar or (N, K*(P+1)) for multiclass.")

        .def_prop_ro("n_nodes", &UnifiedTree::n_nodes)
        .def_prop_ro("n_leaves", &UnifiedTree::n_leaves)
        .def_prop_ro("depth", &UnifiedTree::depth)

        .def("feature_importance_gain", [](const UnifiedTree& self) { return self.feature_importance_gain(); })
        .def("feature_importance_cover", [](const UnifiedTree& self) { return self.feature_importance_cover(); })
        .def("feature_importance_frequency", [](const UnifiedTree& self) { return self.feature_importance_frequency(); })
        .def("post_prune_ccp", &UnifiedTree::post_prune_ccp, nb::arg("ccp_alpha"))
        .def(
            "get_packed_tree",
            [](const UnifiedTree& self) -> const PackedTree& {
                return self.get_packed_tree();
            },
            nb::rv_policy::reference_internal,
            "Return the packed tree representation built during fit.");

#ifdef FORETREE_HAS_CUDA
    // GPU prediction engine - accepts PackedTree or tuple of numpy arrays
    nb::class_<GpuPredictionEngine>(m, "GpuPredictionEngine")
        .def(nb::init<const PackedTree&>(), nb::arg("packed_tree"))
        .def(
            "__init__",
            [](GpuPredictionEngine* self,
               nb::tuple tree_arrays) {
                // Expect 23 numpy arrays in tuple (matching foreforest.PackedTree tuple order)
                if (tree_arrays.size() != 23)
                    throw std::invalid_argument("GpuPredictionEngine: expected tuple of 23 arrays");

                // Helper to get span from ndarray
                auto get_span = [](const nb::object& arr, const char* name, bool is_float = false) {
                    auto na = nb::cast<nb::ndarray<nb::numpy, double, nb::c_contig>>(arr);
                    if (is_float) {
                        throw std::invalid_argument(std::string(name) + ": expected float64");
                    }
                    auto buf = request(na);
                    if (buf.ndim != 1)
                        throw std::invalid_argument(std::string(name) + ": expected 1D array");
                    return std::span<const double>(static_cast<const double*>(buf.ptr), buf.shape[0]);
                };
                auto get_ispan = [](const nb::object& arr, const char* name) {
                    auto na = nb::cast<nb::ndarray<nb::numpy, int, nb::c_contig>>(arr);
                    auto buf = request(na);
                    if (buf.ndim != 1)
                        throw std::invalid_argument(std::string(name) + ": expected 1D int array");
                    return std::span<const int>(static_cast<const int*>(buf.ptr), buf.shape[0]);
                };
                auto get_u8span = [](const nb::object& arr, const char* name) {
                    auto na = nb::cast<nb::ndarray<nb::numpy, uint8_t, nb::c_contig>>(arr);
                    auto buf = request(na);
                    if (buf.ndim != 1)
                        throw std::invalid_argument(std::string(name) + ": expected 1D uint8 array");
                    return std::span<const uint8_t>(static_cast<const uint8_t*>(buf.ptr), buf.shape[0]);
                };

                std::vector<std::span<const int>> int_a(20);
                int_a[0] = get_ispan(tree_arrays[0], "features");
                int_a[1] = get_ispan(tree_arrays[1], "thresholds");
                int_a[5] = get_ispan(tree_arrays[5], "left_children");
                int_a[6] = get_ispan(tree_arrays[6], "right_children");
                int_a[9] = get_ispan(tree_arrays[9], "cat_offsets");
                int_a[10] = get_ispan(tree_arrays[10], "cat_counts");
                int_a[11] = get_ispan(tree_arrays[11], "cat_bins");
                int_a[12] = get_ispan(tree_arrays[12], "pair_a");
                int_a[13] = get_ispan(tree_arrays[13], "pair_b");
                int_a[14] = get_ispan(tree_arrays[14], "pair_thresh_a");
                int_a[15] = get_ispan(tree_arrays[15], "pair_thresh_b");
                int_a[17] = get_ispan(tree_arrays[17], "oblique_offsets");
                int_a[18] = get_ispan(tree_arrays[18], "oblique_counts");
                int_a[19] = get_ispan(tree_arrays[19], "oblique_features");

                std::vector<std::span<const double>> dbl_a(5);
                dbl_a[0] = get_span(tree_arrays[2], "split_values");
                dbl_a[1] = get_span(tree_arrays[8], "cover");
                dbl_a[2] = get_span(tree_arrays[20], "oblique_weights");
                dbl_a[3] = get_span(tree_arrays[21], "oblique_thresholds");
                dbl_a[4] = get_span(tree_arrays[22], "leaf_values");

                std::vector<std::span<const uint8_t>> u8_a(4);
                u8_a[0] = get_u8span(tree_arrays[3], "split_kinds");
                u8_a[1] = get_u8span(tree_arrays[4], "missing_left");
                u8_a[2] = get_u8span(tree_arrays[7], "leaf_flags");
                u8_a[3] = get_u8span(tree_arrays[16], "pair_quadrant_masks");

                // num_features is computed at predict time from codes.shape[1],
                // not from the tree's feature IDs. Pass 0 as placeholder.
                new (self) GpuPredictionEngine(int_a, dbl_a, u8_a, 0);
            },
            nb::arg("tree_arrays"), "Construct from tuple of 23 numpy arrays (matching PackedTree field order).")
        .def(
            "predict",
            [](const GpuPredictionEngine& self,
               const nb::ndarray<nb::numpy, uint16_t, nb::c_contig>& codes) {
                auto buf = request(codes);
                if (buf.ndim != 2)
                    throw std::invalid_argument("codes must be 2D (N, P)");
                int N = static_cast<int>(buf.shape[0]);
                std::vector<uint16_t> v(buf.shape[0] * buf.shape[1]);
                std::memcpy(v.data(), buf.ptr, v.size() * sizeof(uint16_t));
                auto result = self.predict_binned_uint16(
                    std::span<const uint16_t>(v.data(), v.size()), N);
                auto out = make_array<double>({static_cast<size_t>(N)});
                if (!result.empty())
                    std::memcpy(out.data(), result.data(),
                                std::min(result.size(), static_cast<size_t>(N)) * sizeof(double));
                return out;
            },
            nb::arg("codes"), "Predict on binned data (uint16 codes). Returns (N,).")
        .def_prop_ro("outputs", [](const GpuPredictionEngine& self) { return self.outputs(); })
        .def_prop_ro("trees", [](const GpuPredictionEngine& self) { return self.trees(); })
        .def_prop_ro("device_bytes", [](const GpuPredictionEngine& self) { return self.device_bytes(); });

    m.def("gpu_available", &foretree::cuda::is_available,
          "Check if CUDA is available for GPU prediction.");
#endif
}
