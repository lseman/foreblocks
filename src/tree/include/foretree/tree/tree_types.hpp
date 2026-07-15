#pragma once

#include "foretree/split/split_engine.hpp"
#include "foretree/tree/neural.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <ranges>
#include <vector>

namespace foretree {

struct TreeConfig {
    int max_depth = 6;
    int max_leaves = 31;
    int min_samples_split = 10;
    int min_samples_leaf = 5;
    double min_child_weight = 1e-3;
    double lambda_ = 1.0;
    double alpha_ = 0.0;
    double gamma_ = 0.0;
    double max_delta_step = 0.0;
    int n_bins = 256;
    int num_classes = 1; // >1 => multiclass C-1 output
    enum class Growth { LeafWise, LevelWise, Oblivious } growth = Growth::LeafWise;
    double leaf_gain_eps = 0.0;
    bool allow_zero_gain = false;
    double leaf_depth_penalty = 0.0;
    double leaf_hess_boost = 0.0;
    int feature_bagging_k = -1;
    bool feature_bagging_with_replacement = false;
    int colsample_bytree_percent = 100;
    int colsample_bylevel_percent = 100;
    int colsample_bynode_percent = 100;
    bool use_sibling_subtract = true;

    enum class MissingPolicy { Learn, AlwaysLeft, AlwaysRight };
    MissingPolicy missing_policy = MissingPolicy::Learn;
    std::vector<int8_t> monotone_constraints;

    enum class SplitMode { Histogram, Exact, Hybrid };
    SplitMode split_mode = SplitMode::Histogram;
    int exact_cutover = 2048;
    bool enable_categorical_splits = false;
    int categorical_max_selected_categories = 8;
    bool enable_oblique_splits = false;
    ObliqueMode oblique_mode = ObliqueMode::Full;
    int oblique_k_features = 4;
    int oblique_newton_steps = 1;
    double oblique_l1 = 0.0;
    double oblique_ridge = 1e-3;
    double axis_vs_oblique_guard = 1.02;
    InteractionSeededConfig interaction_seeded_oblique{};
    bool enable_pair_interaction_splits = false;
    PairInteractionConfig pair_interaction{};
    std::vector<std::vector<int>> interaction_constraints;

    double subsample_bytree = 1.0;
    double subsample_bylevel = 1.0;
    double subsample_bynode = 1.0;
    bool subsample_with_replacement = true;
    bool subsample_importance_scale = false;

    int rng_seed = 123456789;

    struct GOSS {
        bool enabled = false;
        double top_rate = 0.2;
        double other_rate = 0.1;
        bool scale_hessian = true;
        int min_node_size = 2000;
        bool use_random_rest = true;
        bool adaptive = true;
        double adaptive_scale = 1.0;
    } goss;

    bool cache_histograms = true;
    int cache_threshold = 1000;
    int max_histogram_pool_size = 100;
    int cuda_min_histogram_work = 262144;

    struct OnTree {
        bool enabled = false;
        double ccp_alpha = 0.0;
        double min_gain = 0.0;
        double min_gain_rel = 0.0;
        double min_impurity_decrease = 0.0;
        double eps = 1e-12;
    } on_tree;

    bool sgld_enabled = false;
    double sgld_noise_scale = 1.0;

    struct NeuralLeaf {
        bool enabled = false;
        int min_samples = 20;
        int max_depth_start = 2;
        double complexity_threshold = 1e-3;
    } neural_leaf;
};

struct HistPair;

// Mutable node used only while fitting. Histogram handles, row ranges and
// gradient statistics intentionally do not belong to the inference model.
struct TrainingNode {
    int id = -1;
    bool is_leaf = true;
    int depth = 0;

    int feature = -1;
    int thr = -1;
    double split_value = std::numeric_limits<double>::quiet_NaN();
    bool miss_left = true;
    splitx::SplitKind split_kind = splitx::SplitKind::Axis;
    std::vector<int> categorical_left_bins;
    std::vector<int> oblique_features;
    std::vector<double> oblique_weights;
    double oblique_threshold = std::numeric_limits<double>::quiet_NaN();
    bool oblique_missing_left = true;
    int pair_feature_a = -1;
    int pair_feature_b = -1;
    int pair_threshold_a = -1;
    int pair_threshold_b = -1;
    uint8_t pair_quadrant_mask = 0;
    bool pair_missing_left = true;
    std::vector<int> path_features;
    int left = -1, right = -1, sibling = -1;

    // Gradient/hessian sums (size 1 for scalar, K=num_classes-1 for multiclass)
    std::vector<double> G;
    std::vector<double> H;
    int C = 0;
    int lo = 0, hi = 0;
    int K = 1; // num classes - 1; 1 means scalar

    double best_gain = -1e300;
    std::vector<double> leaf_values; // size 1 for scalar, K for multiclass
    // GOSS weighted sums (size K)
    std::vector<double> goss_weighted_G;
    std::vector<double> goss_weighted_H;
    double goss_rest_scale = 1.0;

    bool uses_goss = false;

    double min_constraint = -std::numeric_limits<double>::infinity();
    double max_constraint = std::numeric_limits<double>::infinity();

    mutable std::shared_ptr<HistPair> histogram;
    mutable std::vector<int> hist_features;
    mutable bool hist_valid = false;
    mutable bool hist_goss_weighted = false;

    mutable std::vector<int> goss_top_indices_;
    mutable std::vector<int> goss_rest_indices_;
    mutable bool goss_samples_valid_ = false;

    std::unique_ptr<NeuralLeafPredictor> neural_leaf;
    bool has_neural_leaf() const {
        return static_cast<bool>(neural_leaf);
    }
};

using Node = TrainingNode; // compatibility for existing public C++ callers

// Allocation-friendly intermediate model node. PackedTreeBuilder compiles
// these semantics into structure-of-arrays storage for inference.
struct ModelNode {
    int feature = -1;
    int threshold = -1;
    double split_value = std::numeric_limits<double>::quiet_NaN();
    splitx::SplitKind split_kind = splitx::SplitKind::Axis;
    bool missing_left = true;
    int left = -1;
    int right = -1;
    bool is_leaf = true;
    std::vector<double> leaf_values;
};

struct HistPair {
    // G and H are flattened: G[class * total_hist_size_ + feature_offset + bin]
    // For scalar (K=1): G[class*H + f*b] == G[f*b]
    // For multiclass (K>1): G[c*T + f*b] where T = total_hist_size_
    std::vector<double> G, H;
    std::vector<int> C;         // counts are shared across classes (size total_hist_size_)
    int K = 1;                  // number of classes (1 = scalar, >1 = multiclass)
    size_t total_hist_size = 0; // total bins across all features

    bool goss_weighted = false;

    void resize(size_t hist_size, int K_ = 1) {
        total_hist_size = hist_size;
        K = K_;
        G.resize(static_cast<size_t>(hist_size) * static_cast<size_t>(K), 0.0);
        H.resize(static_cast<size_t>(hist_size) * static_cast<size_t>(K), 0.0);
        C.resize(hist_size, 0);
    }

    void clear() {
        std::ranges::fill(G, 0.0);
        std::ranges::fill(H, 0.0);
        std::ranges::fill(C, 0);
        goss_weighted = false;
    }

    void subtract(const HistPair& o) {
        for (size_t i = 0; i < G.size(); ++i) {
            G[i] -= o.G[i];
            H[i] -= o.H[i];
        }
        for (size_t i = 0; i < C.size(); ++i) {
            C[i] -= o.C[i];
        }
    }

    // Access per-class per-bin gradient/hessian
    inline double& g(size_t class_idx, size_t bin) {
        return G[class_idx * total_hist_size + bin];
    }
    inline double& h(size_t class_idx, size_t bin) {
        return H[class_idx * total_hist_size + bin];
    }
    inline const double& g(size_t class_idx, size_t bin) const {
        return G[class_idx * total_hist_size + bin];
    }
    inline const double& h(size_t class_idx, size_t bin) const {
        return H[class_idx * total_hist_size + bin];
    }
};

class HistogramPool {
private:
    std::vector<std::unique_ptr<HistPair>> pool_;
    size_t hist_size_;
    int K_;
    size_t max_pool_size_;
    std::mutex mutex_;

public:
    explicit HistogramPool(size_t size, int K, size_t max_size = 100)
        : hist_size_(size), K_(K), max_pool_size_(max_size) {}

    std::unique_ptr<HistPair> get() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pool_.empty()) {
            auto hp = std::make_unique<HistPair>();
            hp->resize(hist_size_, K_);
            return hp;
        }
        auto hp = std::move(pool_.back());
        pool_.pop_back();
        hp->clear();
        return hp;
    }

    void return_histogram(std::unique_ptr<HistPair> hp) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!hp || static_cast<int>(hp->G.size()) != static_cast<int>(hist_size_ * K_))
            return;
        if (pool_.size() >= max_pool_size_)
            return;
        pool_.push_back(std::move(hp));
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.clear();
    }
};

} // namespace foretree

enum class ClassWeight { Auto = 0, Balanced = 1, None = 2 };
