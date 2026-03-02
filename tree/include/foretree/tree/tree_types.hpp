#pragma once

#include "foretree/tree/neural.hpp"
#include "foretree/split/split_engine.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <ranges>
#include <vector>

namespace foretree {

struct TreeConfig {
    int    max_depth                                 = 6;
    int    max_leaves                                = 31;
    int    min_samples_split                         = 10;
    int    min_samples_leaf                          = 5;
    double min_child_weight                          = 1e-3;
    double lambda_                                   = 1.0;
    double alpha_                                    = 0.0;
    double gamma_                                    = 0.0;
    double max_delta_step                            = 0.0;
    int    n_bins                                    = 256;
    enum class Growth { LeafWise, LevelWise, Oblivious } growth = Growth::LeafWise;
    double leaf_gain_eps                             = 0.0;
    bool   allow_zero_gain                           = false;
    double leaf_depth_penalty                        = 0.0;
    double leaf_hess_boost                           = 0.0;
    int    feature_bagging_k                         = -1;
    bool   feature_bagging_with_replacement          = false;
    int    colsample_bytree_percent                  = 100;
    int    colsample_bylevel_percent                 = 100;
    int    colsample_bynode_percent                  = 100;
    bool   use_sibling_subtract                      = true;

    enum class MissingPolicy { Learn, AlwaysLeft, AlwaysRight };
    MissingPolicy       missing_policy = MissingPolicy::Learn;
    std::vector<int8_t> monotone_constraints;

    enum class SplitMode { Histogram, Exact, Hybrid };
    SplitMode split_mode          = SplitMode::Histogram;
    int       exact_cutover       = 2048;
    bool      enable_kway_splits  = false;
    int       kway_max_groups     = 8;
    bool      enable_oblique_splits = false;
    ObliqueMode oblique_mode        = ObliqueMode::Full;
    int       oblique_k_features    = 4;
    int       oblique_newton_steps  = 1;
    double    oblique_l1            = 0.0;
    double    oblique_ridge         = 1e-3;
    double    axis_vs_oblique_guard = 1.02;
    InteractionSeededConfig interaction_seeded_oblique{};

    double subsample_bytree           = 1.0;
    double subsample_bylevel          = 1.0;
    double subsample_bynode           = 1.0;
    bool   subsample_with_replacement = true;
    bool   subsample_importance_scale = false;

    int rng_seed = 123456789;

    struct GOSS {
        bool   enabled         = false;
        double top_rate        = 0.2;
        double other_rate      = 0.1;
        bool   scale_hessian   = true;
        int    min_node_size   = 2000;
        bool   use_random_rest = true;
        bool   adaptive        = true;
        double adaptive_scale  = 1.0;
    } goss;

    bool cache_histograms        = true;
    int  cache_threshold         = 1000;
    int  max_histogram_pool_size = 100;

    struct OnTree {
        bool   enabled               = false;
        double ccp_alpha             = 0.0;
        double min_gain              = 0.0;
        double min_gain_rel          = 0.0;
        double min_impurity_decrease = 0.0;
        double eps                   = 1e-12;
    } on_tree;

    bool   sgld_enabled     = false;
    double sgld_noise_scale = 1.0;

    struct NeuralLeaf {
        bool   enabled              = false;
        int    min_samples          = 20;
        int    max_depth_start      = 2;
        double complexity_threshold = 1e-3;
    } neural_leaf;
};

struct Node {
    int  id      = -1;
    bool is_leaf = true;
    int  depth   = 0;

    int  feature   = -1;
    int  thr       = -1;
    double split_value = std::numeric_limits<double>::quiet_NaN();
    bool miss_left = true;
    splitx::SplitKind split_kind = splitx::SplitKind::Axis;
    std::vector<int>  left_groups;
    std::vector<int>     oblique_features;
    std::vector<double>  oblique_weights;
    double               oblique_threshold    = std::numeric_limits<double>::quiet_NaN();
    bool                 oblique_missing_left = true;
    int  left = -1, right = -1, sibling = -1;

    double G = 0.0, H = 0.0;
    int    C  = 0;
    int    lo = 0, hi = 0;

    double best_gain       = -1e300;
    double leaf_value      = 0.0;
    double goss_weighted_G = 0.0;
    double goss_weighted_H = 0.0;
    double goss_rest_scale = 1.0;

    bool   uses_goss       = false;

    double min_constraint = -std::numeric_limits<double>::infinity();
    double max_constraint = std::numeric_limits<double>::infinity();

    mutable std::vector<double> hist_G, hist_H;
    mutable std::vector<int>    hist_C;
    mutable std::vector<int>    hist_features;
    mutable bool                hist_valid         = false;
    mutable bool                hist_goss_weighted = false;

    mutable std::vector<int> goss_top_indices_;
    mutable std::vector<int> goss_rest_indices_;
    mutable bool             goss_samples_valid_ = false;

    std::unique_ptr<NeuralLeafPredictor> neural_leaf;
    bool                                 has_neural_leaf() const { return static_cast<bool>(neural_leaf); }
};

struct HistPair {
    std::vector<double> G, H;
    std::vector<int>    C;
    bool                goss_weighted = false;

    void resize(size_t size) {
        G.resize(size);
        H.resize(size);
        C.resize(size);
    }

    void clear() {
        std::ranges::fill(G, 0.0);
        std::ranges::fill(H, 0.0);
        std::ranges::fill(C, 0);
        goss_weighted = false;
    }

    void subtract(const HistPair &o) {
        for (size_t i = 0; i < G.size(); ++i) {
            G[i] -= o.G[i];
            H[i] -= o.H[i];
        }
        for (size_t i = 0; i < C.size(); ++i) {
            C[i] -= o.C[i];
        }
    }
};

class HistogramPool {
private:
    std::vector<std::unique_ptr<HistPair>> pool_;
    std::queue<size_t>                     available_;
    size_t                                 hist_size_;
    size_t                                 max_pool_size_;
    std::mutex                             mutex_;

public:
    explicit HistogramPool(size_t size, size_t max_size = 100) : hist_size_(size), max_pool_size_(max_size) {}

    std::unique_ptr<HistPair> get() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (available_.empty()) {
            auto hp = std::make_unique<HistPair>();
            hp->resize(hist_size_);
            return hp;
        }
        const auto idx = available_.front();
        available_.pop();
        auto hp = std::move(pool_[idx]);
        hp->clear();
        return hp;
    }

    void return_histogram(std::unique_ptr<HistPair> hp) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!hp || hp->G.size() != hist_size_) return;
        if (pool_.size() >= max_pool_size_) return;
        const size_t idx = pool_.size();
        pool_.push_back(std::move(hp));
        available_.push(idx);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.clear();
        while (!available_.empty()) available_.pop();
    }
};

} // namespace foretree
