#include <vector>
#include <memory>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <array>
#include <span>
#include <optional>
#include <concepts>
#include <unordered_set>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
//#include <pybind11/eigen.h>

namespace py = pybind11;

// Forward declarations
template<typename T>
concept FloatingPoint = std::floating_point<T>;

template<typename T>
concept SignedInteger = std::signed_integral<T>;

// Configuration constants
namespace config {
    constexpr int MAX_DEPTH_GUARD = 100;
    constexpr double EPSILON = 1e-12;
    constexpr int DEFAULT_N_BINS = 256;
    constexpr int DEFAULT_MAX_LEAVES = 31;
}

// Tree method enumeration
enum class TreeMethod : uint8_t {
    HIST = 0,
    EXACT = 1, 
    APPROX = 2
};

enum class GrowthPolicy : uint8_t {
    LEAF_WISE = 0,
    LEVEL_WISE = 1
};

// Template-based node structure
template<FloatingPoint Float = double, SignedInteger Int = int32_t>
struct TreeNode {
    using float_type = Float;
    using int_type = Int;
    using index_vector = std::vector<int_type>;
    using float_vector = std::vector<float_type>;
    
    // Core node data
    int_type node_id;
    index_vector data_indices;
    float_vector gradients;
    float_vector hessians;
    int_type depth;
    
    // Computed values
    int_type n_samples = 0;
    float_type g_sum = 0.0;
    float_type h_sum = 0.0;
    
    // Split information
    std::optional<int_type> best_feature = std::nullopt;
    float_type best_threshold = std::numeric_limits<float_type>::quiet_NaN();
    float_type best_gain = -std::numeric_limits<float_type>::infinity();
    std::optional<int_type> best_bin_idx = std::nullopt;
    bool missing_go_left = false;
    
    // Tree structure
    std::unique_ptr<TreeNode> left_child = nullptr;
    std::unique_ptr<TreeNode> right_child = nullptr;
    std::optional<float_type> leaf_value = std::nullopt;
    bool is_leaf = true;
    
    // Pruning data
    int_type prune_leaves = 0;
    int_type prune_internal = 0;
    float_type prune_R_subtree = 0.0;
    float_type prune_R_collapse = 0.0;
    float_type prune_alpha_star = std::numeric_limits<float_type>::infinity();
    
    // Exact method sorted lists (one per feature)
    std::vector<index_vector> sorted_lists;
    
    // Histogram data
    std::optional<std::pair<float_vector, float_vector>> histograms = std::nullopt;
    std::optional<int_type> sibling_node_id = std::nullopt;
    
    // Constructor
    TreeNode(int_type id, index_vector&& indices, float_vector&& grad, float_vector&& hess, int_type d)
        : node_id(id), data_indices(std::move(indices)), gradients(std::move(grad)), 
          hessians(std::move(hess)), depth(d) {
        init_sums();
    }
    
    void init_sums() {
        n_samples = static_cast<int_type>(data_indices.size());
        g_sum = std::accumulate(gradients.begin(), gradients.end(), float_type{0.0});
        h_sum = std::accumulate(hessians.begin(), hessians.end(), float_type{0.0});
    }
};

// Template-based utility functions
template<FloatingPoint Float>
constexpr Float calc_leaf_value_newton(Float g_sum, Float h_sum, Float lambda_reg, 
                                     Float alpha_reg = 0.0, Float max_delta_step = 0.0) {
    if (h_sum <= 0.0) return 0.0;
    
    Float raw_value = -g_sum / (h_sum + lambda_reg);
    
    // L1 regularization (soft thresholding)
    if (alpha_reg > 0.0) {
        if (raw_value > alpha_reg / (h_sum + lambda_reg)) {
            raw_value -= alpha_reg / (h_sum + lambda_reg);
        } else if (raw_value < -alpha_reg / (h_sum + lambda_reg)) {
            raw_value += alpha_reg / (h_sum + lambda_reg);
        } else {
            raw_value = 0.0;
        }
    }
    
    // Max delta step clipping
    if (max_delta_step > 0.0) {
        raw_value = std::clamp(raw_value, -max_delta_step, max_delta_step);
    }
    
    return raw_value;
}

template<FloatingPoint Float>
constexpr Float calc_split_gain(Float g_left, Float h_left, Float g_right, Float h_right,
                              Float lambda_reg, Float gamma = 0.0) {
    if (h_left <= 0.0 || h_right <= 0.0) return -std::numeric_limits<Float>::infinity();
    
    Float gain_left = (g_left * g_left) / (h_left + lambda_reg);
    Float gain_right = (g_right * g_right) / (h_right + lambda_reg);
    Float gain_parent = ((g_left + g_right) * (g_left + g_right)) / (h_left + h_right + lambda_reg);
    
    return 0.5 * (gain_left + gain_right - gain_parent) - gamma;
}

// Templated histogram computation
template<FloatingPoint Float, SignedInteger Int>
std::pair<std::vector<Float>, std::vector<Float>> 
compute_histogram_single_feature(const std::vector<Int>& bin_indices,
                               const std::vector<Float>& gradients,
                               const std::vector<Float>& hessians,
                               Int n_bins) {
    std::vector<Float> hist_g(n_bins, 0.0);
    std::vector<Float> hist_h(n_bins, 0.0);
    
    for (size_t i = 0; i < bin_indices.size(); ++i) {
        Int bin = bin_indices[i];
        if (bin >= 0 && bin < n_bins) {
            hist_g[bin] += gradients[i];
            hist_h[bin] += hessians[i];
        }
    }
    
    return {std::move(hist_g), std::move(hist_h)};
}

// Templated split finding
template<FloatingPoint Float, SignedInteger Int>
struct SplitResult {
    Float gain;
    Int split_bin;
    bool missing_left;
    Float threshold;
};

template<FloatingPoint Float, SignedInteger Int>
SplitResult<Float, Int> find_best_split_with_missing(
    const std::vector<Float>& hist_g,
    const std::vector<Float>& hist_h,
    Float g_missing,
    Float h_missing,
    Float lambda_reg,
    Float gamma,
    Float min_child_weight) {
    
    SplitResult<Float, Int> result{-std::numeric_limits<Float>::infinity(), -1, false, 
                                   std::numeric_limits<Float>::quiet_NaN()};
    
    Int n_bins = static_cast<Int>(hist_g.size());
    if (n_bins < 2) return result;
    
    // Precompute cumulative sums for efficiency
    std::vector<Float> cum_g(n_bins + 1, 0.0);
    std::vector<Float> cum_h(n_bins + 1, 0.0);
    
    for (Int i = 0; i < n_bins; ++i) {
        cum_g[i + 1] = cum_g[i] + hist_g[i];
        cum_h[i + 1] = cum_h[i] + hist_h[i];
    }
    
    Float total_g = cum_g[n_bins];
    Float total_h = cum_h[n_bins];
    
    // Try splits at bin boundaries
    for (Int split_bin = 0; split_bin < n_bins - 1; ++split_bin) {
        Float g_left_finite = cum_g[split_bin + 1];
        Float h_left_finite = cum_h[split_bin + 1];
        Float g_right_finite = total_g - g_left_finite;
        Float h_right_finite = total_h - h_left_finite;
        
        // Try both missing directions
        std::array<bool, 2> missing_directions = {false, true};
        
        for (bool missing_left : missing_directions) {
            Float g_left = g_left_finite + (missing_left ? g_missing : 0.0);
            Float h_left = h_left_finite + (missing_left ? h_missing : 0.0);
            Float g_right = g_right_finite + (missing_left ? 0.0 : g_missing);
            Float h_right = h_right_finite + (missing_left ? 0.0 : h_missing);
            
            if (h_left < min_child_weight || h_right < min_child_weight) {
                continue;
            }
            
            Float gain = calc_split_gain(g_left, h_left, g_right, h_right, lambda_reg, gamma);
            
            if (gain > result.gain) {
                result.gain = gain;
                result.split_bin = split_bin;
                result.missing_left = missing_left;
            }
        }
    }
    
    return result;
}

// Main templated tree class
template<FloatingPoint Float = double, SignedInteger Int = int32_t>
class UnifiedTree {
public:
    using float_type = Float;
    using int_type = Int;
    using Node = TreeNode<Float, Int>;
    using Matrix = py::array_t<Float>;
    using IntVector = std::vector<Int>;
    using FloatVector = std::vector<Float>;
    
private:
    // Configuration
    GrowthPolicy growth_policy_;
    TreeMethod tree_method_;
    int_type max_depth_;
    int_type max_leaves_;
    int_type min_samples_split_;
    int_type min_samples_leaf_;
    float_type min_child_weight_;
    float_type lambda_reg_;
    float_type gamma_;
    float_type alpha_reg_;
    float_type max_delta_step_;
    int_type n_bins_;
    
    // Feature handling
    IntVector feature_indices_;
    std::unordered_map<int_type, int_type> feature_map_;
    std::vector<FloatVector> bin_edges_;
    std::unordered_map<int_type, int_type> monotone_constraints_;
    
    // Tree structure
    std::unique_ptr<Node> root_;
    std::unordered_map<int_type, Node*> nodes_;
    int_type next_node_id_ = 0;
    
    // Prediction arrays for fast inference
    struct PredictionArrays {
        IntVector node_features;
        FloatVector node_thresholds;
        std::vector<bool> node_missing_go_left;
        IntVector left_children;
        IntVector right_children;
        FloatVector leaf_values;
        std::vector<bool> is_leaf_flags;
        IntVector feature_map_array;
    };
    std::optional<PredictionArrays> pred_arrays_;
    
    // Exact method data
    FloatVector g_global_;
    FloatVector h_global_;
    std::unordered_map<int_type, IntVector> sorted_indices_;
    std::unordered_map<int_type, IntVector> missing_indices_;
    
    // Approx method quantile edges
    std::unordered_map<int_type, FloatVector> approx_edges_;
    
    int_type new_node_id() { return next_node_id_++; }
    
    // Template-based evaluation strategy with constexpr dispatch
    template<TreeMethod Method>
    struct EvaluationStrategy {
        static constexpr bool has_presorted_data = (Method == TreeMethod::EXACT);
        static constexpr bool uses_quantile_binning = (Method == TreeMethod::APPROX);
        static constexpr bool uses_histogram = (Method == TreeMethod::HIST) || (Method == TreeMethod::APPROX);
    };
    
    // Unified template method for all evaluation strategies
    template<TreeMethod Method>
    bool evaluate_split(const Matrix& X, Node& node) {
        // Common stopping criteria
        if (!should_split_node(node)) {
            return false;
        }
        
        constexpr auto strategy = EvaluationStrategy<Method>{};
        
        float_type best_gain = -std::numeric_limits<float_type>::infinity();
        int_type best_feature = -1;
        float_type best_threshold = std::numeric_limits<float_type>::quiet_NaN();
        bool best_missing_left = false;
        
        // Feature iteration with method-specific data extraction
        for (size_t fi = 0; fi < feature_indices_.size(); ++fi) {
            int_type global_feat = feature_indices_[fi];
            
            auto split_result = [&]() {
                if constexpr (Method == TreeMethod::EXACT) {
                    return evaluate_feature_exact<Method>(X, node, global_feat, fi);
                } else if constexpr (Method == TreeMethod::HIST) {
                    return evaluate_feature_histogram<Method>(X, node, global_feat, fi);
                } else if constexpr (Method == TreeMethod::APPROX) {
                    return evaluate_feature_approx<Method>(X, node, global_feat, fi);
                } else {
                    static_assert(Method == TreeMethod::HIST || Method == TreeMethod::EXACT || Method == TreeMethod::APPROX);
                    return SplitResult<float_type, int_type>{-std::numeric_limits<float_type>::infinity(), -1, false, 
                                                           std::numeric_limits<float_type>::quiet_NaN()};
                }
            }();
            
            if (split_result.gain > best_gain) {
                best_gain = split_result.gain;
                best_feature = global_feat;
                best_threshold = split_result.threshold;
                best_missing_left = split_result.missing_left;
            }
        }
        
        if (best_feature == -1 || best_gain <= 0.0) {
            return false;
        }
        
        // Store results
        node.best_feature = best_feature;
        node.best_threshold = best_threshold;
        node.best_gain = best_gain;
        node.missing_go_left = best_missing_left;
        
        return true;
    }
    
    // Unified stopping criteria check
    constexpr bool should_split_node(const Node& node) const {
        return node.n_samples >= min_samples_split_ && 
               node.depth < max_depth_ && 
               node.h_sum >= min_child_weight_;
    }
    
    // Template specializations for different methods
    template<TreeMethod Method>
    SplitResult<float_type, int_type> evaluate_feature_histogram(
        const Matrix& X, const Node& node, int_type global_feat, size_t feature_idx) {
        
        static_assert(Method == TreeMethod::HIST || Method == TreeMethod::APPROX);
        
        int_type local_feat = feature_map_.at(global_feat);
        if (local_feat >= X.shape(1)) {
            return create_invalid_split_result();
        }
        
        // Extract feature data for this node
        auto [feature_values, node_gradients, node_hessians] = 
            extract_node_feature_data(X, node, local_feat);
        
        if (feature_values.empty()) {
            return create_invalid_split_result();
        }
        
        // Get binning edges based on method
        FloatVector edges;
        if constexpr (Method == TreeMethod::HIST) {
            edges = get_histogram_edges(global_feat);
        } else if constexpr (Method == TreeMethod::APPROX) {
            edges = get_quantile_edges(feature_values, node_gradients, node_hessians);
        }
        
        if (edges.size() < 2) {
            return create_invalid_split_result();
        }
        
        return find_best_split_on_bins(feature_values, node_gradients, node_hessians, edges);
    }
    
    template<TreeMethod Method>
    SplitResult<float_type, int_type> evaluate_feature_exact(
        const Matrix& X, const Node& node, int_type global_feat, size_t feature_idx) {
        
        static_assert(Method == TreeMethod::EXACT);
        
        int_type local_feat = feature_map_.at(global_feat);
        if (local_feat >= X.shape(1) || node.sorted_lists.empty() || 
            feature_idx >= node.sorted_lists.size()) {
            return create_invalid_split_result();
        }
        
        // Use presorted data for exact evaluation
        const auto& sorted_indices = node.sorted_lists[feature_idx];
        if (sorted_indices.empty()) {
            return create_invalid_split_result();
        }
        
        return find_best_split_exact(X, sorted_indices, local_feat, global_feat);
    }
    
    template<TreeMethod Method>
    SplitResult<float_type, int_type> evaluate_feature_approx(
        const Matrix& X, const Node& node, int_type global_feat, size_t feature_idx) {
        
        static_assert(Method == TreeMethod::APPROX);
        
        // Delegate to histogram method with quantile-based binning
        return evaluate_feature_histogram<Method>(X, node, global_feat, feature_idx);
    }
    
private:
    // Helper methods with clear single responsibilities
    
    std::tuple<FloatVector, FloatVector, FloatVector> 
    extract_node_feature_data(const Matrix& X, const Node& node, int_type local_feat) const {
        auto X_ptr = static_cast<const float_type*>(X.data());
        int_type n_features = X.shape(1);
        
        FloatVector feature_values, gradients, hessians;
        feature_values.reserve(node.n_samples);
        gradients.reserve(node.n_samples);
        hessians.reserve(node.n_samples);
        
        for (int_type i = 0; i < node.n_samples; ++i) {
            int_type sample_idx = node.data_indices[i];
            feature_values.push_back(X_ptr[sample_idx * n_features + local_feat]);
            gradients.push_back(node.gradients[i]);
            hessians.push_back(node.hessians[i]);
        }
        
        return {std::move(feature_values), std::move(gradients), std::move(hessians)};
    }
    
    FloatVector get_histogram_edges(int_type global_feat) const {
        if (global_feat >= static_cast<int_type>(bin_edges_.size()) || 
            bin_edges_[global_feat].empty()) {
            return {};
        }
        return bin_edges_[global_feat];
    }
    
    FloatVector get_quantile_edges(const FloatVector& values, 
                                  const FloatVector& gradients,
                                  const FloatVector& hessians) const {
        // Use weighted quantiles with hessians as weights (XGBoost approx style)
        if (values.empty()) return {};
        
        // Filter out non-finite values
        FloatVector finite_values, finite_weights;
        for (size_t i = 0; i < values.size(); ++i) {
            if (std::isfinite(values[i])) {
                finite_values.push_back(values[i]);
                finite_weights.push_back(hessians[i] + config::EPSILON); // Add small epsilon for stability
            }
        }
        
        if (finite_values.empty()) return {};
        if (finite_values.size() == 1) return {finite_values[0], finite_values[0]};
        
        // Use equally spaced quantiles for approximation
        std::vector<float_type> quantiles;
        for (int_type i = 0; i <= n_bins_; ++i) {
            quantiles.push_back(static_cast<float_type>(i) / n_bins_);
        }
        
        auto edges = compute_weighted_quantiles(finite_values, finite_weights, quantiles);
        
        // Ensure edges are strictly increasing
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        
        if (edges.size() < 2) {
            auto [min_val, max_val] = std::minmax_element(finite_values.begin(), finite_values.end());
            return {*min_val, *max_val};
        }
        
        return edges;
    }
    
    // Compute weighted quantiles using linear interpolation
    FloatVector compute_weighted_quantiles(
        const FloatVector& values,
        const FloatVector& weights,
        const std::vector<float_type>& quantiles) const {
        
        if (values.empty() || weights.empty() || values.size() != weights.size()) {
            return FloatVector(quantiles.size(), 0.0);
        }

        // Create sorted indices by value
        std::vector<size_t> indices(values.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        std::sort(indices.begin(), indices.end(), 
                  [&values](size_t a, size_t b) { return values[a] < values[b]; });

        // Compute cumulative weights
        FloatVector cum_weights(values.size());
        cum_weights[0] = weights[indices[0]];
        for (size_t i = 1; i < values.size(); ++i) {
            cum_weights[i] = cum_weights[i-1] + weights[indices[i]];
        }

        float_type total_weight = cum_weights.back();
        FloatVector result;
        result.reserve(quantiles.size());

        for (float_type q : quantiles) {
            float_type target_weight = q * total_weight;
            
            // Find position using binary search
            auto it = std::lower_bound(cum_weights.begin(), cum_weights.end(), target_weight);
            size_t pos = std::distance(cum_weights.begin(), it);
            
            if (pos >= values.size()) {
                result.push_back(values[indices.back()]);
            } else if (pos == 0) {
                result.push_back(values[indices[0]]);
            } else {
                // Linear interpolation between adjacent values
                float_type w1 = cum_weights[pos-1];
                float_type w2 = cum_weights[pos];
                float_type v1 = values[indices[pos-1]];
                float_type v2 = values[indices[pos]];
                
                float_type alpha = (target_weight - w1) / (w2 - w1);
                result.push_back(v1 + alpha * (v2 - v1));
            }
        }

        return result;
    }
    
    SplitResult<float_type, int_type> find_best_split_on_bins(
        const FloatVector& values, const FloatVector& gradients, 
        const FloatVector& hessians, const FloatVector& edges) const {
        
        int_type n_bins = static_cast<int_type>(edges.size()) - 1;
        if (n_bins <= 0) {
            return create_invalid_split_result();
        }
        
        // Separate finite and missing values
        IntVector bin_indices;
        FloatVector finite_gradients, finite_hessians;
        float_type g_missing = 0.0, h_missing = 0.0;
        
        bin_indices.reserve(values.size());
        finite_gradients.reserve(values.size());
        finite_hessians.reserve(values.size());
        
        for (size_t i = 0; i < values.size(); ++i) {
            if (std::isfinite(values[i])) {
                // Find appropriate bin
                int_type bin = static_cast<int_type>(
                    std::upper_bound(edges.begin(), edges.end(), values[i]) - edges.begin()) - 1;
                bin = std::clamp(bin, int_type{0}, n_bins - 1);
                
                bin_indices.push_back(bin);
                finite_gradients.push_back(gradients[i]);
                finite_hessians.push_back(hessians[i]);
            } else {
                g_missing += gradients[i];
                h_missing += hessians[i];
            }
        }
        
        if (bin_indices.empty()) {
            return create_invalid_split_result();
        }
        
        // Build histogram
        auto [hist_g, hist_h] = compute_histogram_single_feature(
            bin_indices, finite_gradients, finite_hessians, n_bins);
        
        // Find best split
        auto result = find_best_split_with_missing<float_type, int_type>(
        hist_g, hist_h, g_missing, h_missing, lambda_reg_, gamma_, min_child_weight_);
    

        // Convert bin index to actual threshold
        if (result.split_bin >= 0 && result.split_bin < n_bins - 1) {
            result.threshold = edges[result.split_bin + 1];
        }
        
        return result;
    }
    
    SplitResult<float_type, int_type> find_best_split_exact(
        const Matrix& X, const IntVector& sorted_indices, 
        int_type local_feat, int_type global_feat) const {
        
        if (sorted_indices.size() < 2 * min_samples_leaf_) {
            return create_invalid_split_result();
        }
        
        auto X_ptr = static_cast<const float_type*>(X.data());
        int_type n_features = X.shape(1);
        
        float_type best_gain = -std::numeric_limits<float_type>::infinity();
        float_type best_threshold = std::numeric_limits<float_type>::quiet_NaN();
        bool best_missing_left = false;
        
        // Cumulative statistics for efficient split evaluation
        float_type cum_g = 0.0, cum_h = 0.0;
        float_type total_g = 0.0, total_h = 0.0;
        
        // Calculate total sums
        for (int_type idx : sorted_indices) {
            total_g += g_global_[idx];
            total_h += h_global_[idx];
        }
        
        // Handle missing values if any
        float_type g_missing = 0.0, h_missing = 0.0;
        auto missing_it = missing_indices_.find(global_feat);
        if (missing_it != missing_indices_.end()) {
            const auto& missing_idx = missing_it->second;
            std::unordered_set<int_type> node_samples(sorted_indices.begin(), sorted_indices.end());
            
            for (int_type idx : missing_idx) {
                if (node_samples.count(idx)) {
                    g_missing += g_global_[idx];
                    h_missing += h_global_[idx];
                }
            }
        }
        
        // Try each potential split point
        for (size_t i = 0; i < sorted_indices.size() - 1; ++i) {
            int_type curr_idx = sorted_indices[i];
            int_type next_idx = sorted_indices[i + 1];
            
            cum_g += g_global_[curr_idx];
            cum_h += h_global_[curr_idx];
            
            // Skip if values are the same (no actual split)
            float_type curr_val = X_ptr[curr_idx * n_features + local_feat];
            float_type next_val = X_ptr[next_idx * n_features + local_feat];
            if (curr_val == next_val) continue;
            
            // Check minimum samples constraint
            if (i + 1 < min_samples_leaf_ || 
                sorted_indices.size() - i - 1 < min_samples_leaf_) continue;
            
            // Try both missing directions
            for (bool missing_left : {false, true}) {
                float_type g_left = cum_g + (missing_left ? g_missing : 0.0);
                float_type h_left = cum_h + (missing_left ? h_missing : 0.0);
                float_type g_right = (total_g - cum_g) + (missing_left ? 0.0 : g_missing);
                float_type h_right = (total_h - cum_h) + (missing_left ? 0.0 : h_missing);
                
                if (h_left < min_child_weight_ || h_right < min_child_weight_) continue;
                
                float_type gain = calc_split_gain(g_left, h_left, g_right, h_right, lambda_reg_, gamma_);
                
                if (gain > best_gain) {
                    best_gain = gain;
                    best_threshold = (curr_val + next_val) / 2.0;
                    best_missing_left = missing_left;
                }
            }
        }
        
        return SplitResult<float_type, int_type>{
            best_gain, -1, best_missing_left, best_threshold
        };
    }
    
    constexpr SplitResult<float_type, int_type> create_invalid_split_result() const {
        return SplitResult<float_type, int_type>{
            -std::numeric_limits<float_type>::infinity(), -1, false,
            std::numeric_limits<float_type>::quiet_NaN()
        };
    }
    
    // Initialize exact method data structures
    void prepare_exact_method(const Matrix& X, const FloatVector& gradients, const FloatVector& hessians) {
        int_type n_samples = X.shape(0);
        int_type n_features = X.shape(1);
        
        // Store global gradients and hessians
        g_global_ = gradients;
        h_global_ = hessians;
        
        // Clear existing data
        sorted_indices_.clear();
        missing_indices_.clear();
        
        // Pre-sort indices for each feature
        for (int_type global_feat : feature_indices_) {
            int_type local_feat = feature_map_.at(global_feat);
            if (local_feat >= n_features) continue;
            
            auto X_ptr = static_cast<const float_type*>(X.data());
            
            // Separate finite and missing indices
            IntVector finite_indices, missing_indices;
            for (int_type i = 0; i < n_samples; ++i) {
                float_type value = X_ptr[i * n_features + local_feat];
                if (std::isfinite(value)) {
                    finite_indices.push_back(i);
                } else {
                    missing_indices.push_back(i);
                }
            }
            
            // Sort finite indices by feature value
            std::sort(finite_indices.begin(), finite_indices.end(),
                     [X_ptr, local_feat, n_features](int_type a, int_type b) {
                         return X_ptr[a * n_features + local_feat] < X_ptr[b * n_features + local_feat];
                     });
            
            sorted_indices_[global_feat] = std::move(finite_indices);
            missing_indices_[global_feat] = std::move(missing_indices);
        }
    }
    
    // Template-optimized split application with method awareness
    std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>> 
    apply_split(const Matrix& X, Node& node) {
        if (!node.best_feature.has_value()) {
            return {nullptr, nullptr};
        }

        int_type feature = node.best_feature.value();
        int_type local_feat = feature_map_.at(feature);
        float_type threshold = node.best_threshold;
        bool missing_left = node.missing_go_left;

        // Method-specific split application
        if (tree_method_ == TreeMethod::EXACT && !node.sorted_lists.empty()) {
            return apply_split_exact(X, node, feature, local_feat, threshold, missing_left);
        } else {
            return apply_split_histogram(X, node, local_feat, threshold, missing_left);
        }
    }

private:
    std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>> 
    apply_split_exact(const Matrix& X, Node& node, int_type feature, int_type local_feat, 
                     float_type threshold, bool missing_left) {
        // Use pre-sorted indices for efficient splitting
        auto X_ptr = static_cast<const float_type*>(X.data());
        int_type n_features = X.shape(1);
        
        // Create split masks efficiently using sorted data
        std::vector<bool> sample_goes_left(X.shape(0), false);
        
        // Mark samples that go left based on feature values
        for (int_type sample_idx : node.data_indices) {
            float_type value = X_ptr[sample_idx * n_features + local_feat];
            bool go_left = std::isfinite(value) ? (value <= threshold) : missing_left;
            sample_goes_left[sample_idx] = go_left;
        }
        
        return create_child_nodes_from_mask(node, sample_goes_left);
    }

    std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>> 
    apply_split_histogram(const Matrix& X, Node& node, int_type local_feat, 
                         float_type threshold, bool missing_left) {
        auto X_ptr = static_cast<const float_type*>(X.data());
        int_type n_features = X.shape(1);

        IntVector left_indices, right_indices;
        FloatVector left_gradients, left_hessians;
        FloatVector right_gradients, right_hessians;

        // Reserve space for efficiency
        left_indices.reserve(node.n_samples / 2);
        right_indices.reserve(node.n_samples / 2);
        left_gradients.reserve(node.n_samples / 2);
        right_gradients.reserve(node.n_samples / 2);
        left_hessians.reserve(node.n_samples / 2);
        right_hessians.reserve(node.n_samples / 2);

        for (int_type i = 0; i < node.n_samples; ++i) {
            int_type sample_idx = node.data_indices[i];
            float_type value = X_ptr[sample_idx * n_features + local_feat];
            bool go_left = std::isfinite(value) ? (value <= threshold) : missing_left;

            if (go_left) {
                left_indices.push_back(sample_idx);
                left_gradients.push_back(node.gradients[i]);
                left_hessians.push_back(node.hessians[i]);
            } else {
                right_indices.push_back(sample_idx);
                right_gradients.push_back(node.gradients[i]);
                right_hessians.push_back(node.hessians[i]);
            }
        }

        return create_child_nodes_from_splits(std::move(left_indices), std::move(right_indices),
                                            std::move(left_gradients), std::move(right_gradients),
                                            std::move(left_hessians), std::move(right_hessians),
                                            node.depth + 1);
    }

    std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>> 
    create_child_nodes_from_mask(Node& parent, const std::vector<bool>& goes_left) {
        IntVector left_indices, right_indices;
        FloatVector left_gradients, left_hessians;
        FloatVector right_gradients, right_hessians;

        for (int_type i = 0; i < parent.n_samples; ++i) {
            int_type sample_idx = parent.data_indices[i];
            if (goes_left[sample_idx]) {
                left_indices.push_back(sample_idx);
                left_gradients.push_back(parent.gradients[i]);
                left_hessians.push_back(parent.hessians[i]);
            } else {
                right_indices.push_back(sample_idx);
                right_gradients.push_back(parent.gradients[i]);
                right_hessians.push_back(parent.hessians[i]);
            }
        }

        return create_child_nodes_from_splits(std::move(left_indices), std::move(right_indices),
                                            std::move(left_gradients), std::move(right_gradients),
                                            std::move(left_hessians), std::move(right_hessians),
                                            parent.depth + 1);
    }

    std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>> 
    create_child_nodes_from_splits(IntVector&& left_indices, IntVector&& right_indices,
                                 FloatVector&& left_gradients, FloatVector&& right_gradients,
                                 FloatVector&& left_hessians, FloatVector&& right_hessians,
                                 int_type depth) {
        if (left_indices.size() < static_cast<size_t>(min_samples_leaf_) || 
            right_indices.size() < static_cast<size_t>(min_samples_leaf_)) {
            return {nullptr, nullptr};
        }

        auto left_child = std::make_unique<Node>(new_node_id(), std::move(left_indices),
                                               std::move(left_gradients), std::move(left_hessians), depth);
        auto right_child = std::make_unique<Node>(new_node_id(), std::move(right_indices),
                                                std::move(right_gradients), std::move(right_hessians), depth);

        // Update exact method data structures if needed
        if (tree_method_ == TreeMethod::EXACT) {
            update_sorted_lists_for_children(*left_child, *right_child);
        }

        // Register nodes
        nodes_[left_child->node_id] = left_child.get();
        nodes_[right_child->node_id] = right_child.get();

        return {std::move(left_child), std::move(right_child)};
    }

    void update_sorted_lists_for_children(Node& left_child, Node& right_child) {
        // Update sorted lists for exact method
        size_t n_features_used = feature_indices_.size();
        left_child.sorted_lists.resize(n_features_used);
        right_child.sorted_lists.resize(n_features_used);

        // Create sample membership maps for efficient filtering
        std::unordered_set<int_type> left_samples(left_child.data_indices.begin(), 
                                                 left_child.data_indices.end());

        for (size_t fi = 0; fi < n_features_used; ++fi) {
            int_type global_feat = feature_indices_[fi];
            auto it = sorted_indices_.find(global_feat);
            if (it == sorted_indices_.end()) {
                left_child.sorted_lists[fi] = IntVector{};
                right_child.sorted_lists[fi] = IntVector{};
                continue;
            }
            
            const auto& parent_sorted = it->second;

            IntVector left_sorted, right_sorted;
            left_sorted.reserve(left_child.n_samples);
            right_sorted.reserve(right_child.n_samples);

            for (int_type sample_idx : parent_sorted) {
                if (left_samples.count(sample_idx)) {
                    left_sorted.push_back(sample_idx);
                } else {
                    right_sorted.push_back(sample_idx);
                }
            }

            left_child.sorted_lists[fi] = std::move(left_sorted);
            right_child.sorted_lists[fi] = std::move(right_sorted);
        }
    }

    void finalize_leaf_node(Node& node) {
        node.is_leaf = true;
        node.leaf_value = calc_leaf_value_newton(node.g_sum, node.h_sum,
                                               lambda_reg_, alpha_reg_, max_delta_step_);
        // Clear any split information
        node.best_feature = std::nullopt;
        node.best_threshold = std::numeric_limits<float_type>::quiet_NaN();
        node.best_gain = -std::numeric_limits<float_type>::infinity();
    }

public:
    // Constructor
    UnifiedTree(GrowthPolicy growth_policy = GrowthPolicy::LEAF_WISE,
               TreeMethod tree_method = TreeMethod::HIST,
               int_type max_depth = 6,
               int_type max_leaves = config::DEFAULT_MAX_LEAVES,
               int_type min_samples_split = 10,
               int_type min_samples_leaf = 5,
               float_type min_child_weight = 1e-3,
               float_type lambda_reg = 1.0,
               float_type gamma = 0.0,
               float_type alpha_reg = 0.0,
               float_type max_delta_step = 0.0,
               int_type n_bins = config::DEFAULT_N_BINS)
        : growth_policy_(growth_policy), tree_method_(tree_method), max_depth_(max_depth),
          max_leaves_(max_leaves), min_samples_split_(min_samples_split),
          min_samples_leaf_(min_samples_leaf), min_child_weight_(min_child_weight),
          lambda_reg_(lambda_reg), gamma_(gamma), alpha_reg_(alpha_reg),
          max_delta_step_(max_delta_step), n_bins_(n_bins) {}
    
    void set_feature_indices(const IntVector& indices) {
        feature_indices_ = indices;
        feature_map_.clear();
        for (size_t i = 0; i < indices.size(); ++i) {
            feature_map_[indices[i]] = static_cast<int_type>(i);
        }
    }
    
    void set_bin_edges(const std::vector<FloatVector>& edges) {
        bin_edges_ = edges;
    }
    
    void set_monotone_constraints(const std::unordered_map<int_type, int_type>& constraints) {
        monotone_constraints_ = constraints;
    }
    
    // Main fitting method
    void fit(const Matrix& X, const FloatVector& gradients, const FloatVector& hessians) {
        // Reset state
        nodes_.clear();
        root_.reset();
        next_node_id_ = 0;
        pred_arrays_ = std::nullopt;
        
        // Method-specific preparation
        if (tree_method_ == TreeMethod::EXACT) {
            prepare_exact_method(X, gradients, hessians);
        }
        
        // Dispatch to appropriate growth method using templates
        if (growth_policy_ == GrowthPolicy::LEAF_WISE) {
            grow_tree<GrowthPolicy::LEAF_WISE>(X, gradients, hessians);
        } else {
            grow_tree<GrowthPolicy::LEVEL_WISE>(X, gradients, hessians);
        }

        // Finalize any remaining leaves
        for (auto& [id, node] : nodes_) {
            if (node->is_leaf && !node->leaf_value.has_value()) {
                finalize_leaf_node(*node);
            }
        }

        build_prediction_arrays();
    }
    
    template<GrowthPolicy Policy>
    void grow_tree(const Matrix& X, const FloatVector& gradients, const FloatVector& hessians) {
        if constexpr (Policy == GrowthPolicy::LEAF_WISE) {
            grow_leaf_wise(X, gradients, hessians);
        } else {
            grow_level_wise(X, gradients, hessians);
        }
    }
    
    void grow_leaf_wise(const Matrix& X, const FloatVector& gradients, const FloatVector& hessians) {
        int_type n_samples = X.shape(0);
        IntVector root_indices(n_samples);
        std::iota(root_indices.begin(), root_indices.end(), 0);
        
        root_ = std::make_unique<Node>(new_node_id(), std::move(root_indices),
                                     FloatVector(gradients), FloatVector(hessians), 0);
        nodes_[root_->node_id] = root_.get();
        
        // Initialize sorted lists for exact method
        if (tree_method_ == TreeMethod::EXACT) {
            size_t n_features_used = feature_indices_.size();
            root_->sorted_lists.resize(n_features_used);
            for (size_t fi = 0; fi < n_features_used; ++fi) {
                int_type global_feat = feature_indices_[fi];
                auto it = sorted_indices_.find(global_feat);
                if (it != sorted_indices_.end()) {
                    root_->sorted_lists[fi] = it->second;
                }
            }
        }
        
        using QueueElement = std::tuple<float_type, int_type, Node*>;
        std::priority_queue<QueueElement> queue;
        
        auto can_split = [this, &X](Node& node) {
            switch (tree_method_) {
                case TreeMethod::HIST:
                    return this->template evaluate_split<TreeMethod::HIST>(X, node);
                case TreeMethod::EXACT:
                    return this->template evaluate_split<TreeMethod::EXACT>(X, node);
                case TreeMethod::APPROX:
                    return this->template evaluate_split<TreeMethod::APPROX>(X, node);
                default:
                    return false;
            }
        };
        
        if (can_split(*root_)) {
            queue.emplace(root_->best_gain, root_->node_id, root_.get());
        } else {
            finalize_leaf_node(*root_);
        }
        
        int_type n_leaves = 1;
        while (!queue.empty() && n_leaves < max_leaves_) {
            auto [gain, node_id, node] = queue.top();
            queue.pop();
            
            auto [left_child, right_child] = apply_split(X, *node);
            if (!left_child || !right_child) {
                finalize_leaf_node(*node);
                continue;
            }
            
            node->is_leaf = false;
            node->left_child = std::move(left_child);
            node->right_child = std::move(right_child);
            n_leaves++;
            
            // Evaluate children for further splitting
            for (auto* child : {node->left_child.get(), node->right_child.get()}) {
                if (can_split(*child)) {
                    queue.emplace(child->best_gain, child->node_id, child);
                } else {
                    finalize_leaf_node(*child);
                }
            }
        }
        
        // Finalize remaining nodes in queue as leaves
        while (!queue.empty()) {
            auto [gain, node_id, node] = queue.top();
            queue.pop();
            finalize_leaf_node(*node);
        }
    }
    
    void grow_level_wise(const Matrix& X, const FloatVector& gradients, const FloatVector& hessians) {
        int_type n_samples = X.shape(0);
        IntVector root_indices(n_samples);
        std::iota(root_indices.begin(), root_indices.end(), 0);
        
        root_ = std::make_unique<Node>(new_node_id(), std::move(root_indices),
                                     FloatVector(gradients), FloatVector(hessians), 0);
        nodes_[root_->node_id] = root_.get();
        
        // Initialize sorted lists for exact method
        if (tree_method_ == TreeMethod::EXACT) {
            size_t n_features_used = feature_indices_.size();
            root_->sorted_lists.resize(n_features_used);
            for (size_t fi = 0; fi < n_features_used; ++fi) {
                int_type global_feat = feature_indices_[fi];
                auto it = sorted_indices_.find(global_feat);
                if (it != sorted_indices_.end()) {
                    root_->sorted_lists[fi] = it->second;
                }
            }
        }
        
        // Level-wise growth uses a queue of nodes at the current level
        std::queue<Node*> current_level;
        current_level.push(root_.get());
        
        // Lambda to evaluate splits based on tree method
        auto can_split = [this, &X](Node& node) {
            switch (tree_method_) {
                case TreeMethod::HIST:
                    return this->template evaluate_split<TreeMethod::HIST>(X, node);
                case TreeMethod::EXACT:
                    return this->template evaluate_split<TreeMethod::EXACT>(X, node);
                case TreeMethod::APPROX:
                    return this->template evaluate_split<TreeMethod::APPROX>(X, node);
                default:
                    return false;
            }
        };
        
        int_type current_depth = 0;
        int_type total_leaves = 1;
        
        // Process each level completely before moving to the next
        while (!current_level.empty() && current_depth < max_depth_ && total_leaves < max_leaves_) {
            std::queue<Node*> next_level;
            
            // Process all nodes at current level
            while (!current_level.empty() && total_leaves < max_leaves_) {
                Node* node = current_level.front();
                current_level.pop();
                
                // Try to split this node
                if (can_split(*node)) {
                    auto [left_child, right_child] = apply_split(X, *node);
                    
                    if (left_child && right_child) {
                        // Successfully split - mark node as internal
                        node->is_leaf = false;
                        node->left_child = std::move(left_child);
                        node->right_child = std::move(right_child);
                        
                        // Add children to next level for processing
                        next_level.push(node->left_child.get());
                        next_level.push(node->right_child.get());
                        
                        // We removed one leaf (parent) and added two leaves (children)
                        total_leaves += 1;
                    } else {
                        // Split failed - make this a leaf
                        finalize_leaf_node(*node);
                    }
                } else {
                    // Cannot split - make this a leaf
                    finalize_leaf_node(*node);
                }
            }
            
            // Move to next level
            current_level = std::move(next_level);
            current_depth++;
        }
        
        // Finalize any remaining nodes as leaves
        while (!current_level.empty()) {
            Node* node = current_level.front();
            current_level.pop();
            finalize_leaf_node(*node);
        }
    }
    
    void build_prediction_arrays() {
        if (nodes_.empty()) {
            pred_arrays_ = std::nullopt;
            return;
        }
        
        int_type max_node_id = 0;
        for (const auto& [id, _] : nodes_) {
            max_node_id = std::max(max_node_id, id);
        }
        
        PredictionArrays arrays;
        arrays.node_features.resize(max_node_id + 1, -1);
        arrays.node_thresholds.resize(max_node_id + 1, std::numeric_limits<float_type>::quiet_NaN());
        arrays.node_missing_go_left.resize(max_node_id + 1, false);
        arrays.left_children.resize(max_node_id + 1, -1);
        arrays.right_children.resize(max_node_id + 1, -1);
        arrays.leaf_values.resize(max_node_id + 1, 0.0);
        arrays.is_leaf_flags.resize(max_node_id + 1, true);
        
        // Build feature map array
        int_type max_feature = feature_indices_.empty() ? 0 : 
            *std::max_element(feature_indices_.begin(), feature_indices_.end());
        arrays.feature_map_array.resize(max_feature + 1, -1);
        for (const auto& [global_idx, local_idx] : feature_map_) {
            if (global_idx <= max_feature) {
                arrays.feature_map_array[global_idx] = local_idx;
            }
        }
        
        // Fill node data
        for (const auto& [node_id, node] : nodes_) {
            if (node->is_leaf) {
                arrays.leaf_values[node_id] = node->leaf_value.value_or(0.0);
            } else {
                arrays.node_features[node_id] = node->best_feature.value_or(-1);
                arrays.node_thresholds[node_id] = node->best_threshold;
                arrays.node_missing_go_left[node_id] = node->missing_go_left;
                arrays.left_children[node_id] = node->left_child ? node->left_child->node_id : -1;
                arrays.right_children[node_id] = node->right_child ? node->right_child->node_id : -1;
                arrays.is_leaf_flags[node_id] = false;
            }
        }
        
        pred_arrays_ = std::move(arrays);
    }
    
    FloatVector predict(const Matrix& X) const {
        if (!root_ || !pred_arrays_.has_value()) {
            return FloatVector(X.shape(0), 0.0);
        }
        
        const auto& arrays = pred_arrays_.value();
        auto X_ptr = static_cast<const float_type*>(X.data());
        int_type n_samples = X.shape(0);
        int_type n_features = X.shape(1);
        
        FloatVector predictions(n_samples);
        
        for (int_type i = 0; i < n_samples; ++i) {
            int_type node_id = root_->node_id;
            
            // Traverse tree with depth guard
            for (int depth = 0; depth < config::MAX_DEPTH_GUARD; ++depth) {
                if (node_id < 0 || node_id >= static_cast<int_type>(arrays.is_leaf_flags.size())) {
                    predictions[i] = 0.0;
                    break;
                }
                
                if (arrays.is_leaf_flags[node_id]) {
                    predictions[i] = arrays.leaf_values[node_id];
                    break;
                }
                
                int_type global_feat = arrays.node_features[node_id];
                if (global_feat < 0 || global_feat >= static_cast<int_type>(arrays.feature_map_array.size())) {
                    predictions[i] = 0.0;
                    break;
                }
                
                int_type local_feat = arrays.feature_map_array[global_feat];
                if (local_feat < 0 || local_feat >= n_features) {
                    predictions[i] = 0.0;
                    break;
                }
                
                float_type value = X_ptr[i * n_features + local_feat];
                float_type threshold = arrays.node_thresholds[node_id];
                bool missing_left = arrays.node_missing_go_left[node_id];
                
                bool go_left = std::isfinite(value) ? (value <= threshold) : missing_left;
                node_id = go_left ? arrays.left_children[node_id] : arrays.right_children[node_id];
            }
        }
        
        return predictions;
    }
    
    // Feature importance calculation
    std::unordered_map<int_type, float_type> get_feature_importance() const {
        std::unordered_map<int_type, float_type> importance;
        
        for (const auto& [id, node] : nodes_) {
            if (!node->is_leaf && node->best_feature.has_value() && node->best_gain > 0.0) {
                int_type feature = node->best_feature.value();
                importance[feature] += node->best_gain * node->n_samples;
            }
        }
        
        return importance;
    }
    
    int_type get_depth() const {
        if (!root_) return 0;
        
        int_type max_depth = 0;
        std::function<void(const Node*)> traverse = [&](const Node* node) {
            if (!node) return;
            max_depth = std::max(max_depth, node->depth);
            if (!node->is_leaf) {
                traverse(node->left_child.get());
                traverse(node->right_child.get());
            }
        };
        
        traverse(root_.get());
        return max_depth;
    }
    
    int_type get_n_leaves() const {
        int_type count = 0;
        for (const auto& [id, node] : nodes_) {
            if (node->is_leaf) count++;
        }
        return count;
    }
    
    // Cost-complexity pruning
    void post_prune_ccp(float_type ccp_alpha) {
        if (!root_) return;
        
        // Bottom-up accumulation of subtree statistics
        std::function<void(Node*)> accumulate_stats = [&](Node* node) {
            if (!node) return;
            
            if (node->is_leaf) {
                node->prune_leaves = 1;
                node->prune_internal = 0;
                node->prune_R_subtree = calc_leaf_objective_optimal(node->g_sum, node->h_sum);
                node->prune_R_collapse = node->prune_R_subtree;
                node->prune_alpha_star = std::numeric_limits<float_type>::infinity();
                return;
            }
            
            // Process children first
            accumulate_stats(node->left_child.get());
            accumulate_stats(node->right_child.get());
            
            // Aggregate child statistics
            node->prune_leaves = node->left_child->prune_leaves + node->right_child->prune_leaves;
            node->prune_internal = node->left_child->prune_internal + node->right_child->prune_internal + 1;
            
            // Subtree objective = sum of leaf objectives - gamma * number of splits
            node->prune_R_subtree = node->left_child->prune_R_subtree + 
                                   node->right_child->prune_R_subtree - gamma_;
            
            // Objective if this node becomes a leaf
            node->prune_R_collapse = calc_leaf_objective_optimal(node->g_sum, node->h_sum);
            
            // Weakest link alpha
            float_type denom = std::max(node->prune_leaves - 1, 1);
            node->prune_alpha_star = (node->prune_R_collapse - node->prune_R_subtree) / denom;
        };
        
        accumulate_stats(root_.get());
        
        // Apply pruning (post-order traversal)
        std::function<void(Node*)> apply_pruning = [&](Node* node) {
            if (!node || node->is_leaf) return;
            
            apply_pruning(node->left_child.get());
            apply_pruning(node->right_child.get());
            
            if (node->prune_alpha_star <= ccp_alpha) {
                // Collapse this node
                nodes_.erase(node->left_child->node_id);
                nodes_.erase(node->right_child->node_id);
                
                node->left_child.reset();
                node->right_child.reset();
                node->is_leaf = true;
                node->leaf_value = calc_leaf_value_newton(node->g_sum, node->h_sum,
                                                        lambda_reg_, alpha_reg_, max_delta_step_);
                node->best_feature = std::nullopt;
                node->best_threshold = std::numeric_limits<float_type>::quiet_NaN();
                node->missing_go_left = false;
                node->best_gain = -std::numeric_limits<float_type>::infinity();
            }
        };
        
        apply_pruning(root_.get());
        build_prediction_arrays();
    }

private:

float_type calc_leaf_objective_optimal(float_type g_sum, float_type h_sum) const {
        if (h_sum <= 0.0) return 0.0;
        
        float_type raw_obj = -(g_sum * g_sum) / (h_sum + lambda_reg_);
        
        // Apply L1 regularization effect
        if (alpha_reg_ > 0.0) {
            float_type threshold = alpha_reg_ / (h_sum + lambda_reg_);
            if (std::abs(g_sum / (h_sum + lambda_reg_)) <= threshold) {
                raw_obj = 0.0;
            } else {
                float_type sign = (g_sum > 0.0) ? 1.0 : -1.0;
                float_type adjusted_g = std::abs(g_sum) - alpha_reg_;
                raw_obj = -(adjusted_g * adjusted_g) / ((h_sum + lambda_reg_) * (h_sum + lambda_reg_)) * 
                         (h_sum + lambda_reg_);
            }
        }
        
        return raw_obj;
    }
};

// Convenience type aliases
using UnifiedTreeF = UnifiedTree<float, int32_t>;
using UnifiedTreeD = UnifiedTree<double, int32_t>;

// Compatibility classes
template<FloatingPoint Float = double, SignedInteger Int = int32_t>
class SingleTree : public UnifiedTree<Float, Int> {
public:
    SingleTree(Int max_depth = 6,
              Int min_samples_split = 10, 
              Int min_samples_leaf = 5,
              Float lambda_reg = 1.0,
              Float gamma = 0.0,
              Float alpha_reg = 0.0,
              TreeMethod tree_method = TreeMethod::HIST,
              Int n_bins = config::DEFAULT_N_BINS,
              Float max_delta_step = 0.0)
        : UnifiedTree<Float, Int>(GrowthPolicy::LEVEL_WISE, tree_method, max_depth,
                                 config::DEFAULT_MAX_LEAVES, min_samples_split,
                                 min_samples_leaf, 1e-3, lambda_reg, gamma, alpha_reg,
                                 max_delta_step, n_bins) {}
};

template<FloatingPoint Float = double, SignedInteger Int = int32_t>  
class LeafWiseSingleTree : public UnifiedTree<Float, Int> {
public:
    LeafWiseSingleTree(Int max_depth = 6,
                      Int min_samples_split = 10,
                      Int min_samples_leaf = 5, 
                      Float lambda_reg = 1.0,
                      Float gamma = 0.0,
                      Float alpha_reg = 0.0,
                      TreeMethod tree_method = TreeMethod::HIST,
                      Int n_bins = config::DEFAULT_N_BINS,
                      Float max_delta_step = 0.0,
                      Int max_leaves = config::DEFAULT_MAX_LEAVES,
                      Float min_child_weight = 1e-3)
        : UnifiedTree<Float, Int>(GrowthPolicy::LEAF_WISE, tree_method, max_depth,
                                 max_leaves, min_samples_split, min_samples_leaf,
                                 min_child_weight, lambda_reg, gamma, alpha_reg,
                                 max_delta_step, n_bins) {}
};

// Python bindings
PYBIND11_MODULE(unified_tree, m) {
    m.doc() = "Modern C++ Unified Tree Implementation with Template-based Optimization";
    
    // Enums
    py::enum_<TreeMethod>(m, "TreeMethod")
        .value("HIST", TreeMethod::HIST)
        .value("EXACT", TreeMethod::EXACT)
        .value("APPROX", TreeMethod::APPROX);
    
    py::enum_<GrowthPolicy>(m, "GrowthPolicy")
        .value("LEAF_WISE", GrowthPolicy::LEAF_WISE)
        .value("LEVEL_WISE", GrowthPolicy::LEVEL_WISE);
    
    // Main template classes - Float precision
    py::class_<UnifiedTreeF>(m, "UnifiedTreeF")
        .def(py::init<GrowthPolicy, TreeMethod, int32_t, int32_t, int32_t, int32_t, 
                     float, float, float, float, float, int32_t>(),
             py::arg("growth_policy") = GrowthPolicy::LEAF_WISE,
             py::arg("tree_method") = TreeMethod::HIST,
             py::arg("max_depth") = 6,
             py::arg("max_leaves") = config::DEFAULT_MAX_LEAVES,
             py::arg("min_samples_split") = 10,
             py::arg("min_samples_leaf") = 5,
             py::arg("min_child_weight") = 1e-3f,
             py::arg("lambda_reg") = 1.0f,
             py::arg("gamma") = 0.0f,
             py::arg("alpha_reg") = 0.0f,
             py::arg("max_delta_step") = 0.0f,
             py::arg("n_bins") = config::DEFAULT_N_BINS)
        .def("set_feature_indices", &UnifiedTreeF::set_feature_indices)
        .def("set_bin_edges", &UnifiedTreeF::set_bin_edges)
        .def("set_monotone_constraints", &UnifiedTreeF::set_monotone_constraints)
        .def("fit", &UnifiedTreeF::fit)
        .def("predict", &UnifiedTreeF::predict)
        .def("get_feature_importance", &UnifiedTreeF::get_feature_importance)
        .def("get_depth", &UnifiedTreeF::get_depth)
        .def("get_n_leaves", &UnifiedTreeF::get_n_leaves)
        .def("post_prune_ccp", &UnifiedTreeF::post_prune_ccp);
    
    // Double precision
    py::class_<UnifiedTreeD>(m, "UnifiedTreeD")
        .def(py::init<GrowthPolicy, TreeMethod, int32_t, int32_t, int32_t, int32_t,
                     double, double, double, double, double, int32_t>(),
             py::arg("growth_policy") = GrowthPolicy::LEAF_WISE,
             py::arg("tree_method") = TreeMethod::HIST,
             py::arg("max_depth") = 6, 
             py::arg("max_leaves") = config::DEFAULT_MAX_LEAVES,
             py::arg("min_samples_split") = 10,
             py::arg("min_samples_leaf") = 5,
             py::arg("min_child_weight") = 1e-3,
             py::arg("lambda_reg") = 1.0,
             py::arg("gamma") = 0.0,
             py::arg("alpha_reg") = 0.0,
             py::arg("max_delta_step") = 0.0,
             py::arg("n_bins") = config::DEFAULT_N_BINS)
        .def("set_feature_indices", &UnifiedTreeD::set_feature_indices)
        .def("set_bin_edges", &UnifiedTreeD::set_bin_edges)
        .def("set_monotone_constraints", &UnifiedTreeD::set_monotone_constraints)
        .def("fit", &UnifiedTreeD::fit)
        .def("predict", &UnifiedTreeD::predict)
        .def("get_feature_importance", &UnifiedTreeD::get_feature_importance)
        .def("get_depth", &UnifiedTreeD::get_depth)
        .def("get_n_leaves", &UnifiedTreeD::get_n_leaves)
        .def("post_prune_ccp", &UnifiedTreeD::post_prune_ccp);
    
    // Compatibility classes
    py::class_<SingleTree<float>>(m, "SingleTreeF")
        .def(py::init<int32_t, int32_t, int32_t, float, float, float, TreeMethod, int32_t, float>(),
             py::arg("max_depth") = 6,
             py::arg("min_samples_split") = 10,
             py::arg("min_samples_leaf") = 5,
             py::arg("lambda_reg") = 1.0f,
             py::arg("gamma") = 0.0f,
             py::arg("alpha_reg") = 0.0f,
             py::arg("tree_method") = TreeMethod::HIST,
             py::arg("n_bins") = config::DEFAULT_N_BINS,
             py::arg("max_delta_step") = 0.0f)
        .def("set_feature_indices", &SingleTree<float>::set_feature_indices)
        .def("set_bin_edges", &SingleTree<float>::set_bin_edges)
        .def("set_monotone_constraints", &SingleTree<float>::set_monotone_constraints)
        .def("fit", &SingleTree<float>::fit)
        .def("predict", &SingleTree<float>::predict)
        .def("get_feature_importance", &SingleTree<float>::get_feature_importance)
        .def("get_depth", &SingleTree<float>::get_depth)
        .def("get_n_leaves", &SingleTree<float>::get_n_leaves)
        .def("post_prune_ccp", &SingleTree<float>::post_prune_ccp);
    
    py::class_<SingleTree<double>>(m, "SingleTreeD")
        .def(py::init<int32_t, int32_t, int32_t, double, double, double, TreeMethod, int32_t, double>(),
             py::arg("max_depth") = 6,
             py::arg("min_samples_split") = 10,
             py::arg("min_samples_leaf") = 5,
             py::arg("lambda_reg") = 1.0,
             py::arg("gamma") = 0.0,
             py::arg("alpha_reg") = 0.0,
             py::arg("tree_method") = TreeMethod::HIST,
             py::arg("n_bins") = config::DEFAULT_N_BINS,
             py::arg("max_delta_step") = 0.0)
        .def("set_feature_indices", &SingleTree<double>::set_feature_indices)
        .def("set_bin_edges", &SingleTree<double>::set_bin_edges)
        .def("set_monotone_constraints", &SingleTree<double>::set_monotone_constraints)
        .def("fit", &SingleTree<double>::fit)
        .def("predict", &SingleTree<double>::predict)
        .def("get_feature_importance", &SingleTree<double>::get_feature_importance)
        .def("get_depth", &SingleTree<double>::get_depth)
        .def("get_n_leaves", &SingleTree<double>::get_n_leaves)
        .def("post_prune_ccp", &SingleTree<double>::post_prune_ccp);
        
    py::class_<LeafWiseSingleTree<float>>(m, "LeafWiseSingleTreeF")
        .def(py::init<int32_t, int32_t, int32_t, float, float, float, TreeMethod, int32_t, float, int32_t, float>(),
             py::arg("max_depth") = 6,
             py::arg("min_samples_split") = 10,
             py::arg("min_samples_leaf") = 5,
             py::arg("lambda_reg") = 1.0f,
             py::arg("gamma") = 0.0f,
             py::arg("alpha_reg") = 0.0f,
             py::arg("tree_method") = TreeMethod::HIST,
             py::arg("n_bins") = config::DEFAULT_N_BINS,
             py::arg("max_delta_step") = 0.0f,
             py::arg("max_leaves") = config::DEFAULT_MAX_LEAVES,
             py::arg("min_child_weight") = 1e-3f)
        .def("set_feature_indices", &LeafWiseSingleTree<float>::set_feature_indices)
        .def("set_bin_edges", &LeafWiseSingleTree<float>::set_bin_edges)
        .def("set_monotone_constraints", &LeafWiseSingleTree<float>::set_monotone_constraints)
        .def("fit", &LeafWiseSingleTree<float>::fit)
        .def("predict", &LeafWiseSingleTree<float>::predict)
        .def("get_feature_importance", &LeafWiseSingleTree<float>::get_feature_importance)
        .def("get_depth", &LeafWiseSingleTree<float>::get_depth)
        .def("get_n_leaves", &LeafWiseSingleTree<float>::get_n_leaves)
        .def("post_prune_ccp", &LeafWiseSingleTree<float>::post_prune_ccp);
        
    py::class_<LeafWiseSingleTree<double>>(m, "LeafWiseSingleTreeD")
        .def(py::init<int32_t, int32_t, int32_t, double, double, double, TreeMethod, int32_t, double, int32_t, double>(),
             py::arg("max_depth") = 6,
             py::arg("min_samples_split") = 10,
             py::arg("min_samples_leaf") = 5,
             py::arg("lambda_reg") = 1.0,
             py::arg("gamma") = 0.0,
             py::arg("alpha_reg") = 0.0,
             py::arg("tree_method") = TreeMethod::HIST,
             py::arg("n_bins") = config::DEFAULT_N_BINS,
             py::arg("max_delta_step") = 0.0,
             py::arg("max_leaves") = config::DEFAULT_MAX_LEAVES,
             py::arg("min_child_weight") = 1e-3)
        .def("set_feature_indices", &LeafWiseSingleTree<double>::set_feature_indices)
        .def("set_bin_edges", &LeafWiseSingleTree<double>::set_bin_edges)
        .def("set_monotone_constraints", &LeafWiseSingleTree<double>::set_monotone_constraints)
        .def("fit", &LeafWiseSingleTree<double>::fit)
        .def("predict", &LeafWiseSingleTree<double>::predict)
        .def("get_feature_importance", &LeafWiseSingleTree<double>::get_feature_importance)
        .def("get_depth", &LeafWiseSingleTree<double>::get_depth)
        .def("get_n_leaves", &LeafWiseSingleTree<double>::get_n_leaves)
        .def("post_prune_ccp", &LeafWiseSingleTree<double>::post_prune_ccp);
    
    // Utility functions
    m.def("calc_leaf_value_newton", &calc_leaf_value_newton<double>,
          "Calculate optimal leaf value using Newton method",
          py::arg("g_sum"), py::arg("h_sum"), py::arg("lambda_reg"), 
          py::arg("alpha_reg") = 0.0, py::arg("max_delta_step") = 0.0);
    m.def("calc_split_gain", &calc_split_gain<double>, 
          "Calculate gain from a potential split",
          py::arg("g_left"), py::arg("h_left"), py::arg("g_right"), py::arg("h_right"),
          py::arg("lambda_reg"), py::arg("gamma") = 0.0);
}