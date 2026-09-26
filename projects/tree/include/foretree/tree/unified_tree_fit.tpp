    void fit_with_row_ids_impl_(const QuantizedDataset& Xb, int N, int P,
                                const std::vector<double>& g,
                                const std::vector<double>& h,
                                const std::vector<int>& root_rows) {
        if (N <= 0 || P <= 0) {
            throw std::invalid_argument(
                "UnifiedTree::fit_with_row_ids: N and P must be positive");
        }
        if (static_cast<int>(g.size()) != N ||
            static_cast<int>(h.size()) != N) {
            throw std::invalid_argument(
                "UnifiedTree::fit_with_row_ids: g/h size must match N");
        }
        if (root_rows.empty()) {
            throw std::invalid_argument(
                "UnifiedTree::fit_with_row_ids: root_rows must not be empty");
        }

        const size_t n_sz = static_cast<size_t>(N);
        const size_t p_sz = static_cast<size_t>(P);
        if (n_sz > std::numeric_limits<size_t>::max() / p_sz) {
            throw std::runtime_error(
                "UnifiedTree::fit_with_row_ids: dataset dimensions overflow");
        }
        const size_t expected_cells = n_sz * p_sz;
        if (Xb.size() != expected_cells) {
            throw std::invalid_argument(
                "UnifiedTree::fit_with_row_ids: Xb size must equal N*P");
        }
        for (int r : root_rows) {
            if (r < 0 || r >= N) {
                throw std::invalid_argument(
                    "UnifiedTree::fit_with_row_ids: root_rows contains "
                    "out-of-range index");
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
        unit_hessian_ = std::all_of(h.begin(), h.end(),
                                    [](double value) { return value == 1.0; });
        N_ = N;
        P_ = P;
        training_context_.bind(Xb, g, h, executor_,
                               std::max(cfg_.num_classes - 1, 1));

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
                n.leaf_values =
                    leaf_values_(GG, HH, n.min_constraint, n.max_constraint);
            }
        }
        pack_();
        cleanup_caching_();
        training_context_.release_dataset();
    }

    std::vector<double> predict_contrib(const std::vector<uint16_t>& Xb, int N,
                                        int P) const {
        return predict_contrib(Xb, N, P, nullptr);
    }

    std::vector<int> tree_shap_split_features_() const {
        std::set<int> seen;
        for (const auto& n : nodes_) {
            if (!n.is_leaf && n.split_kind == splitx::SplitKind::Axis &&
                n.feature >= 0) {
                seen.insert(n.feature);
            }
        }
        return std::vector<int>(seen.begin(), seen.end());
    }

    bool tree_shap_has_repeated_feature_path_(
        int node_id, std::vector<uint8_t>& on_path) const {
        if (node_id < 0 ||
            node_id >= static_cast<int>(packed_tree_.leaf_flags.size()) ||
            packed_tree_.leaf_flags[static_cast<size_t>(node_id)] != 0) {
            return false;
        }

        const int feat = packed_tree_.features[static_cast<size_t>(node_id)];
        if (feat < 0 || feat >= P_) return false;
        if (on_path[static_cast<size_t>(feat)] != 0) return true;

        on_path[static_cast<size_t>(feat)] = 1;
        const bool found =
            tree_shap_has_repeated_feature_path_(
                packed_tree_.left_children[static_cast<size_t>(node_id)],
                on_path) ||
            tree_shap_has_repeated_feature_path_(
                packed_tree_.right_children[static_cast<size_t>(node_id)],
                on_path);
        on_path[static_cast<size_t>(feat)] = 0;
        return found;
    }

    bool tree_shap_has_repeated_feature_() const {
        std::vector<uint8_t> on_path(static_cast<size_t>(P_), 0);
        return tree_shap_has_repeated_feature_path_(root_id_, on_path);
    }

    double tree_expected_value_from_node_(int id) const {
        if (id < 0) return 0.0;
        if (id >= static_cast<int>(packed_tree_.leaf_flags.size())) return 0.0;
        if (packed_tree_.leaf_flags[id] != 0)
            return packed_tree_.leaf_values[static_cast<size_t>(id) * K_];

        const int left_id = packed_tree_.left_children[static_cast<size_t>(id)];
        const int right_id =
            packed_tree_.right_children[static_cast<size_t>(id)];
        const double cover = node_cover_(id);
        const double left_cover = node_cover_(left_id);
        const double right_cover = node_cover_(right_id);
        if (cover <= 0.0) {
            return 0.5 * tree_expected_value_from_node_(left_id) +
                   0.5 * tree_expected_value_from_node_(right_id);
        }
        const double left_frac = left_cover / cover;
        const double right_frac = right_cover / cover;
        return left_frac * tree_expected_value_from_node_(left_id) +
               right_frac * tree_expected_value_from_node_(right_id);
    }

    double tree_expected_value_() const {
        return tree_expected_value_from_node_(root_id_);
    }

    double tree_shap_expected_value_mask_(
        int node_id, const uint16_t* row_binned, int row_idx,
        const PredictRawView& raw_view, const std::vector<int>& feature_to_pos,
        uint64_t mask) const {
        if (node_id < 0) return 0.0;
        if (packed_tree_.leaf_flags[static_cast<size_t>(node_id)] != 0)
            return packed_tree_.leaf_values[static_cast<size_t>(node_id)];

        const int split_feature =
            packed_tree_.features[static_cast<size_t>(node_id)];
        const int feature_pos =
            (split_feature >= 0 &&
             split_feature < static_cast<int>(feature_to_pos.size()))
                ? feature_to_pos[static_cast<size_t>(split_feature)]
                : -1;

        if (feature_pos >= 0 &&
            (mask & (uint64_t{1} << static_cast<uint64_t>(feature_pos))) != 0) {
            const bool go_left = predict_go_left_axis_(
                node_id, row_binned, row_idx, split_feature,
                packed_tree_.thresholds[static_cast<size_t>(node_id)],
                packed_tree_.missing_left[static_cast<size_t>(node_id)] != 0,
                raw_view);
            return tree_shap_expected_value_mask_(
                go_left
                    ? packed_tree_.left_children[static_cast<size_t>(node_id)]
                    : packed_tree_.right_children[static_cast<size_t>(node_id)],
                row_binned, row_idx, raw_view, feature_to_pos, mask);
        }

        const int left_id =
            packed_tree_.left_children[static_cast<size_t>(node_id)];
        const int right_id =
            packed_tree_.right_children[static_cast<size_t>(node_id)];
        const double cover = std::max(1e-12, node_cover_(node_id));
        const double left_fraction = node_cover_(left_id) / cover;
        const double right_fraction = node_cover_(right_id) / cover;
        return left_fraction * tree_shap_expected_value_mask_(
                                   left_id, row_binned, row_idx, raw_view,
                                   feature_to_pos, mask) +
               right_fraction * tree_shap_expected_value_mask_(
                                    right_id, row_binned, row_idx, raw_view,
                                    feature_to_pos, mask);
    }

    void brute_force_tree_shap_row_(const uint16_t* row_binned, int row_idx,
                                    const PredictRawView& raw_view, double* phi,
                                    const std::vector<int>& feature_to_pos,
                                    int M) const {
        if (M <= 0) {
            phi[P_] = tree_expected_value_();
            return;
        }

        const size_t n_masks = size_t{1} << static_cast<size_t>(M);
        std::vector<double> values(n_masks, 0.0);
        for (size_t mask = 0; mask < n_masks; ++mask) {
            values[mask] = tree_shap_expected_value_mask_(
                root_id_, row_binned, row_idx, raw_view, feature_to_pos,
                static_cast<uint64_t>(mask));
        }

        phi[P_] = values[0];
        for (int j = 0; j < P_; ++j) phi[j] = 0.0;

        for (int j = 0; j < P_; ++j) {
            const int pos = feature_to_pos[static_cast<size_t>(j)];
            if (pos < 0) continue;
            const uint64_t bit = uint64_t{1} << static_cast<uint64_t>(pos);
            double contrib = 0.0;
            for (size_t mask = 0; mask < n_masks; ++mask) {
                if ((static_cast<uint64_t>(mask) & bit) != 0) continue;
                const int s =
                    __builtin_popcountll(static_cast<unsigned long long>(mask));
                const double weight = 1.0 / (static_cast<double>(M) *
                                             tree_shap_combination_(M - 1, s));
                contrib += weight * (values[mask | bit] - values[mask]);
            }
            phi[j] = contrib;
        }
    }

    double tree_expected_value_from_node_class_(int id, int cls) const {
        if (id < 0) return 0.0;
        if (id >= static_cast<int>(packed_tree_.leaf_flags.size())) return 0.0;
        if (packed_tree_.leaf_flags[id] != 0)
            return packed_tree_.leaf_values[static_cast<size_t>(id) * K_ + static_cast<size_t>(cls)];

        const int left_id = packed_tree_.left_children[static_cast<size_t>(id)];
        const int right_id =
            packed_tree_.right_children[static_cast<size_t>(id)];
        const double cover = node_cover_(id);
        const double left_cover = node_cover_(left_id);
        const double right_cover = node_cover_(right_id);
        if (cover <= 0.0) {
            return 0.5 * tree_expected_value_from_node_class_(left_id, cls) +
                   0.5 * tree_expected_value_from_node_class_(right_id, cls);
        }
        const double left_frac = left_cover / cover;
        const double right_frac = right_cover / cover;
        return left_frac * tree_expected_value_from_node_class_(left_id, cls) +
               right_frac * tree_expected_value_from_node_class_(right_id, cls);
    }

    double tree_expected_value_class_(int cls) const {
        return tree_expected_value_from_node_class_(root_id_, cls);
    }

    double tree_shap_expected_value_mask_class_(
        int node_id, const uint16_t* row_binned, int row_idx,
        const PredictRawView& raw_view, const std::vector<int>& feature_to_pos,
        uint64_t mask, int cls) const {
        if (node_id < 0) return 0.0;
        if (packed_tree_.leaf_flags[static_cast<size_t>(node_id)] != 0)
            return packed_tree_.leaf_values[static_cast<size_t>(node_id) * K_ + static_cast<size_t>(cls)];

        const int split_feature =
            packed_tree_.features[static_cast<size_t>(node_id)];
        const int feature_pos =
            (split_feature >= 0 &&
             split_feature < static_cast<int>(feature_to_pos.size()))
                ? feature_to_pos[static_cast<size_t>(split_feature)]
                : -1;

        if (feature_pos >= 0 &&
            (mask & (uint64_t{1} << static_cast<uint64_t>(feature_pos))) != 0) {
            const bool go_left = predict_go_left_axis_(
                node_id, row_binned, row_idx, split_feature,
                packed_tree_.thresholds[static_cast<size_t>(node_id)],
                packed_tree_.missing_left[static_cast<size_t>(node_id)] != 0,
                raw_view);
            return tree_shap_expected_value_mask_class_(
                go_left
                    ? packed_tree_.left_children[static_cast<size_t>(node_id)]
                    : packed_tree_.right_children[static_cast<size_t>(node_id)],
                row_binned, row_idx, raw_view, feature_to_pos, mask, cls);
        }

        const int left_id =
            packed_tree_.left_children[static_cast<size_t>(node_id)];
        const int right_id =
            packed_tree_.right_children[static_cast<size_t>(node_id)];
        const double cover = std::max(1e-12, node_cover_(node_id));
        const double left_fraction = node_cover_(left_id) / cover;
        const double right_fraction = node_cover_(right_id) / cover;
        return left_fraction * tree_shap_expected_value_mask_class_(
                                   left_id, row_binned, row_idx, raw_view,
                                   feature_to_pos, mask, cls) +
               right_fraction * tree_shap_expected_value_mask_class_(
                                    right_id, row_binned, row_idx, raw_view,
                                    feature_to_pos, mask, cls);
    }

    std::vector<double> compute_multiclass_tree_shap_(
        std::span<const uint16_t> Xb, int N, int /*P*/, const double* Xraw_opt) const {
        const size_t row_size = static_cast<size_t>(K_) * static_cast<size_t>(P_ + 1);
        std::vector<double> out(static_cast<size_t>(N) * row_size, 0.0);
        std::vector<int> feature_to_pos(static_cast<size_t>(P_), -1);
        const auto split_features = tree_shap_split_features_();
        for (size_t k = 0; k < split_features.size(); ++k) {
            feature_to_pos[static_cast<size_t>(split_features[k])] =
                static_cast<int>(k);
        }

        if (split_features.size() <= kBruteforceTreeShapMaxFeatures) {
            for (int i = 0; i < N; ++i) {
                double* row_out = out.data() + static_cast<size_t>(i) * row_size;
                const uint16_t* row_binned =
                    Xb.data() + static_cast<size_t>(i) * static_cast<size_t>(P_);
                const auto raw_view = resolve_predict_raw_view_(Xraw_opt);
                for (int cls = 0; cls < K_; ++cls) {
                    brute_force_tree_shap_row_class_(
                        row_binned, i, raw_view, row_out, feature_to_pos,
                        static_cast<int>(split_features.size()), cls);
                }
            }
        } else {
            for (int i = 0; i < N; ++i) {
                double* row_out = out.data() + static_cast<size_t>(i) * row_size;
                const uint16_t* row_binned =
                    Xb.data() + static_cast<size_t>(i) * static_cast<size_t>(P_);
                const auto raw_view = resolve_predict_raw_view_(Xraw_opt);
                for (int cls = 0; cls < K_; ++cls) {
                    const double bias = tree_expected_value_class_(cls);
                    row_out[P_ + 1 + cls * (P_ + 1)] = bias;
                    std::vector<PathElement> path(static_cast<size_t>(depth() + 2));
                    tree_shap_recursive_class_(root_id_, row_binned, i, raw_view, row_out, cls,
                                         0, path, 1.0, 1.0, -1);
                }
            }
        }
        return out;
    }

    void post_prune_ccp(double ccp_alpha) {
        if (!packed_)
            throw std::runtime_error(
                "post_prune_ccp can only be called on packed trees");
        if (nodes_.empty() || ccp_alpha <= 0.0) return;

        std::vector<Node*> by_id;
        by_id.reserve(nodes_.size());
        for (auto& n : nodes_) {
            if (static_cast<int>(by_id.size()) <= n.id)
                by_id.resize(n.id + 1, nullptr);
            by_id[n.id] = &n;
        }
        Node* root = by_id[root_id_];
        if (!root) return;

        struct Stats {
            int leaves = 0;
            int internal = 0;
            double R_sub = 0.0;
            double R_collapse = 0.0;
            double alpha_star = std::numeric_limits<double>::infinity();
        };
        std::vector<Stats> S(by_id.size());

        std::function<void(Node*)> acc = [&](Node* nd) {
            if (!nd) return;
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
            dst.alpha_star =
                (dst.R_collapse - dst.R_sub) / static_cast<double>(denom);
        };
        acc(root);

        std::function<void(Node*)> apply = [&](Node* nd) {
            if (!nd || nd->is_leaf) return;
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
                nd->leaf_values = leaf_values_(GG, HH, nd->min_constraint,
                                               nd->max_constraint);
            }
        };
        apply(root);
        pack_();
    }

    inline bool uses_goss_() const {
        return cfg_.goss.enabled &&
               (cfg_.goss.top_rate + cfg_.goss.other_rate < 1.0);
    }

    inline bool should_use_goss_for_node_(const Node& n) const {
        return uses_goss_() && (n.hi - n.lo) >= cfg_.goss.min_node_size;
    }

    inline bool node_uses_goss_(const Node& n) const {
        return should_use_goss_for_node_(n) && n.uses_goss &&
               n.goss_samples_valid_;
    }

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
        cfg_.max_histogram_pool_size =
            std::max(1, cfg_.max_histogram_pool_size);
        cfg_.categorical_max_selected_categories =
            std::max(2, cfg_.categorical_max_selected_categories);
        cfg_.oblique_k_features = std::max(2, cfg_.oblique_k_features);
        cfg_.oblique_ridge = std::max(1e-9, cfg_.oblique_ridge);
        cfg_.axis_vs_oblique_guard = std::max(1.0, cfg_.axis_vs_oblique_guard);
        cfg_.pair_interaction.max_features =
            std::max(2, cfg_.pair_interaction.max_features);
        cfg_.pair_interaction.interaction_bins =
            std::clamp(cfg_.pair_interaction.interaction_bins, 2, 16);
        cfg_.pair_interaction.min_node_rows =
            std::max(2, cfg_.pair_interaction.min_node_rows);
        cfg_.pair_interaction.axis_guard_factor =
            std::max(1.0, cfg_.pair_interaction.axis_guard_factor);
        for (auto& group : cfg_.interaction_constraints) {
            std::erase_if(group, [](int feature) { return feature < 0; });
            std::ranges::sort(group);
            group.erase(std::unique(group.begin(), group.end()), group.end());
        }
        std::erase_if(cfg_.interaction_constraints,
                      [](const auto& group) { return group.empty(); });

        auto& is_cfg = cfg_.interaction_seeded_oblique;
        is_cfg.pairs = std::max(1, is_cfg.pairs);
        is_cfg.max_var_candidates = std::max(2, is_cfg.max_var_candidates);
        is_cfg.max_top_features =
            std::clamp(is_cfg.max_top_features, 2, is_cfg.max_var_candidates);
        is_cfg.first_i_cap = std::clamp(
            is_cfg.first_i_cap, 1, std::max(1, is_cfg.max_top_features - 1));
        is_cfg.second_j_cap =
            std::clamp(is_cfg.second_j_cap, 2, is_cfg.max_top_features);
        is_cfg.ridge = std::max(0.0, is_cfg.ridge);
        is_cfg.axis_guard_factor = std::max(1.0, is_cfg.axis_guard_factor);
    }

    void validate_monotone_constraints_() {
        if (cfg_.monotone_constraints.empty()) return;
        if (cfg_.monotone_constraints.size() != static_cast<size_t>(P_)) {
            throw std::invalid_argument(
                "UnifiedTree::fit: monotone_constraints size must match P");
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
                feature_offsets_[j + 1] =
                    feature_offsets_[j] + ghs_->total_bins(j);
            }
            total_hist_size_ = feature_offsets_[P_];
            bin_centers_flat_.assign(total_hist_size_,
                                     std::numeric_limits<double>::quiet_NaN());
            for (int j = 0; j < P_; ++j) {
                const auto& fb = ghs_->feature_bins(j);
                const int finite = finite_bins_per_feat_[j];
                for (int b = 0; b < finite; ++b) {
                    const size_t off =
                        feature_offsets_[j] + static_cast<size_t>(b);
                    if (static_cast<size_t>(b + 1) < fb.edges.size()) {
                        bin_centers_flat_[off] =
                            0.5 *
                            (fb.edges[(size_t)b] + fb.edges[(size_t)b + 1]);
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
            total_hist_size_ =
                static_cast<size_t>(P_) * static_cast<size_t>(cfg_.n_bins + 1);
            bin_centers_flat_.assign(total_hist_size_,
                                     std::numeric_limits<double>::quiet_NaN());
            for (int j = 0; j < P_; ++j) {
                for (int b = 0; b < cfg_.n_bins; ++b) {
                    const size_t off =
                        feature_offsets_[j] + static_cast<size_t>(b);
                    bin_centers_flat_[off] = static_cast<double>(b) + 0.5;
                }
            }
        }
    }

    void initialize_caching_() {
        if (cfg_.cache_histograms) {
            hist_pool_ = std::make_unique<HistogramPool>(
                total_hist_size_, K_, cfg_.max_histogram_pool_size);
        }
        if (cfg_.cache_histograms && !cfg_.goss.enabled &&
            !tree_subsample_applied_ &&
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
        HistPair* histogram = hist_pool_->get(/*clear=*/false).release();
        clear_histogram_(*histogram);
        return std::shared_ptr<HistPair>(
            histogram, [pool = hist_pool_.get()](HistPair* value) {
                pool->return_histogram(std::unique_ptr<HistPair>(value));
            });
    }

    void clear_histogram_(HistPair& hist) const {
        const int n = static_cast<int>(hist.G.size());
        auto zero = [&](int begin, int end) {
            std::fill(hist.G.begin() + begin, hist.G.begin() + end, 0.0);
            std::fill(hist.H.begin() + begin, hist.H.begin() + end, 0.0);
            if (static_cast<int>(hist.C.size()) >= end)
                std::fill(hist.C.begin() + begin, hist.C.begin() + end, 0);
        };
        zero(0, n);
        if (hist.C.size() != hist.G.size())
            std::ranges::fill(hist.C, 0);
        hist.goss_weighted = false;
    }

    void accumulate_hist_bin_(HistPair& hist, int r, int f) const {
        uint16_t b = code_at_(r, f);
        if (b >= static_cast<uint16_t>(missing_ids_per_feat_[f]))
            b = static_cast<uint16_t>(missing_ids_per_feat_[f]);
        const size_t off = feature_offsets_[f] + static_cast<size_t>(b);

        if (off < hist.G.size()) {
            hist.G[off] += (*g_)[r];
            if (!unit_hessian_) hist.H[off] += (*h_)[r];
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

        // Same path as node histograms: the vectorized feature-major kernel
        // (or the CUDA engine for large enough work).
        HistogramBuilder(*this, index_pool_)
            .build_for_range(0, static_cast<int>(index_pool_.size()),
                             tree_features_, *tree_histogram_);
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

    inline const Node* by_id_(int id) const {
        if (id < 0 || id >= static_cast<int>(id2pos_.size())) return nullptr;
        const int pos = id2pos_[id];
        if (pos < 0 || pos >= static_cast<int>(nodes_.size())) return nullptr;
        return &nodes_[pos];
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

    bool select_goss_rows_(Node& n) {
        n.goss_top_indices_.clear();
        n.goss_rest_indices_.clear();
        n.goss_rest_scale = 1.0;
        n.goss_samples_valid_ = false;

        const int total = n.hi - n.lo;
        if (total <= 0) return false;

        auto [a, b] = compute_goss_rates_(n);
        const int k_top =
            std::clamp(static_cast<int>(std::round(a * total)), 1, total);
        const int k_rest = std::clamp(static_cast<int>(std::round(b * total)),
                                      0, total - k_top);

        std::vector<std::pair<double, int>> ranked;
        ranked.reserve(total);
        std::normal_distribution<double> noise(0.0, cfg_.sgld_noise_scale);
        for (int i = n.lo; i < n.hi; ++i) {
            const int r = index_pool_[i];
            double g = (*g_)[r];
            if (cfg_.sgld_enabled) g += noise(rng_);
            ranked.emplace_back(std::abs(g), r);
        }
        std::ranges::sort(ranked, std::greater<>{},
                          &std::pair<double, int>::first);

        n.goss_top_indices_.reserve(k_top);
        n.goss_rest_indices_.reserve(k_rest);
        for (int i = 0; i < k_top; ++i) {
            n.goss_top_indices_.push_back(
                ranked[static_cast<size_t>(i)].second);
        }

        if (k_rest > 0) {
            if (cfg_.goss.use_random_rest) {
                std::vector<int> rest_pool;
                rest_pool.reserve(total - k_top);
                for (int i = k_top; i < total; ++i) rest_pool.push_back(i);
                std::ranges::shuffle(rest_pool, rng_);
                for (int i = 0; i < k_rest; ++i) {
                    n.goss_rest_indices_.push_back(
                        ranked[static_cast<size_t>(
                                   rest_pool[static_cast<size_t>(i)])]
                            .second);
                }
            } else {
                const int rest_end = std::min(total, k_top + k_rest);
                for (int i = k_top; i < rest_end; ++i) {
                    n.goss_rest_indices_.push_back(
                        ranked[static_cast<size_t>(i)].second);
                }
            }
        }

        n.goss_rest_scale = (1.0 - a) / std::max(b, 1e-15);
        n.goss_samples_valid_ = true;
        return true;
    }

    double leaf_value_scalar_(
        double G, double H,
        double min_constraint = -std::numeric_limits<double>::infinity(),
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

    std::vector<double> leaf_values_(double G, double H, double min_constraint,
                                     double max_constraint) const {
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
                step =
                    std::clamp(step, -cfg_.max_delta_step, cfg_.max_delta_step);
            out[static_cast<size_t>(c)] =
                std::clamp(step, min_constraint, max_constraint);
        }
        return out;
    }

    double leaf_objective_optimal_(double G, double H) const {
        return -0.5 * splitx::soft(G, cfg_.alpha_) *
               splitx::soft(G, cfg_.alpha_) / (H + cfg_.lambda_);
    }

    inline std::pair<double, double> node_totals_summed_(const Node& n) const {
        double G = 0.0, H = 0.0;
        const int K_ = std::max(n.K, 1);
        for (int c = 0; c < K_; ++c) {
            G += node_uses_goss_(n) ? n.goss_weighted_G[static_cast<size_t>(c)]
                                    : n.G[static_cast<size_t>(c)];
            H += node_uses_goss_(n) ? n.goss_weighted_H[static_cast<size_t>(c)]
                                    : n.H[static_cast<size_t>(c)];
        }
        return {G, H};
    }

    void build_feature_pool_() {
        std::vector<int> all(P_);
        std::iota(all.begin(), all.end(), 0);

        if (cfg_.feature_bagging_k > 0) {
            const int k = std::min(std::max(1, cfg_.feature_bagging_k), P_);
            feat_pool_ =
                sample_k_(all, k, cfg_.feature_bagging_with_replacement);
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

    std::vector<int> sample_k_(const std::vector<int>& pool, int k,
                               bool with_replacement) const {
        if (k <= 0) return {};
        if (static_cast<int>(pool.size()) <= k && !with_replacement)
            return pool;

        std::vector<int> out;
        out.reserve(k);
        if (with_replacement) {
            std::uniform_int_distribution<int> J(
                0, static_cast<int>(pool.size()) - 1);
            for (int i = 0; i < k; ++i) out.push_back(pool[J(rng_)]);
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

        const int base = (cfg_.growth == TreeConfig::Growth::LeafWise
                              ? cfg_.colsample_bynode_percent
                              : cfg_.colsample_bylevel_percent);
        const int pct = std::clamp(base, 1, 100);
        if (pct >= 100) return pool;

        const int k = std::max(1, static_cast<int>(pool.size()) * pct / 100);
        return sample_k_(pool, k, false);
    }

    bool interaction_set_allowed_(const std::vector<int>& path,
                                  const std::vector<int>& proposed) const {
        if (cfg_.interaction_constraints.empty()) return true;
        return std::ranges::any_of(
            cfg_.interaction_constraints, [&](const auto& group) {
                auto contains = [&](int feature) {
                    return std::ranges::find(group, feature) != group.end();
                };
                return std::ranges::all_of(path, contains) &&
                       std::ranges::all_of(proposed, contains);
            });
    }

    void filter_interaction_features_(std::vector<int>& features,
                                      const Node& node) const {
        if (cfg_.interaction_constraints.empty()) return;
        std::erase_if(features, [&](int feature) {
            return !interaction_set_allowed_(node.path_features,
                                             std::vector<int>{feature});
        });
    }

    void apply_tree_level_row_subsample_(std::vector<int>& rows) {
        const double rate = cfg_.subsample_bytree;
        if (rate >= 1.0 || rows.empty()) return;

        std::vector<int> out;
        out.reserve(static_cast<size_t>(std::ceil(rate * rows.size())));

        if (cfg_.subsample_importance_scale) {
            apply_importance_weighted_subsample_(rows, rate, out);
        } else if (!cfg_.subsample_with_replacement) {
            std::uniform_real_distribution<double> U(0.0, 1.0);
            for (int r : rows)
                if (U(rng_) < rate) out.push_back(r);
        } else {
            const int k =
                std::max(1, static_cast<int>(std::round(rate * rows.size())));
            std::uniform_int_distribution<int> J(
                0, static_cast<int>(rows.size()) - 1);
            for (int i = 0; i < k; ++i) out.push_back(rows[J(rng_)]);
            std::ranges::sort(out);
            out.erase(std::unique(out.begin(), out.end()), out.end());
        }

        if (!out.empty()) rows.swap(out);
    }

    void apply_importance_weighted_subsample_(const std::vector<int>& rows,
                                              double rate,
                                              std::vector<int>& out) {
        std::vector<double> weights;
        weights.reserve(rows.size());
        for (int r : rows) weights.push_back(std::abs((*g_)[r]) + 1e-10);

        const int k =
            std::max(1, static_cast<int>(std::round(rate * rows.size())));
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        for (int i = 0; i < k; ++i) out.push_back(rows[dist(rng_)]);
        std::ranges::sort(out);
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }

    const std::vector<int8_t>* maybe_monotone_() const {
        return (cfg_.monotone_constraints.size() == static_cast<size_t>(P_))
                   ? &cfg_.monotone_constraints
                   : nullptr;
    }

    bool use_exact_for_(const Node& nd) const {
        if (!Xraw_) return false;
        if (cfg_.split_mode == TreeConfig::SplitMode::Exact) return true;
        if (cfg_.split_mode == TreeConfig::SplitMode::Hybrid)
            return nd.C <= cfg_.exact_cutover;
        return false;
    }

        void build_missing_aggregates_(const Node& nd,
                                       std::vector<double>& Gmiss,
                                       std::vector<double>& Hmiss,
                                       std::vector<int>& Cmiss) const {
            Gmiss.assign(static_cast<size_t>(T.P_), 0.0);
            Hmiss.assign(static_cast<size_t>(T.P_), 0.0);
            Cmiss.assign(T.P_, 0);

            if (!T.Xraw_) return;

            const bool has_mask = (T.Xmiss_ != nullptr);
            for (int i = nd.lo; i < nd.hi; ++i) {
                const int r = index_pool[i];
                const size_t row =
                    static_cast<size_t>(r) * static_cast<size_t>(T.P_);
                for (int f = 0; f < T.P_; ++f) {
                    const bool miss =
                        has_mask ? (T.Xmiss_[row + static_cast<size_t>(f)] != 0)
                                 : !std::isfinite(
                                       T.Xraw_[row + static_cast<size_t>(f)]);
                    if (miss) {
                        Gmiss[static_cast<size_t>(f)] += (*T.g_)[r];
                        Hmiss[static_cast<size_t>(f)] += (*T.h_)[r];
                        Cmiss[f] += 1;
                    }
                }
            }
        }

    bool eval_with_provider_(Node& nd, foretree::splitx::Candidate& out) {
        if (nd.C < cfg_.min_samples_split || nd.depth >= cfg_.max_depth)
            return false;

        Provider prov(*this, index_pool_);
        const auto* mono = maybe_monotone_();
        const SplitHyper hyp = make_hyper_();

        auto cand = prov.best_split(nd, hyp, mono);
        if (!std::isfinite(cand.gain)) return false;
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
            if (cand.feat < 0) return false;
            if (!std::isfinite(cand.split_value) && cand.thr < 0) return false;
        } else if (cand.kind == splitx::SplitKind::CategoricalPartition) {
            if (cand.feat < 0) return false;
            if (cand.categorical_left_bins.empty()) return false;
        } else if (cand.kind == splitx::SplitKind::Oblique) {
            if (cand.oblique_features.empty()) return false;
            if (cand.oblique_features.size() != cand.oblique_weights.size())
                return false;
            if (!std::isfinite(cand.oblique_threshold)) return false;
        } else if (cand.kind == splitx::SplitKind::PairInteraction) {
            if (cand.pair_feature_a < 0 || cand.pair_feature_b < 0 ||
                cand.pair_feature_a == cand.pair_feature_b)
                return false;
            if (cand.pair_threshold_a < 0 || cand.pair_threshold_b < 0 ||
                cand.pair_quadrant_mask == 0 || cand.pair_quadrant_mask == 15)
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

    int partition_hist_(Node& nd, int feat, int thr, bool miss_left) {
        const uint16_t miss =
            static_cast<uint16_t>(missing_ids_per_feat_[feat]);
        // Read the feature-major column: one contiguous array per feature
        // instead of a cache line per row in the row-major matrix.
        return Xb_->visit_feature_major_codes([&](auto codes) {
            const auto* column = codes.data() + static_cast<size_t>(feat) *
                                                    static_cast<size_t>(N_);
            return RowPartitioner::stable_partition(
                index_pool_, partition_scratch_, nd.lo, nd.hi,
                [&](int row) {
                    const uint16_t bin = static_cast<uint16_t>(column[row]);
                    return bin == miss ? miss_left
                                       : bin <= static_cast<uint16_t>(thr);
                },
                executor_.get());
        });
    }

    int partition_hist_categorical_(
        Node& nd, int feat, const std::vector<int>& categorical_left_bins,
        bool miss_left) {
        const uint16_t miss =
            static_cast<uint16_t>(missing_ids_per_feat_[feat]);
        return RowPartitioner::partition(
            index_pool_, nd.lo, nd.hi, [&](int row) {
                const uint16_t bin = code_at_(row, feat);
                if (bin == miss) return miss_left;
                return std::binary_search(categorical_left_bins.begin(),
                                          categorical_left_bins.end(),
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
                    const size_t off =
                        static_cast<size_t>(r) * static_cast<size_t>(P_) +
                        static_cast<size_t>(f);
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
            if (miss) return sp.oblique_missing_left;
            return z <= sp.oblique_threshold;
        };

        return RowPartitioner::partition(index_pool_, nd.lo, nd.hi, go_left);
    }

    int partition_pair_interaction_(Node& nd,
                                    const foretree::splitx::Candidate& sp) {
        return RowPartitioner::partition(
            index_pool_, nd.lo, nd.hi, [&](int row) {
                const uint16_t a = code_at_(row, sp.pair_feature_a);
                const uint16_t b = code_at_(row, sp.pair_feature_b);
                if (a == static_cast<uint16_t>(
                             missing_ids_per_feat_[sp.pair_feature_a]) ||
                    b == static_cast<uint16_t>(
                             missing_ids_per_feat_[sp.pair_feature_b]))
                    return sp.pair_missing_left;
                const int quadrant = (a > sp.pair_threshold_a ? 2 : 0) |
                                     (b > sp.pair_threshold_b ? 1 : 0);
                return (sp.pair_quadrant_mask & (uint8_t{1} << quadrant)) != 0;
            });
    }

    int partition_exact_(Node& nd, int feat, double split_value,
                         bool miss_left) {
        const size_t stride = static_cast<size_t>(P_);
        const size_t off = static_cast<size_t>(feat);
        const bool has_mask = (Xmiss_ != nullptr);
        return RowPartitioner::partition(
            index_pool_, nd.lo, nd.hi, [&](int row) {
                const size_t index = static_cast<size_t>(row) * stride + off;
                const double value = Xraw_[index];
                const bool missing =
                    has_mask ? Xmiss_[index] != 0 : !std::isfinite(value);
                return missing ? miss_left : value <= split_value;
            });
    }

    void apply_split_(Node& nd, const foretree::splitx::Candidate& sp) {
        std::vector<int> categorical_left_bins;
        if (sp.kind == splitx::SplitKind::CategoricalPartition) {
            categorical_left_bins = sp.categorical_left_bins;
            std::ranges::sort(categorical_left_bins);
            categorical_left_bins.erase(
                std::unique(categorical_left_bins.begin(),
                            categorical_left_bins.end()),
                categorical_left_bins.end());
        }

        const int mid =
            (sp.kind == splitx::SplitKind::CategoricalPartition)
                ? partition_hist_categorical_(
                      nd, sp.feat, categorical_left_bins, sp.miss_left)
            : (sp.kind == splitx::SplitKind::Oblique)
                ? partition_oblique_(nd, sp)
            : (sp.kind == splitx::SplitKind::PairInteraction)
                ? partition_pair_interaction_(nd, sp)
                : ((std::isfinite(sp.split_value) && Xraw_)
                       ? partition_exact_(nd, sp.feat, sp.split_value,
                                          sp.miss_left)
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
            ln.path_features.insert(ln.path_features.end(),
                                    sp.oblique_features.begin(),
                                    sp.oblique_features.end());
        } else if (sp.kind == splitx::SplitKind::PairInteraction) {
            ln.path_features.push_back(sp.pair_feature_a);
            ln.path_features.push_back(sp.pair_feature_b);
        } else {
            ln.path_features.push_back(sp.feat);
        }
        std::ranges::sort(ln.path_features);
        ln.path_features.erase(
            std::unique(ln.path_features.begin(), ln.path_features.end()),
            ln.path_features.end());
        rn.path_features = ln.path_features;

        ln.min_constraint = nd.min_constraint;
        ln.max_constraint = nd.max_constraint;
        rn.min_constraint = nd.min_constraint;
        rn.max_constraint = nd.max_constraint;

        if (sp.kind == splitx::SplitKind::Axis && sp.feat >= 0 &&
            cfg_.monotone_constraints.size() > static_cast<size_t>(sp.feat)) {
            const int8_t mono = cfg_.monotone_constraints[sp.feat];
            if (mono != 0) {
                // To compute the exact mid, we'd need the left and right
                // weights, but they aren't computed here cleanly. We'll
                // approximate the `mid` bound directly using the parent's
                // actual weight, which guarantees safety and simplicity.
                const double node_weight = leaf_value_scalar_(
                    nd.G[0], nd.H[0], nd.min_constraint, nd.max_constraint);
                const double mid_bound = std::clamp(
                    node_weight, nd.min_constraint, nd.max_constraint);
                if (mono > 0) {
                    ln.max_constraint = mid_bound;
                    rn.min_constraint = mid_bound;
                } else if (mono < 0) {
                    ln.min_constraint = mid_bound;
                    rn.max_constraint = mid_bound;
                }
            }
        }

        accum_children_(nd, ln, rn);
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

        if (sp.feat >= 0 && sp.feat < static_cast<int>(feat_gain_.size()) &&
            std::isfinite(sp.gain)) {
            feat_gain_[sp.feat] += sp.gain;
            feat_frequency_[sp.feat]++;
            // Cover = sum of left+right hessian (total node hessian before
            // split)
            double node_h = 0.0;
            for (size_t c = 0; c < nd.H.size(); ++c) node_h += nd.H[c];
            feat_cover_[sp.feat] += node_h;
        }

        if (cfg_.cache_histograms && cfg_.use_sibling_subtract &&
            !node_uses_goss_(nd) && !node_uses_goss_(ln) &&
            !node_uses_goss_(rn)) {
            const auto features = nd.hist_features.empty() ? select_features_()
                                                           : nd.hist_features;
            HistogramProvider prov(*this, index_pool_);

            auto parent_hist = prov.build_histogram(nd, features);
            Node& smaller = (ln.C <= rn.C) ? ln : rn;
            Node& larger = (ln.C <= rn.C) ? rn : ln;
            auto small_hist = prov.build_histogram(smaller, features);

            if (cfg_.cache_histograms) {
                // The parent's histogram is dead after this split: reuse its
                // buffer for the larger child when nothing else holds it.
                nd.histogram.reset();
                if (nd.depth == 0) tree_histogram_.reset();
                if (parent_hist.use_count() == 1) {
                    larger.histogram = std::move(parent_hist);
                } else {
                    larger.histogram = acquire_histogram_();
                    *larger.histogram = *parent_hist;
                }
                subtract_histogram_(*larger.histogram, *small_hist, features);
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
            if (nd.depth == 0) tree_histogram_.reset();
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

    inline bool accept_split_(const Node& nd,
                              const foretree::splitx::Candidate& sp) const {
        if (!cfg_.on_tree.enabled) return true;

        double g = sp.gain;
        if (cfg_.on_tree.ccp_alpha > 0.0) g -= cfg_.on_tree.ccp_alpha;

        const double min_abs =
            std::max(cfg_.on_tree.min_gain, cfg_.on_tree.min_impurity_decrease);
        if (min_abs > 0.0 && g < (min_abs - cfg_.on_tree.eps)) return false;

        if (cfg_.on_tree.min_gain_rel > 0.0) {
            const double base = std::abs(parent_leaf_objective_(nd));
            const double rel_thresh = cfg_.on_tree.min_gain_rel * base;
            if (g < (rel_thresh - cfg_.on_tree.eps)) return false;
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
            n.leaf_values =
                leaf_values_(G, H, n.min_constraint, n.max_constraint);
        n.histogram.reset();
        n.hist_features.clear();
        n.hist_valid = false;
    }

    void grow_leaf_() {
        GrowthPolicy::leaf_wise(
            nodes_[0].id, cfg_.max_leaves,
            [&](int id, auto& split) {
                return eval_node_split_(*by_id_(id), split);
            },
            [&](int id, const auto& split) {
                return accept_split_(*by_id_(id), split);
            },
            [&](int id, const auto& split) {
                apply_split_(*by_id_(id), split);
            },
            [&](int id) { finalize_leaf_(*by_id_(id)); },
            [&](int id) {
                const Node* n = by_id_(id);
                return std::pair{n->left, n->right};
            },
            [&](int id, const auto& split) {
                return priority_(split.gain, *by_id_(id));
            });
    }

    void grow_level_() {
        GrowthPolicy::level_wise(
            nodes_[0].id, cfg_.max_depth,
            [&](int id, auto& split) {
                return eval_node_split_(*by_id_(id), split);
            },
            [&](int id, const auto& split) {
                return accept_split_(*by_id_(id), split);
            },
            [&](int id, const auto& split) {
                apply_split_(*by_id_(id), split);
            },
            [&](int id) { finalize_leaf_(*by_id_(id)); },
            [&](int id) {
                const Node* n = by_id_(id);
                return std::pair{n->left, n->right};
            });
    }

    void grow_oblivious_() {
        GrowthPolicy::oblivious(
            nodes_[0].id, cfg_.max_depth,
            [&](int id, auto& split) {
                return eval_node_split_(*by_id_(id), split);
            },
            [&](int id, const auto& split) {
                return accept_split_(*by_id_(id), split);
            },
            [&](int id, const auto& split) {
                apply_split_(*by_id_(id), split);
            },
            [&](int id) { finalize_leaf_(*by_id_(id)); },
            [&](int id) {
                const Node* n = by_id_(id);
                return std::pair{n->left, n->right};
            });
    }

    bool should_use_neural_leaf_(const Node& n) const {
        if (!cfg_.neural_leaf.enabled) return false;
        if (n.C < cfg_.neural_leaf.min_samples) return false;
        if (n.depth < cfg_.neural_leaf.max_depth_start) return false;
        if (!Xraw_eval_) return false;

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
        n.leaf_values =
            leaf_values_(GG, HH, n.min_constraint, n.max_constraint);

        if (!cfg_.neural_leaf.enabled) return;
        if (!Xraw_eval_) return;
        if (n.C < cfg_.neural_leaf.min_samples) return;

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
                    Xmiss_eval_
                        ? (Xmiss_eval_[(size_t)r * (size_t)P_ + (size_t)j] != 0)
                        : !std::isfinite(v);
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
        auto candidate =
            std::make_unique<GpuNeuralLeafPredictor>(neural_leaf_cfg_);
        candidate->fit(Xbuf.data(), M, ybuf.data(), wbuf.data());

        if (candidate->valid()) {
            n.neural_leaf = std::move(candidate);
        } else {
            n.neural_leaf.reset();
        }
    }

    double compute_feature_correlation_strength_(
        const std::vector<double>& residuals, const Node& n) const {
        if (!Xraw_eval_ || residuals.empty() || n.C < 3) return 0.0;
        double max_abs_r = 0.0;

        for (int feat = 0; feat < P_; ++feat) {
            std::vector<double> xs, ys;
            xs.reserve(n.C);
            ys.reserve(n.C);

            for (int i = n.lo, k = 0; i < n.hi; ++i, ++k) {
                const int r = index_pool_[i];
                const double val =
                    Xraw_eval_[(size_t)r * (size_t)P_ + (size_t)feat];
                const bool miss =
                    Xmiss_eval_
                        ? (Xmiss_eval_[(size_t)r * (size_t)P_ + (size_t)feat] !=
                           0)
                        : !std::isfinite(val);
                if (miss) continue;
                xs.push_back(val);
                ys.push_back(residuals[(size_t)k]);
            }

            if (xs.size() < 3) continue;

            auto mean = [](const std::vector<double>& v) {
                return std::accumulate(v.begin(), v.end(), 0.0) /
                       std::max<size_t>(1, v.size());
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
            if (Sxx <= 0.0 || Syy <= 0.0) continue;

            const double r = Sxy / std::sqrt(Sxx * Syy);
            max_abs_r = std::max(max_abs_r, std::abs(r));
        }
        return max_abs_r;
    }

    void pack_() {
        K_ = std::max(cfg_.num_classes - 1, 1);
        root_id_ = PackedTreeBuilder::build(
            nodes_, K_,
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

