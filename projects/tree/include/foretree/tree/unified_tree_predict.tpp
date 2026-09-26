    bool training_predictions(std::vector<double>& out, int N) const {
        if (!packed_ || K_ != 1 || cfg_.neural_leaf.enabled || N != N_ ||
            static_cast<int>(index_pool_.size()) != N)
            return false;
        std::vector<const Node*> leaves;
        for (const Node& n : nodes_) {
            if (n.is_leaf) {
                if (n.leaf_values.empty()) return false;
                leaves.push_back(&n);
            } else if (n.split_kind == splitx::SplitKind::Oblique ||
                       std::isfinite(n.split_value)) {
                return false;  // routed by raw values during training
            }
        }
        out.resize(static_cast<size_t>(N));
        auto fill = [&](int begin, int end) {
            for (int l = begin; l < end; ++l) {
                const Node& n = *leaves[static_cast<size_t>(l)];
                const double value = n.leaf_values[0];
                for (int i = n.lo; i < n.hi; ++i)
                    out[static_cast<size_t>(index_pool_[static_cast<size_t>(i)])] =
                        value;
            }
        };
        if (executor_)
            executor_->parallel_for(0, static_cast<int>(leaves.size()), 1, fill);
        else
            fill(0, static_cast<int>(leaves.size()));
        return true;
    }

    void release_training_rows() {
        std::vector<int>().swap(index_pool_);
        std::vector<int>().swap(partition_scratch_);
    }

    std::vector<double> predict(const QuantizedDataset& Xb,
                                const double* Xraw_opt = nullptr) const {
        if (Xb.features() != P_)
            throw std::invalid_argument("UnifiedTree::predict: P mismatch");
        std::vector<double> out(static_cast<size_t>(Xb.rows()), 0.0);
        if (!packed_) return out;
        Xb.visit_codes([&](auto codes) {
            using Code = typename decltype(codes)::value_type;
            auto predict_rows = [&](int begin, int end) {
                for (int row = begin; row < end; ++row) {
                    const Code* row_binned =
                        codes.data() +
                        static_cast<size_t>(row) * static_cast<size_t>(P_);
                    out[static_cast<size_t>(row)] =
                        predict_one_compact_(row_binned, row, Xraw_opt);
                }
            };
            // Read-only traversal: rows are independent. The forest calls this
            // on the whole training set after every tree.
            if (executor_)
                executor_->parallel_for(0, Xb.rows(), 4096, predict_rows);
            else
                predict_rows(0, Xb.rows());
        });
        return out;
    }

    double predict_one_binned(const QuantizedDataset& Xb, int row_idx,
                              const double* Xraw_opt = nullptr) const {
        return Xb.visit_codes([&](auto codes) {
            const auto* row = codes.data() + static_cast<size_t>(row_idx) *
                                                 static_cast<size_t>(P_);
            return predict_one_compact_(row, row_idx, Xraw_opt);
        });
    }

    double predict_one_binned_typed(const Code* row_binned, int row_idx,
                                    const double* Xraw_opt = nullptr) const {
        return predict_one_compact_(row_binned, row_idx, Xraw_opt);
    }

    inline double predict_feature_value_(const uint16_t* row_binned,
                                         int row_idx, int feat,
                                         const PredictRawView& raw_view) const {
        if (raw_view.Xraw) {
            const size_t raw_off =
                static_cast<size_t>(row_idx) * static_cast<size_t>(P_) +
                static_cast<size_t>(feat);
            if (raw_view.Xmiss && raw_view.Xmiss[raw_off] != 0)
                return std::numeric_limits<double>::quiet_NaN();
            return raw_view.Xraw[raw_off];
        }
        return binned_value_for_feature_(feat, row_binned[feat]);
    }

    inline bool predict_go_left_oblique_(int id, const uint16_t* row_binned,
                                         int row_idx, bool miss_left,
                                         const PredictRawView& raw_view) const {
        bool go_left = miss_left;
        const int oblique_off = packed_tree_.oblique_offsets[id];
        const int oblique_cnt = packed_tree_.oblique_counts[id];
        if (oblique_off < 0 || oblique_cnt <= 0) return go_left;

        double z = 0.0;
        for (int k = 0; k < oblique_cnt; ++k) {
            const int fz = packed_tree_.oblique_features[oblique_off + k];
            const double xv =
                predict_feature_value_(row_binned, row_idx, fz, raw_view);
            if (!std::isfinite(xv)) return miss_left;
            z += packed_tree_.oblique_weights[oblique_off + k] * xv;
        }
        return (z <= packed_tree_.oblique_thresholds[id]);
    }

    inline bool predict_go_left_categorical_(int id, const uint16_t* row_binned,
                                             int feat, bool miss_left) const {
        const uint16_t b = row_binned[feat];
        const uint16_t feat_miss =
            static_cast<uint16_t>(missing_ids_per_feat_[feat]);
        if (b == feat_miss) return miss_left;

        const int off = packed_tree_.categorical_offsets[id];
        const int cnt = packed_tree_.categorical_counts[id];
        if (off < 0 || cnt <= 0) return false;
        const auto beg = packed_tree_.categorical_bins.begin() + off;
        const auto end = beg + cnt;
        return std::binary_search(beg, end, static_cast<int>(b));
    }

    inline bool predict_go_left_pair_(int id, const uint16_t* row_binned,
                                      bool miss_left) const {
        const int fa = packed_tree_.pair_features_a[id];
        const int fb = packed_tree_.pair_features_b[id];
        const uint16_t a = row_binned[fa];
        const uint16_t b = row_binned[fb];
        if (a == static_cast<uint16_t>(missing_ids_per_feat_[fa]) ||
            b == static_cast<uint16_t>(missing_ids_per_feat_[fb]))
            return miss_left;
        const int quadrant = (a > packed_tree_.pair_thresholds_a[id] ? 2 : 0) |
                             (b > packed_tree_.pair_thresholds_b[id] ? 1 : 0);
        return (packed_tree_.pair_quadrant_masks[id] &
                (uint8_t{1} << quadrant)) != 0;
    }

    inline bool predict_go_left_axis_(int id, const uint16_t* row_binned,
                                      int row_idx, int feat, int thr,
                                      bool miss_left,
                                      const PredictRawView& raw_view) const {
        const uint16_t b = row_binned[feat];
        const uint16_t feat_miss =
            static_cast<uint16_t>(missing_ids_per_feat_[feat]);
        const bool is_miss = (b == feat_miss);

        if (std::isfinite(packed_tree_.split_values[id])) {
            if (raw_view.Xraw) {
                const double xv =
                    predict_feature_value_(row_binned, row_idx, feat, raw_view);
                return std::isfinite(xv) ? (xv <= packed_tree_.split_values[id])
                                         : miss_left;
            }
            if (!is_miss) {
                const double xv = binned_value_for_feature_(feat, b);
                return std::isfinite(xv) ? (xv <= packed_tree_.split_values[id])
                                         : miss_left;
            }
            return miss_left;
        }

        return is_miss ? miss_left : (b <= static_cast<uint16_t>(thr));
    }

    inline double predict_one_with_raw_opt_(const uint16_t* row_binned,
                                            int row_idx,
                                            const double* Xraw_opt) const {
        const auto raw_view = resolve_predict_raw_view_(Xraw_opt);
        int id = root_id_;
        while (id >= 0 && packed_tree_.leaf_flags[id] == 0) {
            const int f = packed_tree_.features[id];
            const int t = packed_tree_.thresholds[id];
            const bool ml = (packed_tree_.missing_left[id] != 0);
            const auto kind =
                static_cast<splitx::SplitKind>(packed_tree_.split_kinds[id]);
            bool go_left = false;
            if (kind == splitx::SplitKind::Oblique) {
                go_left = predict_go_left_oblique_(id, row_binned, row_idx, ml,
                                                   raw_view);
            } else if (kind == splitx::SplitKind::PairInteraction) {
                go_left = predict_go_left_pair_(id, row_binned, ml);
            } else if (kind == splitx::SplitKind::CategoricalPartition) {
                go_left = predict_go_left_categorical_(id, row_binned, f, ml);
            } else {
                go_left = predict_go_left_axis_(id, row_binned, row_idx, f, t,
                                                ml, raw_view);
            }
            id = go_left ? packed_tree_.left_children[id]
                         : packed_tree_.right_children[id];
        }

        return predict_leaf_value_(id, row_idx, raw_view);
    }

    inline double predict_one_with_raw_(const uint16_t* row_binned,
                                        int row_idx) const {
        return predict_one_with_raw_opt_(row_binned, row_idx, nullptr);
    }

    bool row_has_valid_neural_inputs_(int row_idx,
                                      const PredictRawView& raw_view) const {
        if (!raw_view.Xraw) return false;
        const size_t row_off =
            static_cast<size_t>(row_idx) * static_cast<size_t>(P_);
        for (int feat = 0; feat < P_; ++feat) {
            const size_t off = row_off + static_cast<size_t>(feat);
            const double xv = raw_view.Xraw[off];
            const bool miss = raw_view.Xmiss ? (raw_view.Xmiss[off] != 0)
                                             : !std::isfinite(xv);
            if (miss || !std::isfinite(xv)) return false;
        }
        return true;
    }

    double predict_leaf_value_(int leaf_id, int row_idx,
                               const PredictRawView& raw_view) const {
        if (leaf_id < 0) return 0.0;

        const Node* leaf_node = by_id_(leaf_id);
        if (!leaf_node) return 0.0;

        if (!leaf_node->has_neural_leaf()) {
            if (K_ <= 1) {
                if (leaf_id >=
                    static_cast<int>(packed_tree_.leaf_values.size()))
                    return 0.0;
                return packed_tree_.leaf_values[static_cast<size_t>(leaf_id)];
            } else {
                // Multiclass: return first class
                if (leaf_id >=
                    static_cast<int>(packed_tree_.leaf_values.size() / K_))
                    return 0.0;
                return packed_tree_
                    .leaf_values[static_cast<size_t>(leaf_id) * K_];
            }
        }

        if (!raw_view.Xraw) {
            throw std::runtime_error(
                "UnifiedTree::predict: neural leaf inference requires raw "
                "features; call "
                "predict(..., Xraw) or use ForeForest.predict(...)");
        }

        if (!row_has_valid_neural_inputs_(row_idx, raw_view))
            return (leaf_node->leaf_values.empty()) ? 0.0
                                                    : leaf_node->leaf_values[0];

        const double* row_ptr = raw_view.Xraw + static_cast<size_t>(row_idx) *
                                                    static_cast<size_t>(P_);
        return leaf_node->neural_leaf->predict_one(row_ptr);
    }

    void validate_tree_shap_support_() const {
        for (const auto& n : nodes_) {
            if (n.has_neural_leaf()) {
                throw std::runtime_error(
                    "UnifiedTree::predict_contrib: TreeSHAP does not support "
                    "neural leaves");
            }
            if (!n.is_leaf && n.split_kind != splitx::SplitKind::Axis) {
                throw std::runtime_error(
                    "UnifiedTree::predict_contrib: TreeSHAP currently supports "
                    "axis-aligned splits only");
            }
        }
    }

    inline double node_cover_(int id) const {
        if (id < 0 || id >= static_cast<int>(packed_tree_.cover.size()))
            return 0.0;
        return packed_tree_.cover[static_cast<size_t>(id)];
    }

    inline double predict_one_compact_(const Code* row_binned, int row_idx,
                                       const double* Xraw_opt) const {
        PredictRawView raw_view = resolve_predict_raw_view_(Xraw_opt);
        int id = root_id_;
        while (!packed_tree_.leaf_flags[static_cast<size_t>(id)]) {
            const int feature = packed_tree_.features[static_cast<size_t>(id)];
            const bool missing_left =
                packed_tree_.missing_left[static_cast<size_t>(id)] != 0;
            bool go_left = false;
            if (packed_tree_.split_kinds[static_cast<size_t>(id)] ==
                static_cast<uint8_t>(splitx::SplitKind::Oblique)) {
                bool missing = false;
                double projection = 0.0;
                const int offset =
                    packed_tree_.oblique_offsets[static_cast<size_t>(id)];
                const int count =
                    packed_tree_.oblique_counts[static_cast<size_t>(id)];
                for (int i = 0; i < count; ++i) {
                    const int f =
                        packed_tree_
                            .oblique_features[static_cast<size_t>(offset + i)];
                    double value =
                        raw_view.Xraw
                            ? raw_view.Xraw[static_cast<size_t>(row_idx) *
                                                static_cast<size_t>(P_) +
                                            static_cast<size_t>(f)]
                            : binned_value_for_feature_(
                                  f, static_cast<uint16_t>(row_binned[f]));
                    if (!std::isfinite(value)) {
                        missing = true;
                        break;
                    }
                    projection +=
                        packed_tree_
                            .oblique_weights[static_cast<size_t>(offset + i)] *
                        value;
                }
                go_left =
                    missing
                        ? missing_left
                        : projection <=
                              packed_tree_
                                  .oblique_thresholds[static_cast<size_t>(id)];
            } else if (packed_tree_.split_kinds[static_cast<size_t>(id)] ==
                       static_cast<uint8_t>(
                           splitx::SplitKind::PairInteraction)) {
                const int fa =
                    packed_tree_.pair_features_a[static_cast<size_t>(id)];
                const int fb =
                    packed_tree_.pair_features_b[static_cast<size_t>(id)];
                const uint16_t a = static_cast<uint16_t>(row_binned[fa]);
                const uint16_t b = static_cast<uint16_t>(row_binned[fb]);
                if (a == static_cast<uint16_t>(missing_ids_per_feat_[fa]) ||
                    b == static_cast<uint16_t>(missing_ids_per_feat_[fb])) {
                    go_left = missing_left;
                } else {
                    const int quadrant =
                        (a > packed_tree_
                                     .pair_thresholds_a[static_cast<size_t>(id)]
                             ? 2
                             : 0) |
                        (b > packed_tree_
                                     .pair_thresholds_b[static_cast<size_t>(id)]
                             ? 1
                             : 0);
                    go_left =
                        (packed_tree_
                             .pair_quadrant_masks[static_cast<size_t>(id)] &
                         (uint8_t{1} << quadrant)) != 0;
                }
            } else {
                const uint16_t code =
                    static_cast<uint16_t>(row_binned[feature]);
                const size_t raw_offset =
                    static_cast<size_t>(row_idx) * static_cast<size_t>(P_) +
                    static_cast<size_t>(feature);
                const bool exact_axis =
                    raw_view.Xraw &&
                    packed_tree_.split_kinds[static_cast<size_t>(id)] ==
                        static_cast<uint8_t>(splitx::SplitKind::Axis) &&
                    std::isfinite(
                        packed_tree_.split_values[static_cast<size_t>(id)]);
                if (exact_axis) {
                    const double value = raw_view.Xraw[raw_offset];
                    const bool missing = raw_view.Xmiss
                                             ? raw_view.Xmiss[raw_offset] != 0
                                             : !std::isfinite(value);
                    go_left =
                        missing
                            ? missing_left
                            : value <=
                                  packed_tree_
                                      .split_values[static_cast<size_t>(id)];
                } else if (code == static_cast<uint16_t>(
                                       missing_ids_per_feat_[feature])) {
                    go_left = missing_left;
                } else if (packed_tree_.split_kinds[static_cast<size_t>(id)] ==
                           static_cast<uint8_t>(
                               splitx::SplitKind::CategoricalPartition)) {
                    const int offset =
                        packed_tree_
                            .categorical_offsets[static_cast<size_t>(id)];
                    const int count =
                        packed_tree_
                            .categorical_counts[static_cast<size_t>(id)];
                    go_left = std::binary_search(
                        packed_tree_.categorical_bins.begin() + offset,
                        packed_tree_.categorical_bins.begin() + offset + count,
                        static_cast<int>(code));
                } else {
                    go_left =
                        code <=
                        static_cast<uint16_t>(
                            packed_tree_.thresholds[static_cast<size_t>(id)]);
                }
            }
            id = go_left ? packed_tree_.left_children[static_cast<size_t>(id)]
                         : packed_tree_.right_children[static_cast<size_t>(id)];
        }
        return packed_tree_
            .leaf_values[static_cast<size_t>(id) * static_cast<size_t>(K_)];
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
        if (off < bin_centers_flat_.size()) return bin_centers_flat_[off];
        return std::numeric_limits<double>::quiet_NaN();
    }

