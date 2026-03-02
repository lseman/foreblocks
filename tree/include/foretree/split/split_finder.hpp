#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "foretree/split/split_helpers.hpp"

namespace foretree {

// ============================================================================
// 1) Axis (histogram/exact) — uses splitx providers & scanners
// ============================================================================
class AxisSplitFinder {
public:
    // Histogram-backed axis split (variable bins supported)
    splitx::Candidate best_axis(const splitx::SplitContext &ctx) const {
        using namespace splitx;
        Candidate best;
        best.kind = SplitKind::Axis;
        best.gain = NEG_INF;
        if (!ctx.G || !ctx.H || !ctx.C || ctx.P <= 0) return best;

        const auto  &G           = *ctx.G;
        const auto  &H           = *ctx.H;
        const auto  &C           = *ctx.C;
        const auto  *mono_vec    = ctx.mono_ptr();
        const double parent_gain = leaf_obj(ctx.Gp, ctx.Hp, ctx.hyp.lambda_, ctx.hyp.alpha_);

        for (int f = 0; f < ctx.P; ++f) {
            const int finite_bins = ctx.get_feature_bins(f);
            if (finite_bins <= 0) continue;

            const int8_t mono = (mono_vec && f < (int)mono_vec->size()) ? (*mono_vec)[f] : 0;

            HistProvider prov(ctx, f, G, H, C);

            double Gm = 0.0, Hm = 0.0;
            int    Cm       = 0;
            bool   has_miss = false;
            prov.missing(Gm, Hm, Cm, has_miss);
            const int totalC = prov.total_count();
            const int steps  = prov.steps();

            Candidate cand = scan_axis_with_policy(ctx, f, prov, steps, mono, ctx.hyp.missing_policy, parent_gain,
                                                   totalC, Gm, Hm, Cm, has_miss);

            if (cand.thr >= 0 && cand.gain > best.gain) best = cand;
        }
        return best;
    }

    // Exact-mode axis split (raw X + node index + missing policy)
    splitx::Candidate best_axis_exact(const splitx::SplitContext &ctx, const double *Xraw, int P, const int *node_idx,
                                      int nidx, int missing_policy, const uint8_t *miss_mask) const {
        using namespace splitx;
        Candidate best;
        best.kind = SplitKind::Axis;
        best.gain = NEG_INF;
        best.thr  = -1;
        if (!Xraw || !ctx.row_g || !ctx.row_h || !node_idx || nidx <= 1 || P <= 0) return best;

        const auto   *mono_vec    = ctx.mono_ptr();
        const double  parent_gain = leaf_obj(ctx.Gp, ctx.Hp, ctx.hyp.lambda_, ctx.hyp.alpha_);
        ExactProvider prov(ctx, Xraw, P, node_idx, nidx, miss_mask);

        for (int f = 0; f < P; ++f) {
            const int n_valid = prov.prepare_feature(f);
            if (n_valid < 2) continue;

            double Gm = 0.0, Hm = 0.0;
            int    Cm       = 0;
            bool   has_miss = false;
            prov.missing(Gm, Hm, Cm, has_miss);
            if (ctx.Gmiss) Gm = ctx.Gmiss[f];
            if (ctx.Hmiss) Hm = ctx.Hmiss[f];

            const int    totalC = prov.total_count_for_nvalid(n_valid);
            const int8_t mono   = (mono_vec && f < (int)mono_vec->size()) ? (*mono_vec)[f] : 0;
            const int    steps  = prov.steps_for_nvalid(n_valid);

            // Convention here: missing_policy == 0 means "use context policy"
            // (ctx.hyp.missing_policy), otherwise use the explicit override.
            constexpr int kUseContextMissingPolicy = 0;
            const int     mpol = (missing_policy != kUseContextMissingPolicy) ? missing_policy : ctx.hyp.missing_policy;
            Candidate     cand =
                scan_axis_with_policy(ctx, f, prov, steps, mono, mpol, parent_gain, totalC, Gm, Hm, Cm, has_miss);

            if (cand.thr >= 0 && cand.thr + 1 < n_valid) {
                const double vl  = prov.col[static_cast<size_t>(cand.thr)].first;
                const double vr  = prov.col[static_cast<size_t>(cand.thr + 1)].first;
                cand.split_value = 0.5 * (vl + vr);
            }

            if (cand.thr >= 0 && cand.gain > best.gain) best = cand;
        }
        return best;
    }
};

// ============================================================================
// 2) Categorical K-way from histograms with variable bins
// ============================================================================
class CategoricalKWaySplitFinder {
public:
    int    max_groups                   = 8; // retained for compatibility; used as a soft cap in ranked-prefix search
    bool   use_exhaustive_search        = true;
    int    max_exhaustive_cardinality   = 16;
    bool   use_ordered_scan             = true;
    bool   use_greedy_merge             = true;
    int    greedy_merge_min_cardinality = 24;
    int    greedy_merge_max_cardinality = 512;
    int    greedy_merge_target_groups   = 16;
    bool   use_ranked_prefix_scan       = true;
    int    max_ranked_prefix_candidates = 32;
    double complexity_penalty           = 0.0; // optional regularization for high-cardinality categoricals

    splitx::Candidate best_kway(const splitx::SplitContext &ctx) const {
        using namespace splitx;
        Candidate best;
        best.kind = SplitKind::KWay;
        best.gain = NEG_INF;
        if (!ctx.G || !ctx.H || !ctx.C || ctx.P <= 0) return best;
        const auto  &G           = *ctx.G;
        const auto  &H           = *ctx.H;
        const auto  &C           = *ctx.C;
        const auto  *mono_vec    = ctx.mono_ptr();
        const double parent_gain = leaf_obj(ctx.Gp, ctx.Hp, ctx.hyp.lambda_, ctx.hyp.alpha_);

        for (int f = 0; f < ctx.P; ++f) {
            const int finite_bins = ctx.get_feature_bins(f);
            if (finite_bins <= 1) continue;

            struct BinStat {
                int    bin         = -1;
                double g           = 0.0;
                double h           = 0.0;
                int    c           = 0;
                double ordered_key = 0.0;
                double rank_key    = 0.0;
            };
            std::vector<BinStat> bins;
            bins.reserve(static_cast<size_t>(finite_bins));

            double finite_h_total = 0.0;
            for (int t = 0; t < finite_bins; ++t) {
                const size_t off = ctx.get_histogram_offset(f, t);
                if (off >= G.size() || off >= H.size() || off >= C.size()) continue;
                const double g = G[off];
                const double h = H[off];
                const int    c = C[off];
                if (c <= 0 && h <= 0.0 && g == 0.0) continue;
                bins.push_back(BinStat{t, g, h, c, 0.0, 0.0});
                finite_h_total += std::max(0.0, h);
            }

            const int non_empty = static_cast<int>(bins.size());
            if (non_empty < 2) continue;

            const double global_mean = ctx.Gp / std::max(ctx.Hp, ctx.eps);
            for (auto &bs : bins) {
                const double mean_g = bs.g / std::max(bs.h, ctx.eps);
                const double smoothed =
                    (mean_g * bs.h + global_mean * ctx.hyp.lambda_) / (bs.h + ctx.hyp.lambda_ + ctx.eps);
                bs.ordered_key    = smoothed;
                double rank_score = std::abs(smoothed);
                if (ctx.use_entropy && finite_h_total > ctx.eps) {
                    const double p = std::max(0.0, bs.h) / finite_h_total;
                    rank_score *= (-p * std::log(p + ctx.eps));
                }
                bs.rank_key = rank_score;
            }

            const int    miss_id  = ctx.get_missing_bin_id(f);
            const size_t miss_off = ctx.get_histogram_offset(f, miss_id);
            const double Gm       = (miss_off < G.size()) ? G[miss_off] : 0.0;
            const double Hm       = (miss_off < H.size()) ? H[miss_off] : 0.0;
            const int    Cm       = (miss_off < C.size()) ? C[miss_off] : 0;

            const int8_t mono = (mono_vec && f < static_cast<int>(mono_vec->size())) ? (*mono_vec)[f] : 0;

            auto eval_candidate = [&](double GL, double HL, int CL, bool &miss_left_out) -> double {
                double best_gain_local = NEG_INF;
                bool   best_dir_local  = true;

                auto eval_dir = [&](bool miss_left) {
                    const double GLx = GL + (miss_left ? Gm : 0.0);
                    const double HLx = HL + (miss_left ? Hm : 0.0);
                    const int    CLx = CL + (miss_left ? Cm : 0);

                    const double GRx = ctx.Gp - GLx;
                    const double HRx = ctx.Hp - HLx;
                    const int    CRx = ctx.Cp - CLx;

                    if (!valid_children(HLx, HRx, CLx, CRx, ctx.hyp)) return;

                    if (mono != 0) {
                        const double wL = weight_from_GRH(GLx, HLx, ctx.hyp);
                        const double wR = weight_from_GRH(GRx, HRx, ctx.hyp);
                        if (!pass_monotone_guard(mono, wL, wR)) return;
                    }

                    const double gain =
                        split_gain_from_parent(parent_gain, GLx, HLx, GRx, HRx, CLx, CRx, ctx.hyp, complexity_penalty,
                                               std::log1p(static_cast<double>(non_empty)));

                    if (gain > best_gain_local) {
                        best_gain_local = gain;
                        best_dir_local  = miss_left;
                    }
                };

                if (ctx.hyp.missing_policy == 1) {
                    eval_dir(true);
                } else if (ctx.hyp.missing_policy == 2) {
                    eval_dir(false);
                } else {
                    eval_dir(true);
                    eval_dir(false);
                }
                miss_left_out = best_dir_local;
                return best_gain_local;
            };

            auto maybe_commit = [&](const std::vector<int> &left_bins, double GL, double HL, int CL) {
                bool         miss_left_local = true;
                const double gain            = eval_candidate(GL, HL, CL, miss_left_local);
                if (!(gain > best.gain)) return;
                std::vector<int> normalized = left_bins;
                std::sort(normalized.begin(), normalized.end());
                normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
                if (normalized.empty()) return;

                best.kind          = SplitKind::KWay;
                best.gain          = gain;
                best.feat          = f;
                best.left_groups   = std::move(normalized);
                best.miss_left     = miss_left_local;
                best.missing_group = miss_left_local ? 0 : 1;
            };

            // 1) Exhaustive search for low-cardinality categoricals.
            if (use_exhaustive_search && non_empty <= std::max(2, max_exhaustive_cardinality) &&
                non_empty < static_cast<int>(8 * sizeof(uint64_t))) {
                const uint64_t full = (uint64_t{1} << non_empty);
                for (uint64_t mask = 1; mask + 1 < full; ++mask) {
                    // Skip mirrored complements by forcing the first category to stay on the left.
                    if ((mask & uint64_t{1}) == 0) continue;
                    double           GL = 0.0, HL = 0.0;
                    int              CL = 0;
                    std::vector<int> left_bins;
                    left_bins.reserve(static_cast<size_t>(non_empty));
                    for (int b = 0; b < non_empty; ++b) {
                        if ((mask & (uint64_t{1} << b)) == 0) continue;
                        const auto &bs = bins[static_cast<size_t>(b)];
                        GL += bs.g;
                        HL += bs.h;
                        CL += bs.c;
                        left_bins.push_back(bs.bin);
                    }
                    maybe_commit(left_bins, GL, HL, CL);
                }
            }

            // 2) Ordered scan (CatBoost-style): sort categories by smoothed statistic and scan prefixes.
            if (use_ordered_scan) {
                std::vector<int> ord(static_cast<size_t>(non_empty));
                std::iota(ord.begin(), ord.end(), 0);
                std::sort(ord.begin(), ord.end(), [&](int ia, int ib) {
                    const auto &a = bins[static_cast<size_t>(ia)];
                    const auto &b = bins[static_cast<size_t>(ib)];
                    if (a.ordered_key != b.ordered_key) return a.ordered_key < b.ordered_key;
                    return a.bin < b.bin;
                });

                double           GL = 0.0, HL = 0.0;
                int              CL = 0;
                std::vector<int> left_bins;
                left_bins.reserve(static_cast<size_t>(non_empty - 1));
                for (int k = 0; k + 1 < non_empty; ++k) {
                    const auto &bs = bins[static_cast<size_t>(ord[static_cast<size_t>(k)])];
                    GL += bs.g;
                    HL += bs.h;
                    CL += bs.c;
                    left_bins.push_back(bs.bin);
                    maybe_commit(left_bins, GL, HL, CL);
                }
            }

            // 3) Ranked multi-candidate scan: prefixes over |stat|-ranked categories.
            if (use_ranked_prefix_scan) {
                std::vector<int> ranked(static_cast<size_t>(non_empty));
                std::iota(ranked.begin(), ranked.end(), 0);
                std::sort(ranked.begin(), ranked.end(), [&](int ia, int ib) {
                    const auto &a = bins[static_cast<size_t>(ia)];
                    const auto &b = bins[static_cast<size_t>(ib)];
                    if (a.rank_key != b.rank_key) return a.rank_key > b.rank_key;
                    return a.bin < b.bin;
                });

                const int prefix_cap =
                    std::min({non_empty - 1, std::max(1, max_ranked_prefix_candidates), std::max(1, max_groups - 1)});
                double           GL = 0.0, HL = 0.0;
                int              CL = 0;
                std::vector<int> left_bins;
                left_bins.reserve(static_cast<size_t>(prefix_cap));
                for (int k = 0; k < prefix_cap; ++k) {
                    const auto &bs = bins[static_cast<size_t>(ranked[static_cast<size_t>(k)])];
                    GL += bs.g;
                    HL += bs.h;
                    CL += bs.c;
                    left_bins.push_back(bs.bin);
                    maybe_commit(left_bins, GL, HL, CL);
                }
            }

            // 4) Greedy merge on ordered categories for medium/high cardinality.
            if (use_greedy_merge && non_empty >= std::max(2, greedy_merge_min_cardinality) &&
                non_empty <= std::max(greedy_merge_min_cardinality, greedy_merge_max_cardinality)) {
                std::vector<int> ord(static_cast<size_t>(non_empty));
                std::iota(ord.begin(), ord.end(), 0);
                std::sort(ord.begin(), ord.end(), [&](int ia, int ib) {
                    const auto &a = bins[static_cast<size_t>(ia)];
                    const auto &b = bins[static_cast<size_t>(ib)];
                    if (a.ordered_key != b.ordered_key) return a.ordered_key < b.ordered_key;
                    return a.bin < b.bin;
                });

                struct GroupAgg {
                    std::vector<int> bins;
                    double           g   = 0.0;
                    double           h   = 0.0;
                    int              c   = 0;
                    double           key = 0.0;
                };

                std::vector<GroupAgg> groups;
                groups.reserve(static_cast<size_t>(non_empty));
                for (int oi : ord) {
                    const auto &bs = bins[static_cast<size_t>(oi)];
                    GroupAgg    grp;
                    grp.bins.push_back(bs.bin);
                    grp.g   = bs.g;
                    grp.h   = bs.h;
                    grp.c   = bs.c;
                    grp.key = bs.ordered_key;
                    groups.push_back(std::move(grp));
                }

                const int target_groups = std::clamp(greedy_merge_target_groups, 2, non_empty);
                while (static_cast<int>(groups.size()) > target_groups) {
                    int    best_i    = -1;
                    double best_cost = std::numeric_limits<double>::infinity();

                    for (int i = 0; i + 1 < static_cast<int>(groups.size()); ++i) {
                        const auto  &a  = groups[static_cast<size_t>(i)];
                        const auto  &b  = groups[static_cast<size_t>(i + 1)];
                        const double wa = std::max(a.h, ctx.eps);
                        const double wb = std::max(b.h, ctx.eps);
                        const double d  = a.key - b.key;
                        // Ward-like adjacent merge cost: merge most similar neighbors first.
                        const double cost = (d * d) * (wa * wb) / (wa + wb + ctx.eps);
                        if (cost < best_cost) {
                            best_cost = cost;
                            best_i    = i;
                        }
                    }
                    if (best_i < 0) break;

                    auto        &lhs  = groups[static_cast<size_t>(best_i)];
                    auto        &rhs  = groups[static_cast<size_t>(best_i + 1)];
                    const double wh_l = std::max(lhs.h, ctx.eps);
                    const double wh_r = std::max(rhs.h, ctx.eps);
                    const double wsum = wh_l + wh_r;
                    lhs.key           = (lhs.key * wh_l + rhs.key * wh_r) / std::max(wsum, ctx.eps);
                    lhs.g += rhs.g;
                    lhs.h += rhs.h;
                    lhs.c += rhs.c;
                    lhs.bins.insert(lhs.bins.end(), rhs.bins.begin(), rhs.bins.end());
                    groups.erase(groups.begin() + (best_i + 1));
                }

                if (groups.size() >= 2) {
                    double           GL = 0.0, HL = 0.0;
                    int              CL = 0;
                    std::vector<int> left_bins;
                    left_bins.reserve(static_cast<size_t>(non_empty));
                    for (size_t gi = 0; gi + 1 < groups.size(); ++gi) {
                        const auto &grp = groups[gi];
                        GL += grp.g;
                        HL += grp.h;
                        CL += grp.c;
                        left_bins.insert(left_bins.end(), grp.bins.begin(), grp.bins.end());
                        maybe_commit(left_bins, GL, HL, CL);
                    }
                }
            }
        }
        return best;
    }
};

// ============================================================================
// 3) Oblique splitter (row-wise and hist) with variable bin support
// ============================================================================
class ObliqueSplitFinder {
public:
    int    k_features                     = 6; // pick top-k by |corr(x,g)|
    int    newton_steps                   = 1;
    double l1                             = 0.0;
    double ridge                          = 1e-3;
    int    hist_exact_cutover             = 2048; // exact projection fallback when node is small
    bool   hist_use_quantile_binning      = true; // adaptive z-binning in histogram path
    int    hist_quantile_binning_max_rows = 4096;
    double hist_clip_quantile             = 0.01; // robust fallback clipping for uniform z-bins

    // New helper: stats-driven w from cov matrix (non-iterative)
    // New helper: stats-driven w from cov matrix (non-iterative)
    void build_stats_hyperplane(const std::vector<const double *> &XS, const double *g, int N, double ridge,
                                std::vector<double> &w) {
        const size_t        K = XS.size();
        std::vector<double> cov(K * K, 0.0); // Symmetric cov(X)
        std::vector<double> cov_g(K, 0.0);   // cov(X, g)
        std::vector<double> sum_x(K, 0.0);   // For mean_x
        double              var_g = 0.0, mean_g = 0.0;
        for (int i = 0; i < N; ++i) {
            mean_g += g[i];
            var_g += g[i] * g[i];
            for (size_t j = 0; j < K; ++j) {
                const double xj = XS[j][i];
                if (!std::isfinite(xj)) continue;
                sum_x[j] += xj;
                cov_g[j] += xj * g[i];
                for (size_t k = j; k < K; ++k) {
                    const double xk = XS[k][i];
                    if (!std::isfinite(xk)) continue;
                    cov[j * K + k] += xj * xk;
                    if (j != k) cov[k * K + j] += xj * xk;
                }
            }
        }
        mean_g /= N;
        var_g                         = (var_g / N) - mean_g * mean_g;
        const double        inv_var_g = (var_g > 1e-12) ? 1.0 / var_g : 0.0;
        std::vector<double> mean_x(K, 0.0);
        for (size_t j = 0; j < K; ++j) { mean_x[j] = sum_x[j] / N; }
        for (size_t j = 0; j < K; ++j) {
            cov_g[j] = (cov_g[j] / N - mean_x[j] * mean_g) * inv_var_g; // Normalize
            for (size_t k = 0; k < K; ++k) cov[j * K + k] = (cov[j * K + k] / N) + ridge;
        }
        // Simple diagonal approx solve: w_j = cov_g_j / cov_jj (stats-driven)
        for (size_t j = 0; j < K; ++j) w[j] = cov_g[j] / cov[j * K + j];
    }

    // Row-wise (exact) oblique
    splitx::Candidate best_oblique(const splitx::SplitContext &ctx, double /*axis_guard_gain*/ = -1.0) const {
        using namespace splitx;
        Candidate out;
        out.kind = SplitKind::Oblique;
        out.gain = NEG_INF;
        if (!ctx.Xcols || ctx.N <= 0 || !ctx.row_g || !ctx.row_h) return out;

        const int P = ctx.P, N = ctx.N;
        if (P <= 1) return out;

        // 1) Rank features by |corr(x, g)|
        std::vector<double> corr(P, 0.0);
        double              mg = 0.0;
        for (int i = 0; i < N; ++i) mg += (double)ctx.row_g[i];
        mg /= std::max(1, N);

        for (int f = 0; f < P; ++f) {
            const double *x   = ctx.Xcols[f];
            double        cnt = 0.0, sx = 0.0;
            for (int i = 0; i < N; ++i) {
                const double xi = x[i];
                if (std::isfinite(xi)) {
                    sx += xi;
                    cnt += 1.0;
                }
            }
            if (cnt < 2.0) {
                corr[f] = 0.0;
                continue;
            }
            const double mx  = sx / cnt;
            double       sxx = 0.0, sgg = 0.0, sxg = 0.0;
            for (int i = 0; i < N; ++i) {
                const double xi = x[i];
                if (!std::isfinite(xi)) continue;
                const double dx = xi - mx, dg = (double)ctx.row_g[i] - mg;
                sxx += dx * dx;
                sgg += dg * dg;
                sxg += dx * dg;
            }
            const double denom = std::sqrt(sxx * sgg) + EPS;
            corr[f]            = (denom <= EPS) ? 0.0 : std::abs(sxg / denom);
        }

        std::vector<int> ord(P);
        std::iota(ord.begin(), ord.end(), 0);
        const int k = std::min(k_features, P);
        std::partial_sort(ord.begin(), ord.begin() + k, ord.end(), [&](int a, int b) { return corr[a] > corr[b]; });
        std::vector<int> S(ord.begin(), ord.begin() + k);
        if ((int)S.size() < 2) return out;

        // 2) Build normal equations on S and solve for w
        std::vector<const double *> XS;
        XS.reserve(S.size());
        for (int f : S) XS.push_back(ctx.Xcols[f]);

        std::vector<double> A, b, w;
        build_normal_eq_cols(XS, ctx.row_g, ctx.row_h, N, ridge + ctx.hyp.lambda_, A, b);
        std::vector<double> A_chol = A;
        if (!cholesky_spd(A_chol, (int)S.size())) return out;
        w.resize(S.size());
        auto b_work = b;
        chol_solve_inplace(A_chol, (int)S.size(), b_work, w);

        if (l1 > 0.0) {
            for (auto &wj : w) {
                if (wj > l1) wj -= l1;
                else if (wj < -l1) wj += l1;
                else wj = 0.0;
            }
        }

        std::vector<double>  z(N, 0.0);
        std::vector<uint8_t> miss(N, 0);

        auto project_and_eval = [&]() {
            for (int i = 0; i < N; ++i) {
                double acc    = 0.0;
                bool   finite = true;
                for (size_t j = 0; j < S.size(); ++j) {
                    const double xi = XS[j][i];
                    if (!std::isfinite(xi)) {
                        finite = false;
                        break;
                    }
                    acc += xi * w[j];
                }
                z[i]    = acc;
                miss[i] = finite ? 0u : 1u;
            }
            return best_split_on_projection(z, ctx.row_g, ctx.row_h, N, miss, ctx.hyp);
        };

        auto [bgain, bthr, bmleft] = project_and_eval();
        std::vector<double> best_w = w;

        for (int step = 1; step < newton_steps; ++step) {
            if (!(bgain > 0.0)) break;
            
            double GL = 0.0, HL = 0.0, GR = 0.0, HR = 0.0;
            for (int i = 0; i < N; ++i) {
                bool go_left = miss[i] ? bmleft : (z[i] <= bthr);
                if (go_left) { GL += ctx.row_g[i]; HL += ctx.row_h[i]; }
                else { GR += ctx.row_g[i]; HR += ctx.row_h[i]; }
            }
            
            double denomL = HL + ctx.hyp.lambda_;
            double denomR = HR + ctx.hyp.lambda_;
            if (!(denomL > 0.0) || !(denomR > 0.0)) break;
            
            double vL = -GL / denomL;
            double vR = -GR / denomR;
            if (std::abs(vL - vR) < 1e-7) break;
            
            std::vector<double> b_new(S.size(), 0.0);
            for (int i = 0; i < N; ++i) {
                bool go_left = miss[i] ? bmleft : (z[i] <= bthr);
                double yi = go_left ? vL : vR;
                double hi_yi = ctx.row_h[i] * yi;
                for (size_t j = 0; j < S.size(); ++j) {
                    if (std::isfinite(XS[j][i])) {
                        b_new[j] += hi_yi * XS[j][i];
                    }
                }
            }
            
            b_work = b_new;
            auto A_work = A;
            chol_solve_inplace(A_work, (int)S.size(), b_work, w);
            
            if (l1 > 0.0) {
                for (auto &wj : w) {
                    if (wj > l1) wj -= l1;
                    else if (wj < -l1) wj += l1;
                    else wj = 0.0;
                }
            }
            
            auto [new_gain, new_thr, new_mleft] = project_and_eval();
            if (new_gain > bgain + 1e-6) {
                bgain  = new_gain;
                bthr   = new_thr;
                bmleft = new_mleft;
                best_w = w;
            } else {
                break;
            }
        }

        if (!(bgain > 0.0)) return out;

        out.gain                 = bgain;
        out.oblique_features     = std::move(S);
        out.oblique_weights      = std::move(best_w);
        out.oblique_threshold    = bthr;
        out.oblique_bias         = 0.0;
        out.oblique_missing_left = bmleft;
        return out;
    }

    // Histogram-backed oblique with variable bin support
    splitx::Candidate best_oblique_hist(const splitx::SplitContext &ctx) const {
        using namespace splitx;
        Candidate out;
        out.kind = SplitKind::Oblique;
        out.gain = NEG_INF;

        if (!ctx.Xb || !ctx.row_index || !ctx.bin_centers || !ctx.row_g || !ctx.row_h) return out;
        const int P = ctx.P, Nn = ctx.N;
        if (P <= 1 || Nn <= 0) return out;

        // 1) rank features by |corr(x, g)| approx using bin centers
        std::vector<double> score(P, 0.0);
        for (int f = 0; f < P; ++f) {
            double sx = 0.0, sxx = 0.0, sg = 0.0, sgg = 0.0, sxg = 0.0;
            int    n = 0;
            for (int rr = 0; rr < Nn; ++rr) {
                const int      i    = ctx.row_index[rr];
                const uint16_t code = ctx.Xb[(size_t)i * (size_t)P + (size_t)f];
                const double   x    = x_from_code_variable(f, code, ctx);
                if (!std::isfinite(x)) continue;
                const double gi = (double)ctx.row_g[i];
                sx += x;
                sxx += x * x;
                sg += gi;
                sgg += gi * gi;
                sxg += x * gi;
                ++n;
            }
            if (n >= 2) {
                const double invn = 1.0 / (double)n;
                const double mx = sx * invn, mg = sg * invn;
                const double cov   = sxg * invn - mx * mg;
                const double varx  = sxx * invn - mx * mx;
                const double varg  = sgg * invn - mg * mg;
                const double denom = std::sqrt(std::max(0.0, varx) * std::max(0.0, varg)) + EPS;
                score[(size_t)f]   = std::abs(cov) / denom;
            }
        }

        std::vector<int> ord(P);
        std::iota(ord.begin(), ord.end(), 0);
        const int k = std::min(k_features, P);
        std::partial_sort(ord.begin(), ord.begin() + k, ord.end(),
                          [&](int a, int b) { return score[(size_t)a] > score[(size_t)b]; });
        std::vector<int> S(ord.begin(), ord.begin() + k);
        if ((int)S.size() < 2) return out;

        // 2) normal equations on codes (ridge includes lambda)
        std::vector<double> A, b, w;
        build_normal_eq_from_codes(S, ctx.Xb, ctx.row_index, Nn, P, ctx, ctx.row_g, ctx.row_h, ridge + ctx.hyp.lambda_, A, b);
        std::vector<double> A_chol = A;
        if (!cholesky_spd(A_chol, (int)S.size())) return out;
        w.resize(S.size());
        auto b_work = b;
        chol_solve_inplace(A_chol, (int)S.size(), b_work, w);

        auto project_and_eval = [&](double &out_gain, double &out_thr, bool &out_mleft) {
            if (Nn <= std::max(2, hist_exact_cutover)) {
                std::vector<double> z((size_t)Nn, 0.0);
                std::vector<uint8_t> miss((size_t)Nn, 0u);
                for (int rr = 0; rr < Nn; ++rr) {
                    const int i = ctx.row_index[rr];
                    bool has_miss = false;
                    double acc = 0.0;
                    for (size_t j = 0; j < S.size(); ++j) {
                        const int f = S[j];
                        const uint16_t code = ctx.Xb[(size_t)i * (size_t)P + (size_t)f];
                        const double x = x_from_code_variable(f, code, ctx);
                        if (!std::isfinite(x)) { has_miss = true; break; }
                        acc += w[j] * x;
                    }
                    miss[rr] = has_miss ? 1u : 0u;
                    if (!has_miss) z[rr] = acc;
                }
                auto [gain, thr, mleft] = best_split_on_projection(z, ctx.row_g, ctx.row_h, Nn, miss, ctx.hyp);
                out_gain = gain; out_thr = thr; out_mleft = mleft;
            } else {
                std::vector<double> Gz, Hz;
                std::vector<int> Cz;
                std::vector<double> z_edges;
                double Gm = 0.0, Hm = 0.0;
                int Cm = 0;
                double zmin = 0.0, zmax = 1.0;

                build_projection_hist_from_codes(
                    S, w, ctx.Xb, ctx.row_index, Nn, P, ctx, ctx.row_g, ctx.row_h, Gz, Hz, Cz, Gm, Hm, Cm, zmin, zmax, &z_edges,
                    hist_use_quantile_binning && Nn <= std::max(2, hist_quantile_binning_max_rows), hist_clip_quantile);

                bool allG0 = std::all_of(Gz.begin(), Gz.end(), [](double v) { return v == 0.0; });
                bool allH0 = std::all_of(Hz.begin(), Hz.end(), [](double v) { return v == 0.0; });
                if (allG0 && allH0) {
                    out_gain = -1.0; return;
                }

                const double Gtot = std::accumulate(Gz.begin(), Gz.end(), 0.0) + Gm;
                const double Htot = std::accumulate(Hz.begin(), Hz.end(), 0.0) + Hm;
                const int Ctot = std::accumulate(Cz.begin(), Cz.end(), 0);
                const double parent = (Gtot * Gtot) / (Htot + ctx.hyp.lambda_);
                const int Bz = (int)Gz.size();

                auto scan_dir = [&](bool mleft) -> std::pair<double, int> {
                    double GL = 0.0, HL = 0.0;
                    int CL = 0;
                    double best_gain = splitx::NEG_INF;
                    int best_t = -1;
                    for (int t = 0; t < Bz - 1; ++t) {
                        GL += Gz[(size_t)t];
                        HL += Hz[(size_t)t];
                        CL += Cz[(size_t)t];
                        const double GLx = GL + (mleft ? Gm : 0.0);
                        const double HLx = HL + (mleft ? Hm : 0.0);
                        const double GRx = Gtot - GLx;
                        const double HRx = Htot - HLx;
                        const int CLx = CL + (mleft ? Cm : 0);
                        const int CRx = (Ctot - CL) + (mleft ? 0 : Cm);
                        const double gain = split_gain_from_parent(parent, GLx, HLx, GRx, HRx, CLx, CRx, ctx.hyp);
                        if (gain > best_gain) { best_gain = gain; best_t = t; }
                    }
                    return {best_gain, best_t};
                };

                auto [gL, tL] = scan_dir(true);
                auto [gR, tR] = scan_dir(false);
                out_mleft = (gL >= gR);
                out_gain = std::max(gL, gR);
                int t = out_mleft ? tL : tR;
                
                if (t < 0 || !(out_gain > 0.0)) { out_gain = -1.0; return; }
                
                if (z_edges.size() == (size_t)Bz + 1) {
                    const int edge_idx = std::clamp(t + 1, 0, Bz);
                    out_thr = z_edges[(size_t)edge_idx];
                } else {
                    const double dz = (zmax - zmin) / (double)Bz;
                    out_thr = zmin + dz * (t + 1);
                }
            }
        };

        if (l1 > 0.0) {
            for (auto &wj : w) {
                if (wj > l1) wj -= l1;
                else if (wj < -l1) wj += l1;
                else wj = 0.0;
            }
        }

        double bgain = -1.0, bthr = std::numeric_limits<double>::quiet_NaN();
        bool bmleft = true;
        project_and_eval(bgain, bthr, bmleft);
        std::vector<double> best_w = w;

        for (int step = 1; step < newton_steps; ++step) {
            if (!(bgain > 0.0)) break;
            
            double GL = 0.0, HL = 0.0, GR = 0.0, HR = 0.0;
            for (int rr = 0; rr < Nn; ++rr) {
                const int i = ctx.row_index[rr];
                double acc = 0.0;
                bool finite = true;
                for (size_t j = 0; j < S.size(); ++j) {
                    const int f = S[j];
                    const uint16_t code = ctx.Xb[(size_t)i * (size_t)P + (size_t)f];
                    const double x = x_from_code_variable(f, code, ctx);
                    if (!std::isfinite(x)) { finite = false; break; }
                    acc += best_w[j] * x;
                }
                bool go_left = !finite ? bmleft : (acc <= bthr);
                if (go_left) { GL += ctx.row_g[i]; HL += ctx.row_h[i]; }
                else { GR += ctx.row_g[i]; HR += ctx.row_h[i]; }
            }
            
            double denomL = HL + ctx.hyp.lambda_;
            double denomR = HR + ctx.hyp.lambda_;
            if (!(denomL > 0.0) || !(denomR > 0.0)) break;
            
            double vL = -GL / denomL;
            double vR = -GR / denomR;
            if (std::abs(vL - vR) < 1e-7) break;
            
            std::vector<double> b_new(S.size(), 0.0);
            for (int rr = 0; rr < Nn; ++rr) {
                const int i = ctx.row_index[rr];
                double acc = 0.0;
                bool finite = true;
                for (size_t j = 0; j < S.size(); ++j) {
                    const int f = S[j];
                    const uint16_t code = ctx.Xb[(size_t)i * (size_t)P + (size_t)f];
                    const double x = x_from_code_variable(f, code, ctx);
                    if (!std::isfinite(x)) { finite = false; break; }
                    acc += best_w[j] * x;
                }
                bool go_left = !finite ? bmleft : (acc <= bthr);
                double yi = go_left ? vL : vR;
                double hi_yi = ctx.row_h[i] * yi;
                
                for (size_t j = 0; j < S.size(); ++j) {
                    const int f = S[j];
                    const uint16_t code = ctx.Xb[(size_t)i * (size_t)P + (size_t)f];
                    const double x = x_from_code_variable(f, code, ctx);
                    if (std::isfinite(x)) {
                        b_new[j] += hi_yi * x;
                    }
                }
            }
            
            b_work = b_new;
            auto A_work = A;
            chol_solve_inplace(A_work, (int)S.size(), b_work, w);
            
            if (l1 > 0.0) {
                for (auto &wj : w) {
                    if (wj > l1) wj -= l1;
                    else if (wj < -l1) wj += l1;
                    else wj = 0.0;
                }
            }
            
            double new_gain = -1.0, new_thr = std::numeric_limits<double>::quiet_NaN();
            bool new_mleft = true;
            project_and_eval(new_gain, new_thr, new_mleft);
            
            if (new_gain > bgain + 1e-6) {
                bgain  = new_gain;
                bthr   = new_thr;
                bmleft = new_mleft;
                best_w = w;
            } else {
                break;
            }
        }

        if (!(bgain > 0.0)) return out;

        out.gain                 = bgain;
        out.oblique_features     = std::move(S);
        out.oblique_weights      = std::move(best_w);
        out.oblique_threshold    = bthr;
        out.oblique_bias         = 0.0;
        out.oblique_missing_left = bmleft;
        return out;
    }
};

// ============================================================================
// 4) Interaction-seeded oblique (2-feature)
// ============================================================================
struct InteractionSeededConfig {
    int    pairs              = 5;
    int    max_top_features   = 8;
    int    max_var_candidates = 16;
    int    first_i_cap        = 4;
    int    second_j_cap       = 8;
    double ridge              = 1e-3;
    double axis_guard_factor  = 1.05;
    bool   use_axis_guard     = true;
};

class InteractionSeededObliqueFinder {
public:
    InteractionSeededObliqueFinder() = default;

    splitx::Candidate best_oblique_interaction(const splitx::SplitContext &ctx, const InteractionSeededConfig &cfg,
                                               double axis_guard_gain = -1.0) const {
        using namespace splitx;
        Candidate best;
        best.kind = SplitKind::Oblique;
        best.gain = -std::numeric_limits<double>::infinity();

        if (!ctx.Xcols || !ctx.row_g || !ctx.row_h || ctx.N <= 0 || ctx.P <= 1) return best;

        const int N = ctx.N, P = ctx.P;
        const int max_var_candidates = std::clamp(cfg.max_var_candidates, 2, P);
        const int max_top_features   = std::clamp(cfg.max_top_features, 2, max_var_candidates);
        const int pair_budget        = std::max(1, cfg.pairs);
        const int i_cap_req          = std::max(1, cfg.first_i_cap);
        const int j_cap_req          = std::max(2, cfg.second_j_cap);

        std::vector<int> cand((size_t)P);
        std::iota(cand.begin(), cand.end(), 0);

        std::vector<double> var((size_t)P, 0.0);
        for (int f = 0; f < P; ++f) {
            const double *x = ctx.Xcols[f];
            int           cnt;
            double        mx, v;
            std::tie(cnt, mx, v) = col_var_ignore_nan(x, N);
            var[(size_t)f]       = (cnt < 2 ? 0.0 : v);
        }
        std::partial_sort(cand.begin(), cand.begin() + max_var_candidates, cand.end(),
                          [&](int a, int b) { return var[(size_t)a] > var[(size_t)b]; });
        cand.resize(max_var_candidates);
        if ((int)cand.size() < 2) return best;

        const auto corr_all = abs_corr_cols_ignore_nan(ctx.Xcols, N, P, ctx.row_g);
        std::sort(cand.begin(), cand.end(), [&](int a, int b) { return corr_all[(size_t)a] > corr_all[(size_t)b]; });

        const int shortlist = std::min(max_top_features, (int)cand.size());
        if (shortlist < 2) return best;

        auto abs_corr_pair_ignore_nan = [&](int fa, int fb) -> double {
            const double *xa = ctx.Xcols[fa];
            const double *xb = ctx.Xcols[fb];
            double        sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
            int           cnt = 0;
            for (int i = 0; i < N; ++i) {
                const double xi = xa[i], yi = xb[i];
                if (!splitx::is_fin(xi) || !splitx::is_fin(yi)) continue;
                sx += xi;
                sy += yi;
                sxx += xi * xi;
                syy += yi * yi;
                sxy += xi * yi;
                ++cnt;
            }
            if (cnt < 2) return 0.0;
            const double inv = 1.0 / (double)cnt;
            const double mx  = sx * inv;
            const double my  = sy * inv;
            const double cov = sxy * inv - mx * my;
            const double vx  = sxx * inv - mx * mx;
            const double vy  = syy * inv - my * my;
            const double den = std::sqrt(std::max(0.0, vx) * std::max(0.0, vy)) + 1e-12;
            return (den <= 1e-12) ? 0.0 : std::abs(cov) / den;
        };

        struct PairScore {
            int    fa    = -1;
            int    fb    = -1;
            double score = -std::numeric_limits<double>::infinity();
        };

        std::vector<PairScore> pair_pool;
        const int              i_cap = std::min(i_cap_req, shortlist - 1);
        const int              j_cap = std::min(j_cap_req, shortlist);
        pair_pool.reserve((size_t)i_cap * (size_t)std::max(1, j_cap - 1));
        for (int ii = 0; ii < i_cap; ++ii) {
            for (int jj = ii + 1; jj < j_cap; ++jj) {
                const int    fa   = cand[(size_t)ii];
                const int    fb   = cand[(size_t)jj];
                const double cg_a = corr_all[(size_t)fa];
                const double cg_b = corr_all[(size_t)fb];
                const double c_ab = abs_corr_pair_ignore_nan(fa, fb);
                // Lightweight interaction proxy:
                // |corr(x1,g)| + |corr(x2,g)| + |corr(x1,x2)|
                pair_pool.push_back(PairScore{fa, fb, cg_a + cg_b + c_ab});
            }
        }
        if (pair_pool.empty()) return best;
        const int keep = std::min(pair_budget, (int)pair_pool.size());
        std::partial_sort(pair_pool.begin(), pair_pool.begin() + keep, pair_pool.end(),
                          [](const PairScore &a, const PairScore &b) { return a.score > b.score; });

        const double ridge_plus_lambda = std::max(0.0, cfg.ridge) + ctx.hyp.lambda_;

        for (int pi = 0; pi < keep; ++pi) {
            const int     fa = pair_pool[(size_t)pi].fa;
            const int     fb = pair_pool[(size_t)pi].fb;
            const double *x1 = ctx.Xcols[fa];
            const double *x2 = ctx.Xcols[fb];

            double A00, A01, A11, b0, b1;
            int    nfinite = 0;
            splitx::build_2x2_Ab(x1, x2, ctx.row_g, ctx.row_h, N, ridge_plus_lambda, A00, A01, A11, b0, b1, nfinite);
            if (nfinite < 2) continue;

            double w0, w1;
            if (!splitx::solve_2x2(A00, A01, A11, b0, b1, w0, w1)) continue;

            std::vector<double>  z((size_t)N, 0.0);
            std::vector<uint8_t> miss((size_t)N, 0u);
            for (int i = 0; i < N; ++i) {
                const double xi = x1[i], yi = x2[i];
                const bool   finite = splitx::is_fin(xi) && splitx::is_fin(yi);
                miss[(size_t)i]     = finite ? 0u : 1u;
                if (finite) z[(size_t)i] = w0 * xi + w1 * yi;
            }

            auto [gain, thr, mleft] =
                splitx::best_split_on_projection_interact(z, miss, ctx.row_g, ctx.row_h, N, ctx.hyp);

            if (gain <= 0.0) continue;
            if (cfg.use_axis_guard && axis_guard_gain > 0.0 && axis_guard_gain * cfg.axis_guard_factor >= gain) {
                continue;
            }

            if (gain > best.gain) {
                best.kind                 = SplitKind::Oblique;
                best.gain                 = gain;
                best.oblique_features     = {fa, fb};
                best.oblique_weights      = {w0, w1};
                best.oblique_bias         = 0.0;
                best.oblique_threshold    = thr;
                best.oblique_missing_left = mleft;
            }
        }
        return best;
    }
};

} // namespace foretree
