#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace foretree {

// ============================================================================
// Split hyper-parameters (unchanged API)
// ============================================================================
struct SplitHyper {
    double lambda_          = 1.0;
    double alpha_           = 0.0;
    double gamma_           = 0.0;
    double max_delta_step_  = 0.0; // reserved for future use
    int    min_samples_leaf_ = 5;
    double min_child_weight_ = 1e-3;
    // 0=Learn, 1=AlwaysLeft, 2=AlwaysRight
    int    missing_policy  = 0; // mapped from TreeConfig::MissingPolicy
    double leaf_gain_eps   = 0.0;
    bool   allow_zero_gain = false;

    // Helper: compute lambda per class/target
    double min_child_weight() const { return min_child_weight_; }
    int min_samples_leaf() const { return min_samples_leaf_; }
    double max_delta_step() const { return max_delta_step_; }
};

// ============================================================================
// Small numerics helpers
// ============================================================================
namespace splitx {
constexpr double NEG_INF = -1.0 * std::numeric_limits<double>::infinity();
constexpr double EPS     = 1e-12;

inline double soft(double g, double alpha) {
    if (alpha <= 0.0) return g;
    if (g > alpha) return g - alpha;
    if (g < -alpha) return g + alpha;
    return 0.0;
}

inline bool pass_monotone_guard(int8_t mono, double wL, double wR) {
    // mono > 0: non-decreasing (wL <= wR), mono < 0: non-increasing (wL >= wR)
    if (mono == 0) return true;
    return mono > 0 ? (wL <= wR) : (wL >= wR);
}

inline double weight_from_GRH(double G, double H, const SplitHyper &hyp) {
    const double denom = H + hyp.lambda_;
    if (!(denom > 0.0)) return 0.0;
    return -soft(G, hyp.alpha_) / denom;
}

// ============================================================================
// Leaf objective (same as before)
// ============================================================================
inline double leaf_obj(double G, double H, double lambda, double alpha) {
    const double denom = H + lambda;
    if (!(denom > 0.0)) return 0.0;
    const double gs = soft(G, alpha);
    return 0.5 * (gs * gs) / denom;
}

inline bool valid_children(double HL, double HR, int nL, int nR, const SplitHyper &hyp) {
    if (nL < hyp.min_samples_leaf_ || nR < hyp.min_samples_leaf_) return false;
    if (HL < hyp.min_child_weight_ || HR < hyp.min_child_weight_) return false;
    return true;
}

inline double split_gain_from_parent(double parent_gain, double GL, double HL, double GR, double HR, int nL, int nR,
                                     const SplitHyper &hyp, double complexity_penalty = 0.0,
                                     double complexity_term = 0.0) {
    if (!valid_children(HL, HR, nL, nR, hyp)) return NEG_INF;

    double gain = leaf_obj(GL, HL, hyp.lambda_, hyp.alpha_) + leaf_obj(GR, HR, hyp.lambda_, hyp.alpha_) - parent_gain -
                  hyp.gamma_;
    if (complexity_penalty > 0.0 && complexity_term > 0.0) { gain -= complexity_penalty * complexity_term; }
    return gain;
}

inline double split_gain(double Gp, double Hp, double GL, double HL, int nL, int nR, const SplitHyper &hyp,
                         double complexity_penalty = 0.0, double complexity_term = 0.0) {
    const double GR          = Gp - GL;
    const double HR          = Hp - HL;
    const double parent_gain = leaf_obj(Gp, Hp, hyp.lambda_, hyp.alpha_);
    return split_gain_from_parent(parent_gain, GL, HL, GR, HR, nL, nR, hyp, complexity_penalty, complexity_term);
}

// ============================================================================
// Candidate: supports axis, k-way, oblique
// ============================================================================
enum class SplitKind : uint8_t { Axis = 0, KWay = 1, Oblique = 2 };

struct Candidate {
    // Common
    SplitKind kind = SplitKind::Axis;
    double    gain = NEG_INF;

    // --- Axis fields ---
    int    feat        = -1; // feature id
    int    thr         = -1; // threshold step/bin
    bool   miss_left   = true;
    double split_value = std::numeric_limits<double>::quiet_NaN(); // exact mode

    // --- K-way (categorical via histogram bins) ---
    std::vector<int> left_groups;        // finite bin ids that go left
    int              missing_group = -1; // which group hosts missing

    // --- Oblique ---
    std::vector<int>    oblique_features; // indices into features
    std::vector<double> oblique_weights;  // same size
    double              oblique_bias         = 0.0;
    double              oblique_threshold    = std::numeric_limits<double>::quiet_NaN();
    bool                oblique_missing_left = true;
};

// ============================================================================
// SplitContext: supports variable bin sizes per feature
// ============================================================================
struct SplitContext {
    // Histogram views (required for axis/k-way)
    const std::vector<double> *G = nullptr; // size depends on variable_bins
    const std::vector<double> *H = nullptr;
    const std::vector<int>    *C = nullptr;

    int    P  = 0;             // #features
    int    B  = 0;             // uniform #bins per feature (when !variable_bins)
    
    // Parent aggregates
    double Gp = 0.0, Hp = 0.0; // parent aggregate
    
    int    Cp = 0;

    // Monotone constraints
    const std::vector<int8_t> *monotone = nullptr;

    SplitHyper hyp;

    // Variable bin support
    bool          variable_bins        = false;
    const size_t *feature_offsets      = nullptr; // cumulative offsets [P+1]
    const int    *finite_bins_per_feat = nullptr; // finite bins per feat [P]
    const int    *missing_ids_per_feat = nullptr; // missing bin id per feat [P]

    // --- exact-mode (optional) ---
    const double *row_g = nullptr; // size N_total (or N at node)
    const double *row_h = nullptr;
    // Optional per-feature missing aggregates (for exact)
    const double *Gmiss       = nullptr; // size P
    const double *Hmiss       = nullptr;
    const int    *Cmiss       = nullptr;
    bool          has_missing = false;

    // Row-level feature access for oblique (optional)
    int                  N     = 0;       // #rows in the node
    const double *const *Xcols = nullptr; // Xcols[f] -> pointer to N doubles

    // --- binned-matrix path for oblique (histogram-based) ---
    const uint16_t *Xb          = nullptr; // prebinned matrix (row-major, N_total x P)
    const int      *row_index   = nullptr; // pointer to node's row indices (length N)
    int             miss_bin_id = -1;      // usually B-1 (uniform) or per-feature (variable)
    const double   *bin_centers = nullptr; // size varies based on variable_bins
    int             Bz          = 256;     // #bins for projection z histogram

    // Helper accessor: choose whichever monotone vector is set
    const std::vector<int8_t> *mono_ptr() const { return monotone; }

    // Variable-bin helpers
    size_t get_histogram_offset(int feature, int bin) const {
        if (variable_bins) {
            return feature_offsets[feature] + static_cast<size_t>(bin);
        } else {
            return static_cast<size_t>(feature) * static_cast<size_t>(B) + static_cast<size_t>(bin);
        }
    }
    int get_feature_bins(int feature) const { return variable_bins ? finite_bins_per_feat[feature] : (B - 1); }
    int get_missing_bin_id(int feature) const { return variable_bins ? missing_ids_per_feat[feature] : (B - 1); }

    bool   use_entropy = false; // reserved for future use
    double eps         = 1e-15; // reserved for future use
};

// ============================================================================
// Providers and generic axis scanners
// ============================================================================
struct HistProvider {
    const SplitContext        &ctx;
    int                        f;
    const std::vector<double> &G;
    const std::vector<double> &H;
    const std::vector<int>    &C;
    int                        finite_bins;
    int                        miss_id;

    HistProvider(const SplitContext &c, int feat, const std::vector<double> &g, const std::vector<double> &h,
                 const std::vector<int> &cc)
        : ctx(c), f(feat), G(g), H(h), C(cc) {
        finite_bins = ctx.get_feature_bins(f);
        miss_id     = ctx.get_missing_bin_id(f);
    }

    int  steps() const { return std::max(0, finite_bins); }
    void reset_prefix(double &GL, double &HL, int &CL) const {
        GL = HL = 0.0;
        CL      = 0;
    }
    void add_prefix(int t, double &GL, double &HL, int &CL) const {
        const size_t off = ctx.get_histogram_offset(f, t);
        if (off < G.size()) {
            GL += G[off];
            HL += H[off];
            CL += C[off];
        }
    }
    bool boundary_valid(int /*t*/) const { return true; }
    void missing(double &Gm, double &Hm, int &Cm, bool &has_miss) const {
        const size_t moff = ctx.get_histogram_offset(f, miss_id);
        Gm                = (moff < G.size()) ? G[moff] : 0.0;
        Hm                = (moff < H.size()) ? H[moff] : 0.0;
        Cm                = (moff < C.size()) ? C[moff] : 0;
        has_miss          = (Cm > 0);
    }
    int total_count() const { return ctx.Cp; }
};

struct ExactProvider {
    const SplitContext &ctx;
    const double       *Xraw;
    int                 P;
    const int          *node_idx;
    int                 nidx;
    const uint8_t      *miss_mask;

    // per-feature buffers
    std::vector<std::pair<double, int>> col;    // (value,row)
    int                                 Cm = 0; // missing count for current feature

    ExactProvider(const SplitContext &c, const double *xraw, int p, const int *node, int n, const uint8_t *mm)
        : ctx(c), Xraw(xraw), P(p), node_idx(node), nidx(n), miss_mask(mm) {}

    int prepare_feature(int f) {
        col.clear();
        Cm = 0;
        for (int ii = 0; ii < nidx; ++ii) {
            const int    r    = node_idx[ii];
            const size_t off  = static_cast<size_t>(r) * static_cast<size_t>(P) + static_cast<size_t>(f);
            const double xv   = Xraw[off];
            const bool   miss = miss_mask ? (miss_mask[off] != 0) : !std::isfinite(xv);
            if (miss)
                ++Cm;
            else
                col.emplace_back(xv, r);
        }
        if ((int)col.size() < 2) return 0;
        std::sort(col.begin(), col.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
        return (int)col.size();
    }
    int  steps_for_nvalid(int n_valid) const { return std::max(0, n_valid - 1); }
    void reset_prefix(double &GL, double &HL, int &CL) const {
        GL = HL = 0.0;
        CL      = 0;
    }
    void add_prefix(int t, double &GL, double &HL, int &CL) const {
        const int r = col[(size_t)t].second;
        GL += (double)ctx.row_g[r];
        HL += (double)ctx.row_h[r];
        ++CL;
    }
    bool boundary_valid(int t) const {
        const double v  = col[(size_t)t].first;
        const double vp = col[(size_t)t + 1].first;
        return (v < vp);
    }
    void missing(double &Gm, double &Hm, int &CmOut, bool &has_miss) const {
        Gm       = 0.0;
        Hm       = 0.0;
        has_miss = (ctx.has_missing && Cm > 0);
        CmOut    = Cm;
    }
    int total_count_for_nvalid(int n_valid) const { return n_valid + Cm; }
};

// Generic axis scanner core
template <class Provider>
inline Candidate scan_axis_core(const SplitContext &ctx, int f, const Provider &prov, int steps, int8_t mono,
                                bool miss_left, double parent_gain, int totalC, double Gm, double Hm, int Cm,
                                bool has_miss) {
    Candidate cand;
    cand.kind      = SplitKind::Axis;
    cand.feat      = f;
    cand.thr       = -1;
    cand.miss_left = miss_left;
    cand.gain      = NEG_INF;

    double GL = 0.0, HL = 0.0;
    int    CL = 0;
    prov.reset_prefix(GL, HL, CL);

    for (int t = 0; t < steps; ++t) {
        prov.add_prefix(t, GL, HL, CL);
        if (!prov.boundary_valid(t)) continue;

        const double GLx = GL + ((miss_left && has_miss) ? Gm : 0.0);
        const double HLx = HL + ((miss_left && has_miss) ? Hm : 0.0);
        const int    CLx = CL + ((miss_left && has_miss) ? Cm : 0);

        const double GRx = (ctx.Gp - GLx);
        const double HRx = (ctx.Hp - HLx);
        const int    CRx = totalC - CLx;

        if (!valid_children(HLx, HRx, CLx, CRx, ctx.hyp)) continue;

        if (mono != 0) {
            const double wL = weight_from_GRH(GLx, HLx, ctx.hyp);
            const double wR = weight_from_GRH(GRx, HRx, ctx.hyp);
            if (!pass_monotone_guard(mono, wL, wR)) continue;
        }

        const double gain = split_gain_from_parent(parent_gain, GLx, HLx, GRx, HRx, CLx, CRx, ctx.hyp);
        if (gain > cand.gain) {
            cand.gain = gain;
            cand.thr  = t;
        }
    }
    return cand;
}

// Try missing→left/right per policy and return best
template <class Provider>
inline Candidate scan_axis_with_policy(const SplitContext &ctx, int f, const Provider &prov, int steps, int8_t mono,
                                       int missing_policy, double parent_gain, int totalC, double Gm, double Hm, int Cm,
                                       bool has_miss) {
    if (missing_policy == 1) {
        return scan_axis_core(ctx, f, prov, steps, mono,
                              /*miss_left=*/true, parent_gain, totalC, Gm, Hm, Cm, has_miss);
    }
    if (missing_policy == 2) {
        return scan_axis_core(ctx, f, prov, steps, mono,
                              /*miss_left=*/false, parent_gain, totalC, Gm, Hm, Cm, has_miss);
    }
    Candidate cL = scan_axis_core(ctx, f, prov, steps, mono,
                                  /*miss_left=*/true, parent_gain, totalC, Gm, Hm, Cm, has_miss);
    Candidate cR = scan_axis_core(ctx, f, prov, steps, mono,
                                  /*miss_left=*/false, parent_gain, totalC, Gm, Hm, Cm, has_miss);
    return (cL.gain >= cR.gain) ? cL : cR;
}

// ============================================================================
// Helpers for oblique/categorical
// ============================================================================
inline double x_from_code_variable(int f, uint16_t code, const SplitContext &ctx) {
    if (ctx.variable_bins) {
        const int finite_bins = ctx.get_feature_bins(f);
        const int miss_id     = ctx.get_missing_bin_id(f);
        if (code == static_cast<uint16_t>(miss_id)) return std::numeric_limits<double>::quiet_NaN();
        if (code >= static_cast<uint16_t>(finite_bins)) return std::numeric_limits<double>::quiet_NaN();
        const size_t center_offset = ctx.feature_offsets[f] + static_cast<size_t>(code);
        return ctx.bin_centers[center_offset];
    } else {
        if (code == static_cast<uint16_t>(ctx.B - 1)) return std::numeric_limits<double>::quiet_NaN();
        return ctx.bin_centers[static_cast<size_t>(f) * static_cast<size_t>(ctx.B) + static_cast<size_t>(code)];
    }
}

// Cholesky + solves
inline bool cholesky_spd(std::vector<double> &A, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[i * n + j];
            for (int k = 0; k < j; ++k) sum -= A[i * n + k] * A[j * n + k];
            if (i == j) {
                if (sum <= 0.0) return false;
                A[i * n + j] = std::sqrt(sum);
            } else {
                A[i * n + j] = sum / A[j * n + j];
            }
        }
        for (int j = i + 1; j < n; ++j) A[i * n + j] = 0.0;
    }
    return true;
}

inline void chol_solve_inplace(const std::vector<double> &L, int n, const std::vector<double> &b,
                               std::vector<double> &x) {
    x = b;
    for (int i = 0; i < n; ++i) {
        double s = x[i];
        for (int k = 0; k < i; ++k) s -= L[i * n + k] * x[k];
        x[i] = s / L[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = x[i];
        for (int k = i + 1; k < n; ++k) s -= L[k * n + i] * x[k];
        x[i] = s / L[i * n + i];
    }
}

inline void build_normal_eq_cols(const std::vector<const double *> &XS, const double *g, const double *h, int N,
                                 double ridge_plus_lambda, std::vector<double> &A, std::vector<double> &b) {
    const int k = (int)XS.size();
    A.assign((size_t)k * (size_t)k, 0.0);
    b.assign((size_t)k, 0.0);

    for (int i = 0; i < N; ++i) {
        const double gi = (double)g[i];
        for (int c = 0; c < k; ++c) {
            const double xic = XS[c][i];
            if (std::isfinite(xic)) b[(size_t)c] -= xic * gi;
        }
    }
    for (int i = 0; i < N; ++i) {
        const double hi = (double)h[i];
        for (int r = 0; r < k; ++r) {
            const double xir = XS[r][i];
            if (!std::isfinite(xir)) continue;
            const double hrx = hi * xir;
            for (int c = r; c < k; ++c) {
                const double xic = XS[c][i];
                if (std::isfinite(xic)) A[(size_t)r * (size_t)k + (size_t)c] += hrx * xic;
            }
        }
    }
    for (int r = 0; r < k; ++r) {
        for (int c = r + 1; c < k; ++c) A[(size_t)c * (size_t)k + (size_t)r] = A[(size_t)r * (size_t)k + (size_t)c];
        A[(size_t)r * (size_t)k + (size_t)r] += ridge_plus_lambda;
    }
}

inline void build_normal_eq_from_codes(const std::vector<int> &S, const uint16_t *Xb, const int *rows, int nrows, int P,
                                       const SplitContext &ctx, const double *g, const double *h,
                                       double ridge_plus_lambda, std::vector<double> &A, std::vector<double> &b) {
    const int k = (int)S.size();
    A.assign((size_t)k * (size_t)k, 0.0);
    b.assign((size_t)k, 0.0);
    std::vector<double> xi((size_t)k, 0.0);

    for (int rr = 0; rr < nrows; ++rr) {
        const int i    = rows[rr];
        bool      miss = false;
        for (int c = 0; c < k; ++c) {
            const int      f     = S[c];
            const uint16_t code  = Xb[(size_t)i * (size_t)P + (size_t)f];
            const double   x_val = x_from_code_variable(f, code, ctx);
            if (!std::isfinite(x_val)) {
                miss = true;
                break;
            }
            xi[(size_t)c] = x_val;
        }
        if (miss) continue;

        const double gi = (double)g[i];
        for (int c = 0; c < k; ++c) b[(size_t)c] -= xi[(size_t)c] * gi;

        const double hi = (double)h[i];
        for (int r = 0; r < k; ++r) {
            const double hrx = hi * xi[(size_t)r];
            for (int c = r; c < k; ++c) A[(size_t)r * (size_t)k + (size_t)c] += hrx * xi[(size_t)c];
        }
    }
    for (int r = 0; r < k; ++r) {
        for (int c = r + 1; c < k; ++c) A[(size_t)c * (size_t)k + (size_t)r] = A[(size_t)r * (size_t)k + (size_t)c];
        A[(size_t)r * (size_t)k + (size_t)r] += ridge_plus_lambda;
    }
}

// Best split along projection z = w^T x with missing-policy search
inline std::tuple<double, double, bool> best_split_on_projection(const std::vector<double> &z, const double *g,
                                                                 const double *h, int N,
                                                                 const std::vector<uint8_t> &miss_mask,
                                                                 const SplitHyper           &hyp) {
    std::vector<int> idx;
    idx.reserve(N);
    double gm = 0.0, hm = 0.0;
    for (int i = 0; i < N; ++i) {
        if (miss_mask[i]) {
            gm += (double)g[i];
            hm += (double)h[i];
        } else
            idx.push_back(i);
    }
    const int nf = (int)idx.size();
    if (nf < 4) return {NEG_INF, 0.0, true};

    std::sort(idx.begin(), idx.end(), [&](int i, int j) { return z[i] < z[j]; });

    std::vector<double> pg(nf), ph(nf);
    for (int k = 0; k < nf; ++k) {
        const int i = idx[k];
        pg[k]       = (double)g[i] + (k ? pg[k - 1] : 0.0);
        ph[k]       = (double)h[i] + (k ? ph[k - 1] : 0.0);
    }

    const double gtot   = pg[nf - 1] + gm;
    const double htot   = ph[nf - 1] + hm;
    const double parent = (gtot * gtot) / (htot + hyp.lambda_);

    double best_gain = NEG_INF, best_thr = 0.0;
    bool   best_mleft = true;

    for (int k = 1; k < nf; ++k) {
        const int iL = idx[k - 1], iR = idx[k];
        if (!(z[iR] > z[iL] + 1e-15)) continue;
        const double thr = 0.5 * (z[iL] + z[iR]);

        const double glb = pg[k - 1];
        const double hlb = ph[k - 1];
        const double grb = pg[nf - 1] - glb;
        const double hrb = ph[nf - 1] - hlb;

        auto eval = [&](bool mleft) -> double {
            const double HL = hlb + (mleft ? hm : 0.0);
            const double HR = hrb + (mleft ? 0.0 : hm);
            if (HL < hyp.min_child_weight() || HR < hyp.min_child_weight()) return NEG_INF;
            const double GL    = glb + (mleft ? gm : 0.0);
            const double GR    = grb + (mleft ? 0.0 : gm);
            const double child = (GL * GL) / (HL + hyp.lambda_) + (GR * GR) / (HR + hyp.lambda_);
            return 0.5 * (child - parent) - hyp.gamma_;
        };

        const double gL = eval(true);
        const double gR = eval(false);
        if (gL > best_gain) {
            best_gain  = gL;
            best_thr   = thr;
            best_mleft = true;
        }
        if (gR > best_gain) {
            best_gain  = gR;
            best_thr   = thr;
            best_mleft = false;
        }
    }
    return {best_gain, best_thr, best_mleft};
}

// Build 1-D histogram on z = w^T x using variable bin centers
inline void build_projection_hist_from_codes(const std::vector<int> &S, const std::vector<double> &w,
                                             const uint16_t *Xb, const int *rows, int nrows, int P,
                                             const SplitContext &ctx, const double *g, const double *h,
                                             std::vector<double> &Gz, std::vector<double> &Hz, std::vector<int> &Cz,
                                             double &Gm, double &Hm, int &Cm, double &zmin, double &zmax,
                                             std::vector<double> *z_edges = nullptr, bool use_quantile_bins = true,
                                             double clip_quantile = 0.01) {
    Gm = Hm      = 0.0;
    Cm           = 0;
    const int Bz = (ctx.Bz > 0) ? ctx.Bz : 256;

    if (nrows <= 0 || S.empty()) {
        Gz.assign(Bz, 0.0);
        Hz.assign(Bz, 0.0);
        Cz.assign(Bz, 0);
        zmin = 0.0;
        zmax = 1.0;
        return;
    }

    std::vector<double>  zbuf((size_t)nrows, 0.0);
    std::vector<uint8_t> finite((size_t)nrows, 1u);
    std::vector<double>  zvals;
    zvals.reserve((size_t)nrows);
    zmin = std::numeric_limits<double>::infinity();
    zmax = -std::numeric_limits<double>::infinity();

    for (int rr = 0; rr < nrows; ++rr) {
        const int i    = rows[rr];
        bool      miss = false;
        double    acc  = 0.0;
        for (size_t j = 0; j < S.size(); ++j) {
            const int      f     = S[j];
            const uint16_t code  = Xb[(size_t)i * (size_t)P + (size_t)f];
            const double   x_val = x_from_code_variable(f, code, ctx);
            if (!std::isfinite(x_val)) {
                miss = true;
                break;
            }
            acc += w[j] * x_val;
        }
        if (miss) {
            finite[(size_t)rr] = 0u;
            Gm += (double)g[i];
            Hm += (double)h[i];
            ++Cm;
        } else {
            zbuf[(size_t)rr] = acc;
            zmin             = std::min(zmin, acc);
            zmax             = std::max(zmax, acc);
            zvals.push_back(acc);
        }
    }

    if (!(zmax > zmin) || zvals.size() < 2) {
        Gz.assign(Bz, 0.0);
        Hz.assign(Bz, 0.0);
        Cz.assign(Bz, 0);
        if (z_edges) {
            z_edges->clear();
            z_edges->push_back(zmin);
            z_edges->push_back(zmax);
        }
        return;
    }

    // Adaptive option: quantile bins for z to reduce sensitivity to outliers / skew.
    if (use_quantile_bins) {
        std::vector<double> sorted = zvals;
        std::sort(sorted.begin(), sorted.end());
        const int nf          = (int)sorted.size();
        const int target_bins = std::max(2, std::min(Bz, nf));

        std::vector<double> edges;
        edges.reserve((size_t)target_bins + 1);
        for (int q = 0; q <= target_bins; ++q) {
            const double pos = ((double)q / (double)target_bins) * (double)(nf - 1);
            const int    lo  = (int)std::floor(pos);
            const int    hi  = (int)std::ceil(pos);
            const double t   = pos - (double)lo;
            const double e =
                (lo == hi) ? sorted[(size_t)lo] : (sorted[(size_t)lo] + t * (sorted[(size_t)hi] - sorted[(size_t)lo]));
            edges.push_back(e);
        }

        // Merge duplicate/near-duplicate quantile boundaries.
        const double        span   = std::max(1.0, std::abs(sorted.back() - sorted.front()));
        const double        min_dz = 1e-12 * span;
        std::vector<double> uniq;
        uniq.reserve(edges.size());
        for (double e : edges) {
            if (uniq.empty() || e > uniq.back() + min_dz) uniq.push_back(e);
        }

        if (uniq.size() >= 2) {
            zmin         = uniq.front();
            zmax         = uniq.back();
            const int qb = (int)uniq.size() - 1;
            Gz.assign((size_t)qb, 0.0);
            Hz.assign((size_t)qb, 0.0);
            Cz.assign((size_t)qb, 0);

            for (int rr = 0; rr < nrows; ++rr) {
                if (!finite[(size_t)rr]) continue;
                const int    i  = rows[rr];
                const double z  = zbuf[(size_t)rr];
                auto         it = std::upper_bound(uniq.begin() + 1, uniq.end() - 1, z);
                int          bz = (int)std::distance(uniq.begin(), it) - 1;
                if (bz < 0)
                    bz = 0;
                else if (bz >= qb)
                    bz = qb - 1;
                Gz[(size_t)bz] += (double)g[i];
                Hz[(size_t)bz] += (double)h[i];
                Cz[(size_t)bz] += 1;
            }
            if (z_edges) *z_edges = std::move(uniq);
            return;
        }
    }

    // Fallback: uniform bins over clipped z-range (robust to extreme outliers).
    {
        std::vector<double> sorted = zvals;
        std::sort(sorted.begin(), sorted.end());
        const int    nf = (int)sorted.size();
        const double cq = std::clamp(clip_quantile, 0.0, 0.49);
        if (cq > 0.0 && nf >= 8) {
            const int    lo  = std::clamp((int)std::floor(cq * (nf - 1)), 0, nf - 1);
            const int    hi  = std::clamp((int)std::ceil((1.0 - cq) * (nf - 1)), 0, nf - 1);
            const double zlo = sorted[(size_t)lo];
            const double zhi = sorted[(size_t)hi];
            if (zhi > zlo) {
                zmin = zlo;
                zmax = zhi;
            }
        }

        if (!(zmax > zmin)) {
            zmin = sorted.front();
            zmax = sorted.back();
        }

        Gz.assign((size_t)Bz, 0.0);
        Hz.assign((size_t)Bz, 0.0);
        Cz.assign((size_t)Bz, 0);
        const double invw = (double)Bz / (zmax - zmin);

        for (int rr = 0; rr < nrows; ++rr) {
            if (!finite[(size_t)rr]) continue;
            const int    i  = rows[rr];
            const double z  = zbuf[(size_t)rr];
            int          bz = (int)std::floor((z - zmin) * invw);
            if (bz < 0)
                bz = 0;
            else if (bz >= Bz)
                bz = Bz - 1;
            Gz[(size_t)bz] += (double)g[i];
            Hz[(size_t)bz] += (double)h[i];
            Cz[(size_t)bz] += 1;
        }
        if (z_edges) {
            z_edges->resize((size_t)Bz + 1);
            const double dz = (zmax - zmin) / (double)Bz;
            for (int b = 0; b <= Bz; ++b) (*z_edges)[(size_t)b] = zmin + dz * (double)b;
        }
    }
}

// -------------------- Interaction helpers --------------------
inline bool is_fin(double x) { return std::isfinite(x); }

inline std::tuple<int, double, double> col_var_ignore_nan(const double *x, int n) {
    int    cnt = 0;
    double sx  = 0.0;
    for (int i = 0; i < n; ++i) {
        const double xi = x[i];
        if (is_fin(xi)) {
            ++cnt;
            sx += xi;
        }
    }
    if (cnt < 2) return {cnt, 0.0, 0.0};
    const double mx  = sx / (double)cnt;
    double       sxx = 0.0;
    for (int i = 0; i < n; ++i) {
        const double xi = x[i];
        if (!is_fin(xi)) continue;
        const double dx = xi - mx;
        sxx += dx * dx;
    }
    const double var = sxx / (double)cnt;
    return {cnt, mx, var};
}

inline std::vector<double> abs_corr_cols_ignore_nan(const double *const *Xcols, int N, int P, const double *g) {
    std::vector<double> out((size_t)P, 0.0);

    double sg = 0.0;
    for (int i = 0; i < N; ++i) sg += (double)g[i];
    const double mg = (N > 0 ? sg / (double)N : 0.0);

    std::vector<int>    cnt((size_t)P, 0);
    std::vector<double> sx((size_t)P, 0.0);
    for (int j = 0; j < P; ++j) {
        const double *x = Xcols[j];
        int           c = 0;
        double        s = 0.0;
        for (int i = 0; i < N; ++i) {
            const double xi = x[i];
            if (is_fin(xi)) {
                ++c;
                s += xi;
            }
        }
        cnt[(size_t)j] = c;
        sx[(size_t)j]  = s;
    }

    for (int j = 0; j < P; ++j) {
        const int c = cnt[(size_t)j];
        if (c < 2) {
            out[(size_t)j] = 0.0;
            continue;
        }
        const double *x   = Xcols[j];
        const double  mx  = sx[(size_t)j] / (double)c;
        double        sxx = 0.0, sgg = 0.0, sxg = 0.0;
        for (int i = 0; i < N; ++i) {
            const double xi = x[i];
            if (!is_fin(xi)) continue;
            const double dx = xi - mx;
            const double dg = (double)g[i] - mg;
            sxx += dx * dx;
            sgg += dg * dg;
            sxg += dx * dg;
        }
        const double denom = std::sqrt(sxx * sgg) + 1e-12;
        out[(size_t)j]     = (denom < 1e-12 ? 0.0 : std::abs(sxg / denom));
    }
    return out;
}

inline void build_2x2_Ab(const double *x1, const double *x2, const double *g, const double *h, int N,
                         double ridge_plus_lambda, double &A00, double &A01, double &A11, double &b0, double &b1,
                         int &n_finite_rows) {
    A00 = A01 = A11 = 0.0;
    b0 = b1       = 0.0;
    n_finite_rows = 0;
    for (int i = 0; i < N; ++i) {
        const double xi = x1[i], yi = x2[i];
        // Keep row eligibility consistent with projection split: require both finite.
        if (!is_fin(xi) || !is_fin(yi)) continue;
        const double gi = (double)g[i], hi = (double)h[i];
        b0 -= xi * gi;
        A00 += hi * xi * xi;
        b1 -= yi * gi;
        A11 += hi * yi * yi;
        A01 += hi * xi * yi;
        ++n_finite_rows;
    }
    A00 += ridge_plus_lambda;
    A11 += ridge_plus_lambda;
}

inline bool solve_2x2(double A00, double A01, double A11, double b0, double b1, double &w0, double &w1) {
    const double det = A00 * A11 - A01 * A01;
    if (!(std::abs(det) > 1e-18)) return false;
    const double inv = 1.0 / det;
    w0               = (A11 * b0 - A01 * b1) * inv;
    w1               = (-A01 * b0 + A00 * b1) * inv;
    return std::isfinite(w0) && std::isfinite(w1);
}

inline std::tuple<double, double, bool> best_split_on_projection_interact(const std::vector<double>  &z,
                                                                          const std::vector<uint8_t> &miss,
                                                                          const double *g, const double *h, int N,
                                                                          const SplitHyper &hyp) {
    std::vector<int> idx;
    idx.reserve((size_t)N);
    double gm = 0.0, hm = 0.0;
    int    Cm = 0;
    for (int i = 0; i < N; ++i) {
        if (miss[(size_t)i]) {
            gm += (double)g[i];
            hm += (double)h[i];
            ++Cm;
        } else
            idx.push_back(i);
    }
    const int nf = (int)idx.size();
    if (nf < 4) return {-1.0, 0.0, true};

    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return z[(size_t)a] < z[(size_t)b]; });

    std::vector<double> pg((size_t)nf), ph((size_t)nf);
    for (int k = 0; k < nf; ++k) {
        const int i   = idx[(size_t)k];
        pg[(size_t)k] = (double)g[i] + (k ? pg[(size_t)k - 1] : 0.0);
        ph[(size_t)k] = (double)h[i] + (k ? ph[(size_t)k - 1] : 0.0);
    }

    const double gtot   = pg[(size_t)nf - 1] + gm;
    const double htot   = ph[(size_t)nf - 1] + hm;
    const double parent = (gtot * gtot) / (htot + hyp.lambda_);

    double best_gain = -1.0, best_thr = 0.0;
    bool   best_mleft = true;

    for (int k = 1; k < nf; ++k) {
        const int    il = idx[(size_t)k - 1], ir = idx[(size_t)k];
        const double zl = z[(size_t)il], zr = z[(size_t)ir];
        if (!(zr > zl + 1e-15)) continue;

        const double thr = 0.5 * (zl + zr);
        const double glb = pg[(size_t)k - 1], hlb = ph[(size_t)k - 1];
        const double grb = pg[(size_t)nf - 1] - glb;
        const double hrb = ph[(size_t)nf - 1] - hlb;

        auto eval_dir = [&](bool miss_left) {
            const double HL = hlb + (miss_left ? hm : 0.0);
            const double HR = hrb + (miss_left ? 0.0 : hm);
            const int    nL = k + (miss_left ? Cm : 0);
            const int    nR = (nf + Cm) - nL;
            if (nL < hyp.min_samples_leaf() || nR < hyp.min_samples_leaf()) return -std::numeric_limits<double>::infinity();
            if (HL < hyp.min_child_weight() || HR < hyp.min_child_weight()) return -std::numeric_limits<double>::infinity();
            const double GL    = glb + (miss_left ? gm : 0.0);
            const double GR    = grb + (miss_left ? 0.0 : gm);
            const double child = (GL * GL) / (HL + hyp.lambda_) + (GR * GR) / (HR + hyp.lambda_);
            return 0.5 * (child - parent) - hyp.gamma_;
        };

        const double gL = eval_dir(true);
        const double gR = eval_dir(false);
        if (gL > best_gain) {
            best_gain  = gL;
            best_thr   = thr;
            best_mleft = true;
        }
        if (gR > best_gain) {
            best_gain  = gR;
            best_thr   = thr;
            best_mleft = false;
        }
    }
    return {best_gain, best_thr, best_mleft};
}

} // namespace splitx
} // namespace foretree
