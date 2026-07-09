#pragma once

#include "foretree/split/split_helpers.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace foretree::splitaux {

using SplitHyper   = ::foretree::SplitHyper;
using SplitContext = ::foretree::splitx::SplitContext;

#define x_from_code_variable splitaux_x_from_code_variable

// ============================================================================
// Small numerics helpers
// ============================================================================
namespace detail {
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
} // namespace detail

// ============================================================================
// Helper function for variable bin centers in oblique splitting
// ============================================================================
inline double x_from_code_variable(int f, uint16_t code, const SplitContext &ctx) {
    if (ctx.variable_bins) {
        const int finite_bins = ctx.get_feature_bins(f);
        const int miss_id     = ctx.get_missing_bin_id(f);

        if (code == static_cast<uint16_t>(miss_id)) { return std::numeric_limits<double>::quiet_NaN(); }
        if (code >= static_cast<uint16_t>(finite_bins)) { return std::numeric_limits<double>::quiet_NaN(); }

        const size_t center_offset = ctx.feature_offsets[f] + static_cast<size_t>(code);
        return ctx.bin_centers[center_offset];
    } else {
        if (code == static_cast<uint16_t>(ctx.B - 1)) { return std::numeric_limits<double>::quiet_NaN(); }
        return ctx.bin_centers[static_cast<size_t>(f) * static_cast<size_t>(ctx.B) + static_cast<size_t>(code)];
    }
}

// ============================================================================
// Tiny linear-algebra helpers for oblique (Cholesky SPD)
// ============================================================================
inline bool cholesky_spd(std::vector<double> &A, int n) {
    // A is row-major, overwritten with lower-tri L
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
        for (int j = i + 1; j < n; ++j) A[i * n + j] = 0.0; // zero upper
    }
    return true;
}

inline void chol_solve_inplace(const std::vector<double> &L, int n, const std::vector<double> &b,
                               std::vector<double> &x) {
    x = b;
    // forward: L y = b
    for (int i = 0; i < n; ++i) {
        double s = x[i];
        for (int k = 0; k < i; ++k) s -= L[i * n + k] * x[k];
        x[i] = s / L[i * n + i];
    }
    // backward: L^T x = y
    for (int i = n - 1; i >= 0; --i) {
        double s = x[i];
        for (int k = i + 1; k < n; ++k) s -= L[k * n + i] * x[k];
        x[i] = s / L[i * n + i];
    }
}

// Build normal equations (X^T diag(h) X + (ridge+λ)I) w = - X^T g
// XS is a list of column pointers for selected features
inline void build_normal_eq_cols(const std::vector<const double *> &XS, const double *g, const double *h, int N,
                                 double ridge_plus_lambda, std::vector<double> &A, std::vector<double> &b) {
    const int k = (int)XS.size();
    A.assign((size_t)k * (size_t)k, 0.0);
    b.assign((size_t)k, 0.0);

    // b = - X^T g
    for (int i = 0; i < N; ++i) {
        const double gi = (double)g[i];
        for (int c = 0; c < k; ++c) {
            const double xic = XS[c][i];
            if (std::isfinite(xic)) b[(size_t)c] -= xic * gi;
        }
    }

    // A = X^T diag(h) X  (upper triangle)
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
    // symmetrize + ridge
    for (int r = 0; r < k; ++r) {
        for (int c = r + 1; c < k; ++c) A[(size_t)c * (size_t)k + (size_t)r] = A[(size_t)r * (size_t)k + (size_t)c];
        A[(size_t)r * (size_t)k + (size_t)r] += ridge_plus_lambda;
    }
}

// ============================================================================
// UPDATED: Build normal equations from binned matrix using variable bin centers
// ============================================================================
inline void build_normal_eq_from_codes(const std::vector<int> &S, const uint16_t *Xb, const int *rows, int nrows, int P,
                                       const SplitContext &ctx, const double *g, const double *h,
                                       double ridge_plus_lambda, std::vector<double> &A, std::vector<double> &b) {
    const int k = (int)S.size();
    A.assign((size_t)k * (size_t)k, 0.0);
    b.assign((size_t)k, 0.0);
    std::vector<double> xi((size_t)k, 0.0);

    // b = - X^T g  and A = X^T diag(h) X (upper)
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
    // symmetrize + ridge
    for (int r = 0; r < k; ++r) {
        for (int c = r + 1; c < k; ++c) A[(size_t)c * (size_t)k + (size_t)r] = A[(size_t)r * (size_t)k + (size_t)c];
        A[(size_t)r * (size_t)k + (size_t)r] += ridge_plus_lambda;
    }
}

// ============================================================================
// Best split along projection z = w^T x with missing-policy search
// ============================================================================
inline std::tuple<double, double, bool> best_split_on_projection(const std::vector<double> &z, const double *g,
                                                                 const double *h, int N,
                                                                 const std::vector<uint8_t> &miss_mask,
                                                                 const SplitHyper           &hyp) {
    // Collect finite indices & missing aggregates
    std::vector<int> idx;
    idx.reserve(N);
    double gm = 0.0, hm = 0.0;
    for (int i = 0; i < N; ++i) {
        if (miss_mask[i]) {
            gm += (double)g[i];
            hm += (double)h[i];
        } else {
            idx.push_back(i);
        }
    }
    const int nf = (int)idx.size();
    if (nf < 4) return {detail::NEG_INF, 0.0, true};

    // sort by z
    std::sort(idx.begin(), idx.end(), [&](int i, int j) { return z[i] < z[j]; });

    // prefix on sorted finite
    std::vector<double> pg(nf), ph(nf);
    for (int k = 0; k < nf; ++k) {
        const int i = idx[k];
        pg[k]       = (double)g[i] + (k ? pg[k - 1] : 0.0);
        ph[k]       = (double)h[i] + (k ? ph[k - 1] : 0.0);
    }

    const double gtot   = pg[nf - 1] + gm;
    const double htot   = ph[nf - 1] + hm;
    const double parent = splitx::leaf_obj(gtot, htot, hyp.lambda_, hyp.alpha_);

    double best_gain = detail::NEG_INF, best_thr = 0.0;
    bool   best_mleft = true;

    for (int k = 1; k < nf; ++k) {
        const int iL = idx[k - 1], iR = idx[k];
        if (!(z[iR] > z[iL] + 1e-15)) continue;
        const double thr = 0.5 * (z[iL] + z[iR]);

        // finite L/R
        const double glb = pg[k - 1];
        const double hlb = ph[k - 1];
        const double grb = pg[nf - 1] - glb;
        const double hrb = ph[nf - 1] - hlb;

        // missing -> left
        {
            const double HL = hlb + hm, HR = hrb;
            const double GL = glb + gm, GR = grb;
            const int    nL   = k + Cm;
            const int    nR   = nf - k;
            const double gain = splitx::split_gain_from_parent(parent, GL, HL, GR, HR, nL, nR, hyp);
            if (gain > best_gain) {
                best_gain  = gain;
                best_thr   = thr;
                best_mleft = true;
            }
        }
        // missing -> right
        {
            const double HL = hlb, HR = hrb + hm;
            const double GL = glb, GR = grb + gm;
            const int    nL   = k;
            const int    nR   = (nf - k) + Cm;
            const double gain = splitx::split_gain_from_parent(parent, GL, HL, GR, HR, nL, nR, hyp);
            if (gain > best_gain) {
                best_gain  = gain;
                best_thr   = thr;
                best_mleft = false;
            }
        }
    }
    return {best_gain, best_thr, best_mleft};
}

// ============================================================================
// UPDATED: Build 1-D histogram on z = w^T x using variable bin centers
// ============================================================================
inline void build_projection_hist_from_codes(const std::vector<int> &S, const std::vector<double> &w,
                                             const uint16_t *Xb, const int *rows, int nrows, int P,
                                             const SplitContext &ctx, const double *g, const double *h,
                                             std::vector<double> &Gz, std::vector<double> &Hz, std::vector<int> &Cz,
                                             double &Gm, double &Hm, int &Cm, double &zmin, double &zmax) {
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

    // pass 1: compute z, find zmin/zmax
    std::vector<double>  zbuf((size_t)nrows, 0.0);
    std::vector<uint8_t> finite((size_t)nrows, 1u);
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
        }
    }

    if (!(zmax > zmin)) {
        Gz.assign(Bz, 0.0);
        Hz.assign(Bz, 0.0);
        Cz.assign(Bz, 0);
        return;
    }

    // pass 2: bin finite z and accumulate G/H
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
}

namespace detail {

inline bool is_fin(double x) { return std::isfinite(x); }

// variance of a column ignoring NaNs; returns {count, mean, var}
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

// |corr(x,g)| per column, ignoring NaNs in x
inline std::vector<double> abs_corr_cols_ignore_nan(const double *const *Xcols, int N, int P, const double *g) {
    std::vector<double> out((size_t)P, 0.0);

    // global mean of g
    double sg = 0.0;
    for (int i = 0; i < N; ++i) sg += (double)g[i];
    const double mg = (N > 0 ? sg / (double)N : 0.0);

    // first pass: counts and means
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

    // second pass: corr components
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

// Build 2x2 normal equations (interaction-seeded)
inline void build_2x2_Ab(const double *x1, const double *x2, const double *g, const double *h, int N,
                         double ridge_plus_lambda, double &A00, double &A01, double &A11, double &b0, double &b1,
                         int &n_finite_rows) {
    A00 = A01 = A11 = 0.0;
    b0 = b1       = 0.0;
    n_finite_rows = 0;
    for (int i = 0; i < N; ++i) {
        const double xi = x1[i], yi = x2[i];
        if (!is_fin(xi) && !is_fin(yi)) continue; // both missing => skip
        const double gi = (double)g[i], hi = (double)h[i];
        if (is_fin(xi)) {
            b0 -= xi * gi;
            A00 += hi * xi * xi;
        }
        if (is_fin(yi)) {
            b1 -= yi * gi;
            A11 += hi * yi * yi;
        }
        if (is_fin(xi) && is_fin(yi)) {
            A01 += hi * xi * yi;
            ++n_finite_rows;
        }
    }
    A00 += ridge_plus_lambda;
    A11 += ridge_plus_lambda;
}

// Solve 2x2 system
inline bool solve_2x2(double A00, double A01, double A11, double b0, double b1, double &w0, double &w1) {
    const double det = A00 * A11 - A01 * A01;
    if (!(std::abs(det) > 1e-18)) return false;
    const double inv = 1.0 / det;
    w0               = (A11 * b0 - A01 * b1) * inv;
    w1               = (-A01 * b0 + A00 * b1) * inv;
    return std::isfinite(w0) && std::isfinite(w1);
}

// Best split along z with missing→left/right (interaction path)
inline std::tuple<double, double, bool> best_split_on_projection(const std::vector<double>  &z,
                                                                 const std::vector<uint8_t> &miss, const double *g,
                                                                 const double *h, int N, const SplitHyper &hyp) {
    // Collect finite slice and missing aggregates
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

    // sort by z (finite only)
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return z[(size_t)a] < z[(size_t)b]; });

    // prefix sums on sorted finite
    std::vector<double> pg((size_t)nf), ph((size_t)nf);
    for (int k = 0; k < nf; ++k) {
        const int i   = idx[(size_t)k];
        pg[(size_t)k] = (double)g[i] + (k ? pg[(size_t)k - 1] : 0.0);
        ph[(size_t)k] = (double)h[i] + (k ? ph[(size_t)k - 1] : 0.0);
    }

    const double gtot   = pg[(size_t)nf - 1] + gm;
    const double htot   = ph[(size_t)nf - 1] + hm;
    const double parent = splitx::leaf_obj(gtot, htot, hyp.lambda_, hyp.alpha_);

    double best_gain = -1.0, best_thr = 0.0;
    bool   best_mleft = true;

    for (int k = 1; k < nf; ++k) {
        const int    il = idx[(size_t)k - 1], ir = idx[(size_t)k];
        const double zl = z[(size_t)il], zr = z[(size_t)ir];
        if (!(zr > zl + 1e-15)) continue; // tie

        const double thr = 0.5 * (zl + zr);
        const double glb = pg[(size_t)k - 1], hlb = ph[(size_t)k - 1];
        const double grb = pg[(size_t)nf - 1] - glb;
        const double hrb = ph[(size_t)nf - 1] - hlb;

        auto eval_dir = [&](bool miss_left) {
            const double HL = hlb + (miss_left ? hm : 0.0);
            const double HR = hrb + (miss_left ? 0.0 : hm);
            const int    nL = k + (miss_left ? Cm : 0);
            const int    nR = (nf + Cm) - nL;
            const double GL = glb + (miss_left ? gm : 0.0);
            const double GR = grb + (miss_left ? 0.0 : gm);
            return splitx::split_gain_from_parent(parent, GL, HL, GR, HR, nL, nR, hyp);
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

} // namespace detail

#undef x_from_code_variable

} // namespace foretree::splitaux
