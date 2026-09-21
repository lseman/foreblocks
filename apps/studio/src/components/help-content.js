export const HELP_CONTENT = {
    emd: {
        title: "Empirical Mode Decomposition (EMD)",
        content: [
            "<p><strong>Motivation.</strong> Classical spectral methods (Fourier, fixed-basis wavelets) assume linearity and stationarity. Real-world signals — such as hydroelectric inflows, physiological recordings, and seismic traces — are nonlinear and nonstationary: their dominant frequencies drift over time. EMD addresses this by decomposing a signal into a finite, data-driven set of oscillatory components called <em>intrinsic mode functions</em> (IMFs), without requiring any a priori basis.</p>",

            "<h4>Definition: Intrinsic Mode Function</h4>",
            "<p>A function \\(c(t)\\) is an IMF if it satisfies two conditions:</p>",
            "<ol><li>The number of extrema and the number of zero crossings differ by at most one globally.</li><li>The local mean envelope is zero at every point: \\(m(t) = \\tfrac{1}{2}\\bigl[e_{\\mathrm{up}}(t)+e_{\\mathrm{lo}}(t)\\bigr] \\equiv 0\\), where \\(e_{\\mathrm{up}}\\) and \\(e_{\\mathrm{lo}}\\) are the upper and lower envelopes obtained by cubic-spline interpolation of local maxima and minima, respectively.</li></ol>",

            "<h4>The Sifting Algorithm</h4>",
            "<p>Given a signal \\(x(t)\\), the \\(k\\)-th IMF is extracted as follows:</p>",
            "<p>Initialize \\(h_0(t) \\leftarrow x(t)\\). At each iteration \\(i\\):</p>",
            "<ol><li>Identify all local maxima and minima of \\(h_i(t)\\).</li><li>Fit cubic splines \\(e_{\\mathrm{up}}^{(i)}(t)\\) and \\(e_{\\mathrm{lo}}^{(i)}(t)\\) through maxima and minima, respectively.</li><li>Compute the mean envelope: \\(m_i(t) = \\tfrac{1}{2}\\bigl[e_{\\mathrm{up}}^{(i)}(t)+e_{\\mathrm{lo}}^{(i)}(t)\\bigr]\\).</li><li>Update the proto-IMF: \\(h_{i+1}(t) = h_i(t) - m_i(t)\\).</li><li>Stop when the stopping criterion is satisfied.</li></ol>",
            "<p>The standard stopping criterion is the <em>standard deviation</em> (SD) between successive sifting iterates:</p>",
            "<p>$$\\mathrm{SD}_i = \\frac{\\|h_{i+1}(t)-h_i(t)\\|^2}{\\|h_i(t)\\|^2} < \\delta,$$</p>",
            "<p>with \\(\\delta \\in [0.2, 0.3]\\) typically. The resulting \\(h_{i+1}(t)\\) is set as \\(c_k(t)\\).</p>",

            "<h4>Full Decomposition</h4>",
            "<p>After extracting \\(c_k(t)\\), update the residual \\(r_k(t) = r_{k-1}(t) - c_k(t)\\), with \\(r_0(t) = x(t)\\), and apply the sifting algorithm to \\(r_k\\). Repeat until the residual is monotone or has at most one extremum. The complete decomposition is:</p>",
            "<p>$$x(t) = \\sum_{k=1}^{K} c_k(t) + r_K(t),$$</p>",
            "<p>where \\(r_K(t)\\) is the final trend residual. Because the IMFs are extracted sequentially from highest to lowest instantaneous frequency, \\(c_1\\) captures the fastest oscillations and \\(c_K\\) the slowest.</p>",

            "<h4>Instantaneous Frequency via the Hilbert Transform</h4>",
            "<p>For each IMF, the analytic signal is \\(z_k(t) = c_k(t) + i\\,\\mathcal{H}\\{c_k\\}(t)\\), where \\(\\mathcal{H}\\) is the Hilbert transform. The <em>instantaneous amplitude</em> is \\(A_k(t) = |z_k(t)|\\) and the <em>instantaneous frequency</em> is \\(f_k(t) = \\tfrac{1}{2\\pi}\\frac{d}{dt}\\arg z_k(t)\\). This yields the Hilbert–Huang spectrum \\(H(f, t)\\), a time–frequency energy distribution without spectral leakage.</p>",

            "<h4>Properties and Limitations</h4>",
            "<ul><li><strong>Completeness.</strong> The reconstruction is exact: \\(x(t) = \\sum_k c_k(t) + r_K(t)\\).</li><li><strong>Adaptivity.</strong> No basis is prescribed; the decomposition is entirely driven by the signal's local extrema.</li><li><strong>Mode mixing.</strong> A single IMF may contain oscillations of disparate scales if the signal has intermittent events or noise. EEMD (see below) mitigates this.</li><li><strong>End effects.</strong> Spline fitting near boundaries is unreliable; mirror or wave-form extension is commonly used.</li><li><strong>No orthogonality guarantee.</strong> IMFs are not in general orthogonal, though empirical orthogonality is often observed.</li></ul>",
        ],
    },

    eemd: {
        title: "Ensemble EMD (EEMD)",
        content: [
            "<p><strong>Motivation.</strong> EMD suffers from <em>mode mixing</em>: when a signal contains impulsive events or broad-band noise, distinct physical oscillations may appear in the same IMF, or a single oscillation may be split across adjacent IMFs. EEMD exploits the noise-assisted data analysis (NADA) principle to statistically separate scales.</p>",

            "<h4>Procedure</h4>",
            "<p>Let \\(J\\) be the ensemble size and \\(\\sigma\\) the noise standard deviation (typically chosen as a fraction of the signal's standard deviation, e.g. \\(\\sigma = 0.1\\,\\mathrm{std}(x)\\)). For each trial \\(j = 1, \\ldots, J\\):</p>",
            "<ol><li>Draw independent white Gaussian noise: \\(\\varepsilon_j(t) \\sim \\mathcal{N}(0, \\sigma^2)\\).</li><li>Form the noisy realization: \\(x_j(t) = x(t) + \\varepsilon_j(t)\\).</li><li>Apply standard EMD to \\(x_j(t)\\), obtaining modes \\(c_k^{(j)}(t)\\), \\(k = 1,\\ldots,K\\).</li></ol>",
            "<p>The \\(k\\)-th ensemble IMF is the trial average:</p>",
            "<p>$$\\bar{c}_k(t) = \\frac{1}{J}\\sum_{j=1}^{J} c_k^{(j)}(t).$$</p>",

            "<h4>Why the Averaging Works</h4>",
            "<p>Since \\(\\varepsilon_j\\) are i.i.d. with zero mean, the expected added noise in the ensemble mean vanishes: \\(\\mathbb{E}[\\bar{c}_k(t)] = c_k(t)\\). The residual noise in the \\(k\\)-th mode decreases as \\(\\sigma/\\sqrt{J}\\), so increasing \\(J\\) improves mode purity. The reconstruction is:</p>",
            "<p>$$x(t) \\approx \\sum_{k=1}^{K} \\bar{c}_k(t) + \\bar{r}(t).$$</p>",
            "<p>Note that strict reconstruction equality no longer holds because \\(\\bar{r}(t)\\) absorbs residual ensemble noise. The reconstruction error scales as \\(O(\\sigma/\\sqrt{J})\\).</p>",

            "<h4>Parameter Guidance</h4>",
            "<ul><li><strong>\\(J\\):</strong> At least 100 trials recommended; 200–500 for noisy data. Runtime scales linearly with \\(J\\).</li><li><strong>\\(\\sigma\\):</strong> Too small: noise has negligible effect on mode mixing. Too large: dominant physical modes are contaminated. The standard guideline is \\(\\sigma \\in [0.1, 0.4]\\,\\mathrm{std}(x)\\).</li><li><strong>Complete EEMD (CEEMDAN)</strong> is a variant that enforces exact reconstruction by adaptively adding noise to each residual stage, rather than the original signal.</li></ul>",
        ],
    },

    ewt: {
        title: "Empirical Wavelet Transform (EWT)",
        content: [
            "<p><strong>Motivation.</strong> EWT (Gilles, 2013) is a rigorous, Fourier-based alternative to EMD. Rather than adaptively sifting in the time domain, it constructs a multiresolution filter bank directly from the signal's Fourier spectrum, using empirically detected spectral boundaries. This confers stronger theoretical guarantees while retaining data adaptivity.</p>",

            "<h4>Step 1: Spectral Segmentation</h4>",
            "<p>Compute the normalized Fourier spectrum \\(|X(\\omega)|\\). Detect the \\(K\\) most prominent local maxima in \\(|X(\\omega)|\\) on \\([0, \\pi]\\) and place boundary points \\(0 = \\omega_0 < \\omega_1 < \\cdots < \\omega_K = \\pi\\) at the midpoints between adjacent maxima. Each interval \\(\\Lambda_k = [\\omega_{k-1}, \\omega_k]\\) defines one empirical frequency band.</p>",

            "<h4>Step 2: Meyer-type Wavelet Filter Bank</h4>",
            "<p>For each boundary \\(\\omega_k\\), define a transition width \\(\\tau_k = \\gamma\\,\\omega_k\\) with \\(\\gamma < \\min_k(\\omega_k - \\omega_{k-1})/(\\omega_k + \\omega_{k-1})\\). The empirical scaling function and wavelets are:</p>",
            "<p>$$\\hat{\\phi}_1(\\omega) = \\begin{cases} 1, & |\\omega| \\le (1-\\gamma)\\omega_1 \\\\ \\cos\\!\\left[\\frac{\\pi}{2}\\beta\\!\\left(\\frac{|\\omega|-(1-\\gamma)\\omega_1}{2\\gamma\\omega_1}\\right)\\right], & (1-\\gamma)\\omega_1 < |\\omega| \\le (1+\\gamma)\\omega_1 \\\\ 0, & \\text{otherwise,}\\end{cases}$$</p>",
            "<p>where \\(\\beta(x) = x^4(35 - 84x + 70x^2 - 20x^3)\\) is the standard Meyer auxiliary polynomial satisfying \\(\\beta(x)+\\beta(1-x)=1\\). For \\(k \\ge 2\\):</p>",
            "<p>$$\\hat{\\psi}_k(\\omega) = \\begin{cases} 1, & (1+\\gamma)\\omega_{k-1} \\le |\\omega| \\le (1-\\gamma)\\omega_k \\\\ \\cos\\!\\left[\\frac{\\pi}{2}\\beta\\!\\left(\\frac{|\\omega|-(1-\\gamma)\\omega_k}{2\\gamma\\omega_k}\\right)\\right], & (1-\\gamma)\\omega_k < |\\omega| \\le (1+\\gamma)\\omega_k \\\\ \\sin\\!\\left[\\frac{\\pi}{2}\\beta\\!\\left(\\frac{|\\omega|-(1-\\gamma)\\omega_{k-1}}{2\\gamma\\omega_{k-1}}\\right)\\right], & (1-\\gamma)\\omega_{k-1} \\le |\\omega| \\le (1+\\gamma)\\omega_{k-1} \\\\ 0, & \\text{otherwise.}\\end{cases}$$</p>",

            "<h4>Step 3: Component Extraction and Reconstruction</h4>",
            "<p>The \\(k\\)-th empirical mode is recovered by inverse Fourier filtering:</p>",
            "<p>$$w_k(t) = \\mathcal{F}^{-1}\\{\\hat{\\psi}_k(\\omega) \\cdot X(\\omega)\\}(t),$$</p>",
            "<p>and the exact reconstruction is guaranteed by the tight-frame property of the filter bank:</p>",
            "<p>$$x(t) = \\mathcal{F}^{-1}\\{\\hat{\\phi}_1(\\omega)X(\\omega)\\}(t) + \\sum_{k=2}^{K} w_k(t).$$</p>",

            "<h4>Properties</h4>",
            "<ul><li><strong>Tight frame.</strong> The filter bank satisfies \\(|\\hat{\\phi}_1|^2 + \\sum_{k=2}^K|\\hat{\\psi}_k|^2 = 1\\) pointwise, ensuring perfect reconstruction and no energy inflation.</li><li><strong>Compact support in frequency.</strong> Each mode is band-limited by construction, unlike EMD modes which may have spectral overlap.</li><li><strong>No mode mixing.</strong> Frequency bands are disjoint, so each \\(w_k\\) contains exactly the spectral energy of \\(\\Lambda_k\\).</li><li><strong>Limitation.</strong> The spectral peak detection is sensitive to noise; pre-smoothing the magnitude spectrum or using a regularized peak finder is often necessary.</li></ul>",
        ],
    },

    isolation_forest: {
        title: "Isolation Forest",
        content: [
            "<p><strong>Motivation.</strong> Most anomaly detectors work by profiling <em>normal</em> behavior (e.g., density or distance estimation) and flagging deviations. Isolation Forest (Liu, Ting & Zhou, 2008) takes the opposite approach: it explicitly tries to <em>isolate</em> individual points through random recursive partitioning, exploiting the fact that anomalies are few and lie in sparse regions, making them easier to isolate.</p>",

            "<h4>Isolation Tree (iTree)</h4>",
            "<p>Given a subsample \\(\\mathbf{X}'\\) of size \\(\\psi\\) from the dataset, an isolation tree is built recursively:</p>",
            "<ol><li>If \\(|\\mathbf{X}'| = 1\\) or all points are identical, return a leaf with size \\(|\\mathbf{X}'|\\).</li><li>Select a feature \\(q\\) uniformly at random from all features.</li><li>Select a split value \\(p\\) uniformly at random in \\([\\min_q, \\max_q]\\) over \\(\\mathbf{X}'\\).</li><li>Partition \\(\\mathbf{X}'\\) into \\(\\mathbf{X}_L = \\{x : x_q < p\\}\\) and \\(\\mathbf{X}_R = \\{x : x_q \\ge p\\}\\).</li><li>Recurse on \\(\\mathbf{X}_L\\) and \\(\\mathbf{X}_R\\) until maximum tree depth \\(\\ell_{\\max} = \\lceil \\log_2 \\psi \\rceil\\) is reached.</li></ol>",
            "<p>The <em>path length</em> \\(h(x)\\) of a point \\(x\\) is the number of edges traversed from the root to the leaf containing \\(x\\). Anomalies, being sparse, tend to land in short-path leaves; normal points require deeper partitioning.</p>",

            "<h4>Anomaly Score</h4>",
            "<p>The expected path length for a BST on \\(n\\) points is:</p>",
            "<p>$$c(n) = 2H(n-1) - \\frac{2(n-1)}{n}, \\qquad H(k) = \\ln k + \\gamma_E \\approx \\ln k + 0.5772,$$</p>",
            "<p>where \\(\\gamma_E\\) is the Euler–Mascheroni constant. This normalises path lengths across different subsample sizes. The anomaly score for point \\(x\\), averaged over a forest of \\(T\\) trees, is:</p>",
            "<p>$$s(x, \\psi) = 2^{-\\,\\overline{h}(x)\\,/\\,c(\\psi)}, \\qquad \\overline{h}(x) = \\frac{1}{T}\\sum_{t=1}^{T} h_t(x).$$</p>",
            "<p>Scores near <strong>1</strong> indicate anomalies (short paths); scores near <strong>0.5</strong> indicate normal points; scores near <strong>0</strong> indicate dense, deeply-embedded observations.</p>",

            "<h4>Complexity and Practical Guidance</h4>",
            "<ul><li><strong>Training:</strong> \\(O(T \\psi \\log \\psi)\\) time; subsampling size \\(\\psi = 256\\) is typically sufficient — larger subsamples yield diminishing returns.</li><li><strong>Scoring:</strong> \\(O(T \\log \\psi)\\) per point.</li><li><strong>\\(T\\):</strong> 100 trees usually gives converged scores; path lengths stabilize well before that.</li><li><strong>Extended Isolation Forest (EIF)</strong> generalises the axis-aligned splits to random hyperplane cuts, eliminating the scoring artifacts (\"ghost clusters\" and \"branches\") caused by 1-D splits at feature boundaries.</li><li><strong>1-D time series context.</strong> For univariate preview here, the feature space is the value range; the isolation score reflects how extreme each observation is relative to the local value distribution within random subsamples.</li></ul>",
        ],
    },

    filter: {
        none: {
            title: "No Filter",
            content: [
                "<p>The raw observed series \\(x_t\\) is displayed without any preprocessing. This mode is useful for inspecting the native signal, identifying obvious outliers, and assessing the degree of noise before selecting a filter.</p>",
            ],
        },
        savgol: {
            title: "Savitzky–Golay Filter",
            content: [
                "<p><strong>Principle.</strong> The Savitzky–Golay filter (Savitzky & Golay, 1964) smooths a signal by fitting a polynomial of degree \\(p\\) to a sliding window of width \\(W = 2m+1\\) centered at each sample, then evaluating the polynomial at the center. This is equivalent to a weighted moving average whose coefficients preserve polynomial moments up to degree \\(p\\), thereby retaining peaks, valleys, and higher-order features that simple averaging suppresses.</p>",
                "<p>$$y_t = \\sum_{k=-m}^{m} c_k\\, x_{t+k},$$</p>",
                "<p>where the weight vector \\(\\mathbf{c} = \\bigl(\\mathbf{V}^\\top \\mathbf{V}\\bigr)^{-1}\\mathbf{V}^\\top \\mathbf{e}_0\\) is the zeroth row of the pseudo-inverse of the Vandermonde matrix \\(\\mathbf{V}_{ij} = j^i\\), \\(i = 0,\\ldots,p\\), \\(j = -m,\\ldots,m\\).</p>",
                "<p><strong>Frequency response.</strong> In the frequency domain the filter is approximately a low-pass filter with a passband that widens with smaller \\(W\\) and narrows with larger \\(p\\). Unlike the ideal low-pass, the transition is smooth and the stopband attenuation grows with the distance between \\(p\\) and \\(W\\).</p>",
                "<p><strong>Parameters.</strong> Window width \\(W\\) controls smoothing strength (larger \\(W\\) = more smoothing). Polynomial order \\(p\\) controls how much local curvature is preserved (higher \\(p\\) = sharper features retained). Constraint: \\(p < W\\).</p>",
            ],
        },
        moving_average: {
            title: "Simple Moving Average",
            content: [
                "<p><strong>Definition.</strong> The symmetric (centred) moving average of width \\(W = 2m+1\\) is:</p>",
                "<p>$$y_t = \\frac{1}{W} \\sum_{k=-m}^{m} x_{t+k}.$$</p>",
                "<p>This is a convolution of \\(x_t\\) with a rectangular kernel \\(h_k = 1/W\\) for \\(|k| \\le m\\).</p>",
                "<p><strong>Frequency response.</strong> The discrete-time Fourier transform of the rectangular kernel is:</p>",
                "<p>$$H(e^{i\\omega}) = \\frac{1}{W}\\frac{\\sin(W\\omega/2)}{\\sin(\\omega/2)},$$</p>",
                "<p>a sinc-like function with the first zero at \\(\\omega = 2\\pi/W\\). This implies significant sidelobe leakage: the filter does not have a sharp cutoff and attenuates slowly beyond the main lobe.</p>",
                "<p><strong>Properties.</strong> Computationally trivial; preserves the local mean; polynomial trend of degree 0 (i.e., constants) are reproduced exactly. Polynomials of degree \\(\\ge 1\\) are not preserved, so ramps and curved trends are attenuated — a key disadvantage compared to Savitzky–Golay.</p>",
            ],
        },
        median: {
            title: "Median Filter",
            content: [
                "<p><strong>Definition.</strong> The median filter replaces each sample by the median of the \\(W = 2m+1\\) samples centered on it:</p>",
                "<p>$$y_t = \\operatorname{median}\\{x_{t-m},\\, x_{t-m+1},\\, \\ldots,\\, x_{t+m}\\}.$$</p>",
                "<p><strong>Statistical motivation.</strong> The median is the minimiser of the \\(L^1\\) cost \\(\\sum_k |x_{t+k} - c|\\), making it inherently robust to large deviations. The mean, by contrast, minimises the \\(L^2\\) cost and is sensitive to outliers. A single impulsive spike in a window of width \\(W\\) leaves the median unchanged as long as fewer than \\(m\\) points are corrupted (breakdown point \\(\\approx 50\\%\\)).</p>",
                "<p><strong>Edge preservation.</strong> Unlike linear filters, the median filter does not blur step discontinuities (edges): a step function is reproduced exactly for \\(m < \\)step width. This property is particularly useful in hydrology and fault detection, where abrupt regime changes coexist with high-frequency noise.</p>",
                "<p><strong>Limitation.</strong> Non-linear: superposition does not hold. Runtime per sample is \\(O(W \\log W)\\) with a sorting approach, or \\(O(W)\\) with histogram-based methods.</p>",
            ],
        },
        wiener: {
            title: "Wiener Filter (adaptive estimation)",
            content: [
                "<p><strong>Optimal linear filter.</strong> The Wiener filter minimises the mean-square error \\(\\mathbb{E}[|x_t - y_t|^2]\\) between the true signal and the filtered output under a signal+noise model \\(x_t = s_t + n_t\\), where \\(s_t\\) is the signal of interest and \\(n_t\\) is zero-mean additive noise.</p>",
                "<p>In the frequency domain, the optimal (non-causal) Wiener filter is:</p>",
                "<p>$$H^*(\\omega) = \\frac{S_{ss}(\\omega)}{S_{ss}(\\omega) + S_{nn}(\\omega)},$$</p>",
                "<p>where \\(S_{ss}\\) and \\(S_{nn}\\) are the power spectral densities (PSDs) of signal and noise, respectively. When the SNR is high, \\(H^* \\approx 1\\) (pass-through); when low, \\(H^* \\approx 0\\) (suppress).</p>",
                "<p><strong>Local (adaptive) approximation used here.</strong> Because global PSDs are unavailable, a sliding-window estimate is used:</p>",
                "<p>$$y_t = \\mu_t + \\frac{\\max(0,\\,\\hat{\\sigma}_x^2 - \\hat{\\sigma}_n^2)}{\\hat{\\sigma}_x^2}\\,(x_t - \\mu_t),$$</p>",
                "<p>where \\(\\mu_t\\) and \\(\\hat{\\sigma}_x^2\\) are the local mean and variance estimated over the window, and \\(\\hat{\\sigma}_n^2\\) is a global noise variance estimate (e.g., from the median absolute deviation of the high-frequency detail). When local variance greatly exceeds noise variance, smoothing is minimal; when local variance approaches noise variance, the output is pulled toward the local mean.</p>",
                "<p><strong>Key property.</strong> Spatially adaptive: smooth regions are strongly smoothed, while high-variance (feature-rich) regions are preserved. This makes it a practical compromise between signal preservation and noise suppression.</p>",
            ],
        },
        gaussian: {
            title: "Gaussian Filter",
            content: [
                "<p><strong>Definition.</strong> The Gaussian filter convolves the signal with a discretized Gaussian kernel:</p>",
                "<p>$$y_t = \\sum_{k=-m}^{m} w_k\\, x_{t+k}, \\qquad w_k = \\frac{\\exp\\!\\left(-k^2/2\\sigma^2\\right)}{\\sum_{j=-m}^{m}\\exp\\!\\left(-j^2/2\\sigma^2\\right)},$$</p>",
                "<p>where \\(\\sigma\\) is the standard deviation of the kernel in samples and \\(m = \\lceil 3\\sigma \\rceil\\) is typically used to truncate the kernel at negligible weight.</p>",
                "<p><strong>Frequency response.</strong> The Fourier transform of the Gaussian kernel is itself a Gaussian:</p>",
                "<p>$$H(\\omega) \\approx \\exp\\!\\left(-\\sigma^2 \\omega^2 / 2\\right),$$</p>",
                "<p>so there are no sidelobes, no ringing, and no Gibbs phenomenon — unlike the rectangular (moving average) kernel. The \\(-3\\,\\mathrm{dB}\\) cutoff frequency is at \\(\\omega_c = \\sigma^{-1}\\sqrt{2\\ln 2}\\).</p>",
                "<p><strong>Optimality.</strong> The Gaussian is the unique kernel that achieves the minimum time–bandwidth product (uncertainty principle equality), making it the smoothest possible low-pass filter for a given scale. It also satisfies the <em>scale-space axioms</em> (Lindeberg, 1994): linearity, shift-invariance, semi-group property, and non-creation of new extrema at coarser scales.</p>",
            ],
        },
        ema: {
            title: "Exponential Moving Average (EMA)",
            content: [
                "<p><strong>Definition.</strong> The EMA is a first-order infinite impulse response (IIR) filter defined by the recursion:</p>",
                "<p>$$y_t = \\alpha\\, x_t + (1-\\alpha)\\, y_{t-1}, \\qquad \\alpha \\in (0, 1].$$</p>",
                "<p>Unrolling the recursion reveals that \\(y_t = \\alpha\\sum_{k=0}^{\\infty}(1-\\alpha)^k x_{t-k}\\), so each output is a weighted average of all past observations with geometrically decaying weights. The <em>effective memory</em> (number of samples that contribute substantially) is \\(\\tau \\approx 1/\\alpha\\).</p>",
                "<p><strong>Frequency response.</strong> The transfer function is:</p>",
                "<p>$$H(z) = \\frac{\\alpha}{1-(1-\\alpha)z^{-1}},$$</p>",
                "<p>a single-pole low-pass filter with pole at \\(z = 1-\\alpha\\). The \\(-3\\,\\mathrm{dB}\\) frequency is \\(f_c = \\frac{\\alpha}{2\\pi}\\) (normalized, \\(0 < f_c < 0.5\\)).</p>",
                "<p><strong>Causality and lag.</strong> Unlike the symmetric filters above, the EMA is strictly causal: \\(y_t\\) depends only on \\(x_t\\) and earlier samples. This introduces a phase lag of \\(1/\\alpha - 1\\) samples. For forecasting applications requiring a causal filter, this is a feature; for visualization of past data, the lag is a drawback.</p>",
                "<p><strong>Relation to \\(\\alpha\\) and half-life.</strong> The \\(\\alpha\\) parameter can be specified via the half-life \\(\\tau_{1/2}\\): the weights halve every \\(\\tau_{1/2}\\) samples, giving \\(\\alpha = 1 - 2^{-1/\\tau_{1/2}}\\).</p>",
            ],
        },
        lowess: {
            title: "LOWESS (Locally Weighted Scatterplot Smoothing)",
            content: [
                "<p><strong>Definition.</strong> LOWESS (Cleveland, 1979) estimates \\(y_t\\) by fitting a low-degree polynomial (typically degree 1 or 2) by weighted least squares within a neighborhood of size \\(\\lfloor f n \\rfloor\\) samples, where \\(f \\in (0,1]\\) is the bandwidth fraction and \\(n\\) is the series length.</p>",
                "<p>At each point \\(t\\), define the neighborhood \\(\\mathcal{N}_t\\) as the \\(q = \\lfloor fn \\rfloor\\) closest points. The tricube weight function assigns weight to each neighbor \\(s\\):</p>",
                "<p>$$w(s; t) = \\left(1 - \\left|\\frac{s-t}{d_t}\\right|^3\\right)^3_+,$$</p>",
                "<p>where \\(d_t = \\max_{s \\in \\mathcal{N}_t} |s - t|\\) is the maximum distance in \\(\\mathcal{N}_t\\) and \\((\\cdot)_+ = \\max(0, \\cdot)\\). The local fit solves:</p>",
                "<p>$$\\hat{\\boldsymbol{\\beta}}_t = \\arg\\min_{\\boldsymbol{\\beta}} \\sum_{s \\in \\mathcal{N}_t} w(s; t)\\,(x_s - \\mathbf{p}_s^\\top \\boldsymbol{\\beta})^2,$$</p>",
                "<p>and \\(y_t = \\mathbf{p}_t^\\top \\hat{\\boldsymbol{\\beta}}_t\\), where \\(\\mathbf{p}_s = [1, s, s^2, \\ldots]^\\top\\) for polynomial degree \\(d\\).</p>",
                "<p><strong>Robustness iterations.</strong> LOWESS optionally adds robustness iterations (RLOWESS): after an initial fit, residuals \\(\\hat{e}_s = x_s - y_s\\) are used to compute bisquare robustness weights \\(\\delta_s = (1-(\\hat{e}_s/6\\,\\mathrm{MAD})^2)^2_+\\), and the weighted least squares is re-run with weights \\(w(s;t)\\cdot\\delta_s\\). This downweights outliers automatically.</p>",
                "<p><strong>Comparison to global polynomial regression.</strong> LOWESS is non-parametric: it makes no global assumption about functional form. The bias–variance tradeoff is controlled by \\(f\\): small \\(f\\) gives low bias but high variance (follows noise); large \\(f\\) gives high bias but low variance (oversmooths). Typical values: \\(f \\in [0.25, 0.75]\\).</p>",
            ],
        },
    },

    stl_decomposition: {
        title: "STL Decomposition",
        content: [
            "<p><strong>Overview.</strong> STL (Seasonal-Trend decomposition using Loess; Cleveland et al., 1990) decomposes a time series additively into three components:</p>",
            "<p>$$x_t = T_t + S_t + R_t,$$</p>",
            "<p>where \\(T_t\\) is the low-frequency trend, \\(S_t\\) is the periodic seasonal component, and \\(R_t = x_t - T_t - S_t\\) is the remainder (residual). The decomposition is exact: given \\(T_t\\) and \\(S_t\\), the residual is determined exactly.</p>",

            "<h4>Algorithm</h4>",
            "<p>STL alternates between an <strong>inner loop</strong> (which updates \\(S_t\\) and \\(T_t\\)) and an optional <strong>outer loop</strong> (which computes robustness weights). Let \\(p\\) be the seasonal period (e.g., \\(p = 12\\) for monthly data, \\(p = 7\\) for daily with weekly seasonality). In the inner loop:</p>",
            "<ol>",
            "<li><strong>Detrending.</strong> Form \\(v_t = x_t - T_t^{(k)}\\) (using current trend estimate; initialise \\(T^{(0)} = 0\\)).</li>",
            "<li><strong>Cycle-subseries smoothing.</strong> For each phase \\(j = 0, 1, \\ldots, p-1\\), collect all samples \\(\\{v_{j}, v_{j+p}, v_{j+2p}, \\ldots\\}\\) and smooth them with a LOESS smoother of bandwidth \\(n_s\\) (seasonal smoothing parameter). This yields a smoothed seasonal subseries \\(C_t\\).</li>",
            "<li><strong>Low-pass filtering of seasonal.</strong> Apply a moving average of length \\(p\\), then another of length \\(p\\), then one of length 3, then a LOESS fit with bandwidth \\(n_l\\) to \\(C_t\\), giving \\(L_t\\). This removes the trend contaminating the seasonal estimate.</li>",
            "<li><strong>Seasonal component.</strong> \\(S_t^{(k+1)} = C_t - L_t\\).</li>",
            "<li><strong>De-seasonalizing.</strong> \\(d_t = x_t - S_t^{(k+1)}\\).</li>",
            "<li><strong>Trend smoothing.</strong> \\(T_t^{(k+1)} = \\operatorname{LOESS}(d_t, n_t)\\), where \\(n_t\\) is the trend bandwidth parameter.</li>",
            "</ol>",
            "<p>The outer loop re-runs the inner loop with bisquare robustness weights based on the residuals \\(R_t\\), downweighting outliers and reducing their influence on the seasonal and trend estimates.</p>",

            "<h4>Key Parameters</h4>",
            "<ul><li><strong>\\(p\\):</strong> Seasonal period. Must match the dominant cycle in the data. Can be inferred from the dominant ACF lag or a periodogram peak.</li><li><strong>\\(n_s\\):</strong> Seasonal LOESS bandwidth. Larger values enforce smoother, less time-varying seasonality. Must be odd and \\(\\ge 7\\).</li><li><strong>\\(n_t\\):</strong> Trend LOESS bandwidth. Controls how rapidly the trend can change. Rule of thumb: \\(n_t \\ge 1.5\\,p\\,(1 - 1.5/n_s)^{-1}\\).</li><li><strong>Robust vs. non-robust:</strong> Enable robustness iterations when outliers or level shifts are suspected.</li></ul>",

            "<h4>Properties</h4>",
            "<ul><li>Handles any seasonal period, including non-integer and long periods (unlike classical decomposition, which requires \\(p\\) to divide \\(n\\)).</li><li>Seasonal component is allowed to evolve slowly over time (unlike X-11 or classical methods which assume fixed seasonality).</li><li>Additive form assumed; for multiplicative seasonality, apply STL to \\(\\log x_t\\) and back-transform.</li></ul>",
        ],
    },

    adf: {
        title: "Augmented Dickey–Fuller (ADF) Unit Root Test",
        content: [
            "<p>The ADF test checks for a unit root in the observed series. The null hypothesis is that the series is nonstationary with a unit root, while the alternative is that the series is stationary. A statistic more negative than the critical value supports rejection of the null.</p>",
            "<p>The test regression typically includes an intercept and trend. The reported output here includes the ADF test statistic, the 5% critical value, and the selected lag order used to account for serial correlation.</p>",
        ],
    },
    kpss: {
        title: "KPSS Level Stationarity Test",
        content: [
            "<p>The KPSS test has the opposite null hypothesis of ADF: it assumes the series is level stationary around a constant mean. The alternative is that the series contains a unit root and is nonstationary.</p>",
            "<p>A large KPSS statistic above the critical value indicates rejection of level stationarity. The statistic uses a long-run variance estimate controlled by the bandwidth parameter to account for serial correlation in residuals.</p>",
        ],
    },
    acf_pacf: {
        title: "ACF and PACF",
        content: [
            "<h4>Autocorrelation Function (ACF)</h4>",
            "<p>The autocorrelation at lag \\(k\\) measures the linear correlation between \\(x_t\\) and \\(x_{t-k}\\):</p>",
            "<p>$$\\rho(k) = \\frac{\\gamma(k)}{\\gamma(0)}, \\qquad \\gamma(k) = \\mathbb{E}[(x_t - \\mu)(x_{t-k} - \\mu)],$$</p>",
            "<p>where \\(\\gamma(k)\\) is the autocovariance and \\(\\gamma(0) = \\mathrm{Var}(x_t)\\). The sample estimator is:</p>",
            "<p>$$\\hat{\\rho}(k) = \\frac{\\sum_{t=k+1}^{n}(x_t - \\bar{x})(x_{t-k}-\\bar{x})}{\\sum_{t=1}^{n}(x_t-\\bar{x})^2}.$$</p>",

            "<h4>Partial Autocorrelation Function (PACF)</h4>",
            "<p>The PACF at lag \\(k\\) isolates the direct linear dependence of \\(x_t\\) on \\(x_{t-k}\\) after removing the indirect influence of all intermediate lags \\(1, 2, \\ldots, k-1\\). Formally, \\(\\phi_{kk}\\) is the last coefficient of the OLS regression:</p>",
            "<p>$$x_t = \\phi_{k1}x_{t-1} + \\phi_{k2}x_{t-2} + \\cdots + \\phi_{kk}x_{t-k} + \\varepsilon_t.$$</p>",
            "<p>It can be computed via the Yule–Walker equations or the Durbin–Levinson recursion:</p>",
            "<p>$$\\phi_{kk} = \\frac{\\rho(k) - \\sum_{j=1}^{k-1}\\phi_{k-1,j}\\,\\rho(k-j)}{1 - \\sum_{j=1}^{k-1}\\phi_{k-1,j}\\,\\rho(j)}.$$</p>",

            "<h4>Confidence Bands</h4>",
            "<p>Under the null hypothesis that \\(x_t\\) is white noise, \\(\\sqrt{n}\\,\\hat{\\rho}(k) \\xrightarrow{d} \\mathcal{N}(0,1)\\) for each fixed \\(k\\) (Bartlett, 1946). The approximate \\(95\\%\\) confidence band is therefore \\(\\pm 1.96/\\sqrt{n}\\). Bars exceeding this threshold suggest statistically significant correlation at that lag. For non-white-noise series, more precise bands based on the asymptotic variance of \\(\\hat{\\rho}(k)\\) under the fitted model should be used.</p>",

            "<h4>Diagnostic Use</h4>",
            "<ul><li><strong>AR(p) process:</strong> ACF decays exponentially or sinusoidally; PACF cuts off sharply after lag \\(p\\).</li><li><strong>MA(q) process:</strong> ACF cuts off sharply after lag \\(q\\); PACF decays geometrically.</li><li><strong>ARMA(p,q):</strong> Both ACF and PACF tail off — require information criteria (AIC/BIC) for order selection.</li><li><strong>Seasonality:</strong> Spikes in the ACF at multiples of the seasonal period \\(p\\) signal periodic structure.</li><li><strong>Nonstationarity:</strong> Slow, near-unit-root decay of the ACF (e.g., ACF at lag 1 near 1) suggests differencing may be needed before ARMA fitting.</li></ul>",
        ],
    },
};
