import { useState } from "react";

import { Panel, NumberField, SelectField, TextField, ToggleField } from "./base.jsx";
import { ArchDiagram } from "./arch-diagram.jsx";

function renderDatasetStatus(datasetState, config) {
    if (datasetState.status === "ready") {
        const targetGapCount = datasetState.summary.missingTargetCount + datasetState.summary.invalidTargetCount;
        const detectedColumns = [];

        if (datasetState.summary.targetColumn) {
            detectedColumns.push(`target: ${datasetState.summary.targetColumn}`);
        }

        if (datasetState.summary.timeColumn) {
            detectedColumns.push(`timestamp: ${datasetState.summary.timeColumn}`);
        }

        return (
            <>
                <div>
                    Loaded {datasetState.points} retained observations from {datasetState.sourceLabel}.
                </div>
                <div>
                    Parsed {datasetState.summary.rowCount} rows across {datasetState.summary.columnCount} columns, with {datasetState.summary.missingCellCount} empty cells and {datasetState.summary.droppedRowCount} dropped rows.
                </div>
                <div>
                    Target gaps: {targetGapCount}. Timestamp gaps: {datasetState.summary.missingTimestampCount}.
                </div>
                {detectedColumns.length > 0 ? <div>Detected columns: {detectedColumns.join(" · ")}.</div> : null}
            </>
        );
    }

    if (datasetState.status === "loading") {
        return `Loading ${datasetState.sourceLabel || config.data.filename} and recalculating diagnostics.`;
    }

    return `Load failed: ${datasetState.errorMessage}`;
}

export function DatasetStep({ config, setConfig, datasetState, onFileLoad }) {
    const [isDragging, setIsDragging] = useState(false);

    const updateDataset = (key, value) => {
        setConfig((current) => ({
            ...current,
            data: {
                ...current.data,
                [key]: value,
            },
        }));
    };

    return (
        <Panel title="Dataset Framing" kicker="Step 01">
            <div className="field-grid">
                <label
                    className={`field dropzone ${isDragging ? "dropzone-active" : ""}`}
                    onDragEnter={(event) => {
                        event.preventDefault();
                        setIsDragging(true);
                    }}
                    onDragOver={(event) => {
                        event.preventDefault();
                        setIsDragging(true);
                    }}
                    onDragLeave={(event) => {
                        event.preventDefault();
                        if (!event.currentTarget.contains(event.relatedTarget)) {
                            setIsDragging(false);
                        }
                    }}
                    onDrop={(event) => {
                        event.preventDefault();
                        setIsDragging(false);
                        const file = event.dataTransfer.files?.[0];
                        if (file) {
                            onFileLoad(file);
                        }
                    }}
                >
                    <span className="field-label">CSV upload</span>
                    <div className="dropzone-copy">
                        <strong>Drop a CSV here</strong>
                        <span>or click to choose a file</span>
                    </div>
                    <input
                        className="dropzone-input"
                        type="file"
                        accept=".csv,text/csv"
                        onChange={(event) => {
                            const file = event.target.files?.[0];
                            if (file) {
                                onFileLoad(file);
                            }
                            event.target.value = "";
                        }}
                    />
                    <span className="field-hint">
                        Upload a CSV to drive the real series preview and lag diagnostics.
                    </span>
                </label>
                <TextField
                    label="Dataset file"
                    value={config.data.filename}
                    onChange={(value) => updateDataset("filename", value)}
                    hint="Relative URL for browser loading, or the filename of an uploaded CSV."
                />
                <TextField
                    label="Target column"
                    value={config.data.target}
                    onChange={(value) => updateDataset("target", value)}
                    hint="Primary forecasting signal."
                />
                <TextField
                    label="Timestamp column"
                    value={config.data.timestamp}
                    onChange={(value) => updateDataset("timestamp", value)}
                    hint="Datetime index for ordering and time features."
                />
                <SelectField
                    label="Cadence"
                    value={config.data.freq}
                    onChange={(value) => updateDataset("freq", value)}
                    options={[
                        { value: "auto", label: "Auto-detect" },
                        { value: "H", label: "Hourly" },
                        { value: "D", label: "Daily" },
                        { value: "W", label: "Weekly" },
                        { value: "M", label: "Monthly" },
                    ]}
                    hint="Used to infer seasonal context."
                />
                <NumberField
                    label="Context window"
                    value={config.prep.windowSize}
                    onChange={(value) =>
                        setConfig((current) => ({
                            ...current,
                            prep: {
                                ...current.prep,
                                windowSize: value,
                            },
                        }))
                    }
                    min={12}
                    max={512}
                    hint="Number of lookback observations per sample."
                />
                <NumberField
                    label="Forecast horizon"
                    value={config.prep.horizon}
                    onChange={(value) =>
                        setConfig((current) => ({
                            ...current,
                            prep: {
                                ...current.prep,
                                horizon: value,
                            },
                        }))
                    }
                    min={1}
                    max={192}
                    hint="Number of future steps predicted per sample."
                />
            </div>
            <div className="callout dataset-callout">
                <div className="callout-title">Dataset status</div>
                <div className="callout-copy dataset-status-copy">
                    {renderDatasetStatus(datasetState, config)}
                </div>
                {datasetState.status === "ready" && datasetState.headers.length > 0 ? (
                    <div className="header-chip-row">
                        {datasetState.headers.map((header) => (
                            <span key={header} className="header-chip">
                                {header}
                            </span>
                        ))}
                    </div>
                ) : null}
            </div>
        </Panel>
    );
}

export function PreprocessStep({ config, setConfig }) {
    const updatePreprocessing = (key, value) => {
        setConfig((current) => ({
            ...current,
            prep: {
                ...current.prep,
                [key]: value,
            },
        }));
    };

    return (
        <Panel title="Preprocessing Stack" kicker="Step 02">
            <div className="field-grid">
                <SelectField
                    label="Scaler"
                    value={config.prep.scalingMethod}
                    onChange={(value) => updatePreprocessing("scalingMethod", value)}
                    options={[
                        { value: "standard", label: "StandardScaler" },
                        { value: "robust", label: "RobustScaler" },
                        { value: "minmax", label: "MinMaxScaler" },
                        { value: "none", label: "No scaling" },
                    ]}
                    hint="Maps to TimeSeriesHandler scaling behavior."
                />
                <SelectField
                    label="Missing values"
                    value={config.prep.imputeMethod}
                    onChange={(value) => updatePreprocessing("imputeMethod", value)}
                    options={[
                        { value: "linear", label: "Linear interpolation" },
                        { value: "ffill", label: "Forward fill" },
                        { value: "bfill", label: "Backward fill" },
                        { value: "auto", label: "Auto" },
                    ]}
                    hint="Default fill path for sparse timestamps."
                />
                <SelectField
                    label="Outlier method"
                    value={config.prep.outlierMethod}
                    onChange={(value) => updatePreprocessing("outlierMethod", value)}
                    options={[
                        { value: "iqr", label: "IQR fences" },
                        { value: "zscore", label: "Z-score" },
                        { value: "mad", label: "MAD" },
                        { value: "isolation_forest", label: "Isolation Forest" },
                        { value: "lof", label: "Local Outlier Factor" },
                        { value: "none", label: "No detection" },
                    ]}
                    hint="Used for both Studio preview and emitted TimeSeriesHandler code."
                />
                <ToggleField
                    label="Apply outlier cleanup"
                    checked={config.prep.removeOutliers}
                    onChange={(value) => updatePreprocessing("removeOutliers", value)}
                    hint="Keeps the detection preview visible while letting you disable removal in the generated pipeline."
                />
                <ToggleField
                    label="Calendar features"
                    checked={config.prep.generateTimeFeatures}
                    onChange={(value) => updatePreprocessing("generateTimeFeatures", value)}
                    hint="Injects month, season, or other temporal covariates."
                />
                <ToggleField
                    label="Seasonal decomposition"
                    checked={config.prep.applyEwt}
                    onChange={(value) => updatePreprocessing("applyEwt", value)}
                    hint="Useful for long seasonal histories."
                />
                <ToggleField
                    label="Auto self-tuning"
                    checked={config.prep.selfTune}
                    onChange={(value) => updatePreprocessing("selfTune", value)}
                    hint="Lets TimeSeriesHandler adapt parts of the preprocessing stack."
                />
            </div>
        </Panel>
    );
}

const ATTENTION_MODE_OPTIONS = [
    {
        value: "standard",
        label: "Standard MHA",
        description: "Scaled dot-product multi-head attention. Quadratic complexity with full receptive field over the whole sequence.",
    },
    {
        value: "linear",
        label: "Linear",
        description: "Kernelized linear attention. Lower memory and compute on long windows, with some loss of exact global attention fidelity.",
    },
    {
        value: "sype",
        label: "SyPE",
        description: "Symmetric positional encoding attention. Injects positional structure directly into the score path instead of relying on learned embeddings.",
    },
    {
        value: "hybrid",
        label: "Hybrid",
        description: "Mostly local linear-style layers with a final full-attention refresh. Balances global context with cheaper intermediate layers.",
    },
    {
        value: "kimi",
        label: "Kimi",
        description: "Kimi sliding-window attention with periodic global sink tokens. Built for long context without paying full quadratic cost everywhere.",
    },
    {
        value: "hybrid_kimi",
        label: "Hybrid Kimi",
        description: "Mostly Kimi layers with a final full global layer. Preserves long-range recovery while reducing KV pressure in deeper stacks.",
    },
    {
        value: "kimi_3to1",
        label: "Kimi 3:1",
        description: "Repeats three Kimi layers, then one standard full-attention layer. Aggressive local compression with scheduled global refresh.",
    },
    {
        value: "gated_delta",
        label: "Gated Delta",
        description: "Delta-rule attention with a learnable gate. Behaves like a selective recurrent memory update at each position.",
    },
    {
        value: "hybrid_gdn",
        label: "Hybrid GDN",
        description: "Mostly gated-delta layers with a final full-attention layer. Useful when memory efficiency matters but periodic global mixing is still important.",
    },
    {
        value: "gdn_3to1",
        label: "GDN 3:1",
        description: "Repeats three gated-delta layers, then one standard full-attention layer. Maximizes long-horizon efficiency with regular global refresh.",
    },
];

const ATTN_DESCRIPTIONS = ATTENTION_MODE_OPTIONS.reduce((lookup, option) => {
    lookup[option.value] = option.description;
    return lookup;
}, {});

export function ArchitectureStep({ config, setConfig, familyMeta, analysis, patchingDiagnostics }) {
    const updateArchitecture = (key, value) => {
        setConfig((current) => ({
            ...current,
            model: {
                ...current.model,
                [key]: value,
            },
        }));
    };

    const updateHeadOption = (key, value) => {
        setConfig((current) => ({
            ...current,
            model: {
                ...current.model,
                ...(key === "useRevIN" ? { revIn: value } : null),
                heads: {
                    ...(current.model.heads ?? {}),
                    [key]: value,
                },
            },
        }));
    };

    const isTransformer = config.model.family === "transformer";
    const headConfig = config.model.heads ?? {};
    const encoderFfnType = config.model.encFfnType ?? config.model.ffnType ?? "swiglu";
    const decoderFfnType = config.model.decFfnType ?? config.model.ffnType ?? "swiglu";
    const encoderFfDim = config.model.encFfDim ?? config.model.ffDim ?? config.model.hiddenSize * 4;
    const decoderFfDim = config.model.decFfDim ?? config.model.ffDim ?? config.model.hiddenSize * 4;
    const encoderNumExperts = config.model.encNumExperts ?? config.model.numExperts ?? 4;
    const decoderNumExperts = config.model.decNumExperts ?? config.model.numExperts ?? 4;
    const encoderTopK = config.model.encTopK ?? config.model.topK ?? 2;
    const decoderTopK = config.model.decTopK ?? config.model.topK ?? 2;
    const ffnOptions = [
        { value: "standard", label: "Standard" },
        { value: "swiglu", label: "SwiGLU" },
        { value: "moe", label: "MoE + SwiGLU" },
    ];
    const useRevIN = headConfig.useRevIN ?? true;
    const useDecomposition = headConfig.useDecomposition ?? false;
    const useMultiScaleConv = headConfig.useMultiScaleConv ?? false;
    const hasActiveHeadConfigs = useRevIN || useDecomposition || useMultiScaleConv;
    const headRecommendations = [];

    if (isTransformer) {
        headRecommendations.push("RevINHead is the safest default when the transformer has to absorb scale drift.");
    }
    if (analysis?.needsDetrend || analysis?.trendLabel === "trend-stationary") {
        headRecommendations.push("DecompositionBlock is worth benchmarking because the diagnostics see deterministic drift.");
    }
    if (["moderate", "high"].includes(patchingDiagnostics?.multiscaleLabel ?? analysis?.multiscaleLabel)) {
        headRecommendations.push("MultiScaleConvHead matches the current multiscale signal score better than a single-scale residual path.");
    }

    return (
        <Panel title="Model Architecture" kicker="Step 03">
            <div className="architecture-banner">
                <div>
                    <div className="architecture-name">{familyMeta.label}</div>
                    <div className="field-hint">{familyMeta.detail}</div>
                </div>
                <SelectField
                    label="Family"
                    value={config.model.family}
                    onChange={(value) => updateArchitecture("family", value)}
                    options={[
                        { value: "direct", label: "Direct baseline" },
                        { value: "lstm", label: "LSTM encoder-decoder" },
                        { value: "gru", label: "GRU encoder-decoder" },
                        { value: "transformer", label: "Transformer encoder-decoder" },
                    ]}
                />
            </div>

            {/* ── Common fields ─────────────────────────────────────────── */}
            <div className="field-grid">
                <NumberField
                    label="Hidden dimension"
                    value={config.model.hiddenSize}
                    onChange={(value) => updateArchitecture("hiddenSize", value)}
                    min={16}
                    max={512}
                    step={8}
                    hint="Latent width shared by encoder and decoder stacks."
                />
                <NumberField
                    label="Dropout"
                    value={config.model.dropout}
                    onChange={(value) => updateArchitecture("dropout", value)}
                    min={0}
                    max={0.6}
                    step={0.05}
                    hint="Regularization applied inside the backbone."
                />
                {!isTransformer && (
                    <>
                        <NumberField
                            label="Encoder layers"
                            value={config.model.encLayers}
                            onChange={(value) => updateArchitecture("encLayers", value)}
                            min={1}
                            max={12}
                            hint="Depth for the encoder stack."
                        />
                        <NumberField
                            label="Decoder layers"
                            value={config.model.decLayers}
                            onChange={(value) => updateArchitecture("decLayers", value)}
                            min={1}
                            max={12}
                            hint="Depth for the decoder stack."
                        />
                        <NumberField
                            label="Attention heads"
                            value={config.model.nheads}
                            onChange={(value) => updateArchitecture("nheads", value)}
                            min={1}
                            max={16}
                            hint="Used for recurrent attention bridges."
                        />
                        <ToggleField
                            label="Attention bridge"
                            checked={config.model.useAttentionModule}
                            onChange={(value) => updateArchitecture("useAttentionModule", value)}
                            hint="Adds an external attention module on recurrent models."
                        />
                    </>
                )}
            </div>

            {/* ── Transformer: encoder / decoder split ──────────────────── */}
            {isTransformer && (
                <div className="enc-dec-split">
                    <div className="enc-dec-card enc-dec-encoder">
                        <div className="enc-dec-heading">Encoder</div>
                        <NumberField
                            label="Layers"
                            value={config.model.encLayers}
                            onChange={(value) => updateArchitecture("encLayers", value)}
                            min={1}
                            max={12}
                            hint="Encoder transformer blocks."
                        />
                        <NumberField
                            label="Heads"
                            value={config.model.nheads}
                            onChange={(value) => updateArchitecture("nheads", value)}
                            min={1}
                            max={16}
                            hint="Self-attention heads."
                        />
                        <SelectField
                            label="FFN type"
                            value={encoderFfnType}
                            onChange={(value) => updateArchitecture("encFfnType", value)}
                            options={ffnOptions}
                            hint="Encoder feed-forward block variant."
                        />
                        <NumberField
                            label="FF dimension"
                            value={encoderFfDim}
                            onChange={(value) => updateArchitecture("encFfDim", value)}
                            min={64}
                            max={4096}
                            step={64}
                            hint="Encoder inner FF dimension."
                        />
                        {encoderFfnType === "moe" && (
                            <>
                                <NumberField
                                    label="MoE experts"
                                    value={encoderNumExperts}
                                    onChange={(value) => updateArchitecture("encNumExperts", value)}
                                    min={2}
                                    max={32}
                                    hint="Encoder expert count."
                                />
                                <NumberField
                                    label="Top-k routes"
                                    value={encoderTopK}
                                    onChange={(value) => updateArchitecture("encTopK", value)}
                                    min={1}
                                    max={8}
                                    hint="Top-k experts selected per token in the encoder."
                                />
                            </>
                        )}
                    </div>
                    <div className="enc-dec-card enc-dec-decoder">
                        <div className="enc-dec-heading">Decoder</div>
                        <NumberField
                            label="Layers"
                            value={config.model.decLayers}
                            onChange={(value) => updateArchitecture("decLayers", value)}
                            min={1}
                            max={12}
                            hint="Decoder transformer blocks."
                        />
                        <NumberField
                            label="Heads"
                            value={config.model.decNheads ?? config.model.nheads}
                            onChange={(value) => updateArchitecture("decNheads", value)}
                            min={1}
                            max={16}
                            hint="Cross-attention and self-attention heads."
                        />
                        <SelectField
                            label="FFN type"
                            value={decoderFfnType}
                            onChange={(value) => updateArchitecture("decFfnType", value)}
                            options={ffnOptions}
                            hint="Decoder feed-forward block variant."
                        />
                        <NumberField
                            label="FF dimension"
                            value={decoderFfDim}
                            onChange={(value) => updateArchitecture("decFfDim", value)}
                            min={64}
                            max={4096}
                            step={64}
                            hint="Decoder inner FF dimension."
                        />
                        {decoderFfnType === "moe" && (
                            <>
                                <NumberField
                                    label="MoE experts"
                                    value={decoderNumExperts}
                                    onChange={(value) => updateArchitecture("decNumExperts", value)}
                                    min={2}
                                    max={32}
                                    hint="Decoder expert count."
                                />
                                <NumberField
                                    label="Top-k routes"
                                    value={decoderTopK}
                                    onChange={(value) => updateArchitecture("decTopK", value)}
                                    min={1}
                                    max={8}
                                    hint="Top-k experts selected per token in the decoder."
                                />
                            </>
                        )}
                    </div>
                </div>
            )}

            {/* ── Transformer-specific options ──────────────────────────── */}
            {isTransformer && (
                <div className="tf-options">
                    <div className="tf-option-group">
                        <div className="tf-option-heading">Patch Tokenization</div>
                        <div className="field-grid">
                            <ToggleField
                                label="Patch encoder"
                                checked={config.model.patchEncoder}
                                onChange={(value) => updateArchitecture("patchEncoder", value)}
                                hint="Divide the sequence into fixed-length patch tokens (PatchTST style)."
                            />
                            <ToggleField
                                label="Patch decoder"
                                checked={config.model.patchDecoder ?? false}
                                onChange={(value) => updateArchitecture("patchDecoder", value)}
                                hint="Patch the decoder too. Not recommended for incremental decoding."
                            />
                            <ToggleField
                                label="Channel-independent"
                                checked={config.model.channelIndep ?? true}
                                onChange={(value) => updateArchitecture("channelIndep", value)}
                                hint="Process each variate independently. Reduces overfitting on small multivariate datasets."
                            />
                            {config.model.patchEncoder && (
                                <>
                                    <NumberField
                                        label="Patch length"
                                        value={config.model.patchLen}
                                        onChange={(value) => updateArchitecture("patchLen", value)}
                                        min={4}
                                        max={64}
                                        step={4}
                                        hint="Length of each patch token in time steps."
                                    />
                                    <NumberField
                                        label="Patch stride"
                                        value={config.model.patchStride}
                                        onChange={(value) => updateArchitecture("patchStride", value)}
                                        min={1}
                                        max={32}
                                        step={1}
                                        hint="Step between consecutive patches. Overlap when stride < patch length."
                                    />
                                </>
                            )}
                        </div>
                    </div>

                    <div className="tf-option-group">
                        <div className="tf-option-heading">Attention</div>
                        <div className="field-grid">
                            <SelectField
                                label="Attention mode"
                                value={config.model.attentionMode}
                                onChange={(value) => updateArchitecture("attentionMode", value)}
                                options={ATTENTION_MODE_OPTIONS.map(({ value, label }) => ({ value, label }))}
                                hint={ATTN_DESCRIPTIONS[config.model.attentionMode] ?? "Matches transformer.py attention_mode routing."}
                            />
                            <SelectField
                                label="Attention backend"
                                value={config.model.attentionBackend}
                                onChange={(value) => updateArchitecture("attentionBackend", value)}
                                options={[
                                    { value: "torch", label: "PyTorch (sdpa)" },
                                    { value: "flash", label: "FlashAttention" },
                                    { value: "triton", label: "Triton kernel" },
                                ]}
                                hint="Execution backend for the attention kernel. Flash requires compatible GPU."
                            />
                        </div>
                        <div className="attention-mode-catalog">
                            {(() => {
                                const selectedAttention = ATTENTION_MODE_OPTIONS.find(
                                    (option) => option.value === config.model.attentionMode,
                                ) ?? ATTENTION_MODE_OPTIONS[0];

                                return (
                                    <div className="attention-mode-card attention-mode-card-active">
                                        <div className="attention-mode-card-head">
                                            <span className="attention-mode-card-name">{selectedAttention.label}</span>
                                            <span className="attention-mode-card-chip">{selectedAttention.value}</span>
                                        </div>
                                        <div className="attention-mode-card-copy">{selectedAttention.description}</div>
                                    </div>
                                );
                            })()}
                        </div>
                    </div>

                    <div className="tf-option-group">
                        <div className="tf-option-heading">Normalization</div>
                        <div className="field-grid">
                            <SelectField
                                label="Norm layer"
                                value={config.model.normType ?? "rms"}
                                onChange={(value) => updateArchitecture("normType", value)}
                                options={[
                                    { value: "rms", label: "RMSNorm" },
                                    { value: "layernorm", label: "LayerNorm" },
                                ]}
                                hint="Maps to custom_norm in transformer.py."
                            />
                            <SelectField
                                label="Norm strategy"
                                value={config.model.normStrategy ?? "pre_norm"}
                                onChange={(value) => updateArchitecture("normStrategy", value)}
                                options={[
                                    { value: "pre_norm", label: "Pre-norm" },
                                    { value: "post_norm", label: "Post-norm" },
                                    { value: "sandwich_norm", label: "Sandwich norm" },
                                ]}
                                hint="Residual/norm ordering used by each transformer block."
                            />
                            <ToggleField
                                label="Final norm"
                                checked={config.model.useFinalNorm ?? true}
                                onChange={(value) => updateArchitecture("useFinalNorm", value)}
                                hint="Apply the final output normalization layer on the stack."
                            />
                        </div>
                    </div>
                </div>
            )}

            <div className="tf-option-group headstack-section">
                <div className="tf-option-heading">Head Stack</div>
                <div className="headstack-group-grid">
                    <div className="headstack-group-card headstack-group-revin">
                        <div className="headstack-group-heading">RevIN</div>
                        <div className="headstack-selector-copy">
                            Invertible normalization before the backbone. Useful when the series shifts in scale across time.
                        </div>
                        <ToggleField
                            label="RevINHead"
                            checked={useRevIN}
                            onChange={(value) => updateHeadOption("useRevIN", value)}
                            hint="Invertible normalization attached through HeadComposer before the backbone."
                        />
                    </div>

                    <div className="headstack-group-card headstack-group-decomposition">
                        <div className="headstack-group-heading">Decomposition</div>
                        <div className="headstack-selector-copy">
                            Split trend and seasonal components before the model so deterministic drift is handled explicitly.
                        </div>
                        <ToggleField
                            label="DecompositionBlock"
                            checked={useDecomposition}
                            onChange={(value) => updateHeadOption("useDecomposition", value)}
                            hint="Splits the signal into seasonal and trend components with additive inversion."
                        />
                    </div>

                    <div className="headstack-group-card headstack-group-multiscale">
                        <div className="headstack-group-heading">MultiScale</div>
                        <div className="headstack-selector-copy">
                            Add a temporal pyramid with optional channel mixing and NHITS-like top-down residual refinement.
                        </div>
                        <ToggleField
                            label="MultiScaleConvHead"
                            checked={useMultiScaleConv}
                            onChange={(value) => updateHeadOption("useMultiScaleConv", value)}
                            hint="Forward multiscale pyramid with optional FFT filtering and channel mixing."
                        />
                    </div>
                </div>

                {hasActiveHeadConfigs && (
                    <div className="headstack-config-stack">
                        {useRevIN && (
                            <div className="headstack-config-card headstack-group-revin">
                                <div className="headstack-group-heading">RevIN Configuration</div>
                                <div className="headstack-config-empty">
                                    RevINHead is active. There are no extra tuning fields exposed here yet, so it stays visually separated without adding noise.
                                </div>
                            </div>
                        )}

                        {useDecomposition && (
                            <div className="headstack-config-card headstack-group-decomposition">
                                <div className="headstack-group-heading">Decomposition Configuration</div>
                                <div className="field-grid">
                                    <NumberField
                                        label="Decomposition kernel"
                                        value={headConfig.decompositionKernelSize ?? 25}
                                        onChange={(value) => updateHeadOption("decompositionKernelSize", value)}
                                        min={3}
                                        max={129}
                                        step={2}
                                        hint="Odd moving-average kernel used to estimate the trend path."
                                    />
                                    <NumberField
                                        label="Decomposition hidden dim"
                                        value={headConfig.decompositionHiddenDim ?? 0}
                                        onChange={(value) => updateHeadOption("decompositionHiddenDim", value)}
                                        min={0}
                                        max={1024}
                                        step={8}
                                        hint="Set 0 to keep the original feature width after the seasonal projection."
                                    />
                                </div>
                            </div>
                        )}

                        {useMultiScaleConv && (
                            <div className="headstack-config-card headstack-group-multiscale">
                                <div className="headstack-group-heading">MultiScale Configuration</div>
                                <div className="headstack-nested-grid">
                                    <div className="headstack-nested-card">
                                        <div className="headstack-nested-heading">Pyramid</div>
                                        <div className="field-grid">
                                            <NumberField
                                                label="MS scales"
                                                value={headConfig.multiScaleNumScales ?? 5}
                                                onChange={(value) => updateHeadOption("multiScaleNumScales", value)}
                                                min={1}
                                                max={8}
                                                hint="Depth of the coarse-to-fine pyramid."
                                            />
                                            <NumberField
                                                label="MS pool factor"
                                                value={headConfig.multiScalePoolFactor ?? 2}
                                                onChange={(value) => updateHeadOption("multiScalePoolFactor", value)}
                                                min={2}
                                                max={4}
                                                hint="Temporal downsampling ratio between scales."
                                            />
                                            <ToggleField
                                                label="FFT filter"
                                                checked={headConfig.multiScaleUseFftFilter ?? true}
                                                onChange={(value) => updateHeadOption("multiScaleUseFftFilter", value)}
                                                hint="Adds per-scale spectral filtering before hierarchical fusion."
                                            />
                                            <NumberField
                                                label="Fuse dropout"
                                                value={headConfig.multiScaleFuseDropout ?? 0}
                                                onChange={(value) => updateHeadOption("multiScaleFuseDropout", value)}
                                                min={0}
                                                max={0.6}
                                                step={0.05}
                                                hint="Dropout inside the hierarchical fusion projections."
                                            />
                                            <ToggleField
                                                label="Residual to input"
                                                checked={headConfig.multiScaleKeepInputResidual ?? true}
                                                onChange={(value) => updateHeadOption("multiScaleKeepInputResidual", value)}
                                                hint="Returns x + multiscale features instead of only the fused multiscale path."
                                            />
                                            <ToggleField
                                                label="NHITS-like refinement"
                                                checked={headConfig.multiScaleUseNhitsRefinement ?? false}
                                                onChange={(value) => updateHeadOption("multiScaleUseNhitsRefinement", value)}
                                                hint="Treats the coarse upsampled path as the top-down base and refines the residual at each finer scale."
                                            />
                                            <SelectField
                                                label="Interpolation"
                                                value={headConfig.multiScaleInterpolationMode ?? "linear"}
                                                onChange={(value) => updateHeadOption("multiScaleInterpolationMode", value)}
                                                options={[
                                                    { value: "linear", label: "Linear" },
                                                    { value: "nearest", label: "Nearest" },
                                                ]}
                                                hint="Upsampling rule used for the coarse top-down path."
                                            />
                                        </div>
                                    </div>

                                    <div className="headstack-nested-card">
                                        <div className="headstack-nested-heading">Channel Mixer</div>
                                        <div className="field-grid">
                                            <ToggleField
                                                label="Channel mixer"
                                                checked={headConfig.multiScaleUseChannelMixer ?? true}
                                                onChange={(value) => updateHeadOption("multiScaleUseChannelMixer", value)}
                                                hint="Parallel feature mixer summed with the pyramid output."
                                            />
                                            {headConfig.multiScaleUseChannelMixer ?? true ? (
                                                <>
                                                    <SelectField
                                                        label="Mixer type"
                                                        value={headConfig.multiScaleChannelMixerType ?? "mlp"}
                                                        onChange={(value) => updateHeadOption("multiScaleChannelMixerType", value)}
                                                        options={[
                                                            { value: "mlp", label: "MLP" },
                                                            { value: "conv1x1", label: "1x1 conv" },
                                                        ]}
                                                        hint="Feature mixing branch used in parallel to the temporal pyramid."
                                                    />
                                                    <NumberField
                                                        label="Mixer hidden mult"
                                                        value={headConfig.multiScaleChannelMixerHiddenMult ?? 2}
                                                        onChange={(value) => updateHeadOption("multiScaleChannelMixerHiddenMult", value)}
                                                        min={1}
                                                        max={8}
                                                        step={0.5}
                                                        hint="Only used by the MLP mixer. Multiplies feature width for the hidden layer."
                                                    />
                                                    <NumberField
                                                        label="Mixer dropout"
                                                        value={headConfig.multiScaleChannelMixerDropout ?? 0}
                                                        onChange={(value) => updateHeadOption("multiScaleChannelMixerDropout", value)}
                                                        min={0}
                                                        max={0.6}
                                                        step={0.05}
                                                        hint="Dropout inside the channel mixer branch."
                                                    />
                                                    <ToggleField
                                                        label="Mixer residual"
                                                        checked={headConfig.multiScaleChannelMixerResidual ?? true}
                                                        onChange={(value) => updateHeadOption("multiScaleChannelMixerResidual", value)}
                                                        hint="Adds the channel mixer output back to its input."
                                                    />
                                                </>
                                            ) : null}
                                        </div>
                                    </div>

                                    {(headConfig.multiScaleUseNhitsRefinement ?? false) && (
                                        <div className="headstack-nested-card">
                                            <div className="headstack-nested-heading">Refinement</div>
                                            <div className="field-grid">
                                                <NumberField
                                                    label="Refinement hidden mult"
                                                    value={headConfig.multiScaleRefinementHiddenMult ?? 2}
                                                    onChange={(value) => updateHeadOption("multiScaleRefinementHiddenMult", value)}
                                                    min={1}
                                                    max={8}
                                                    step={0.5}
                                                    hint="Width multiplier for the per-scale residual refinement block."
                                                />
                                                <NumberField
                                                    label="Refinement dropout"
                                                    value={headConfig.multiScaleRefinementDropout ?? 0}
                                                    onChange={(value) => updateHeadOption("multiScaleRefinementDropout", value)}
                                                    min={0}
                                                    max={0.6}
                                                    step={0.05}
                                                    hint="Dropout inside the NHITS-like refinement block."
                                                />
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {headRecommendations.length > 0 && (
                    <div className="callout dataset-callout">
                        <div className="callout-title">Diagnostics hint</div>
                        <div className="callout-copy dataset-status-copy">
                            {headRecommendations.join(" ")}
                        </div>
                    </div>
                )}

            </div>

            <ArchDiagram config={config} />
        </Panel>
    );
}

export function TrainingStep({ config, setConfig }) {
    const updateTraining = (key, value) => {
        setConfig((current) => ({
            ...current,
            train: {
                ...current.train,
                [key]: value,
            },
        }));
    };

    return (
        <Panel title="Training Strategy" kicker="Step 04">
            <div className="field-grid">
                <NumberField
                    label="Epochs"
                    value={config.train.epochs}
                    onChange={(value) => updateTraining("epochs", value)}
                    min={5}
                    max={400}
                    hint="Total training passes."
                />
                <NumberField
                    label="Batch size"
                    value={config.train.batchSize}
                    onChange={(value) => updateTraining("batchSize", value)}
                    min={4}
                    max={512}
                    hint="Window batches per optimization step."
                />
                <NumberField
                    label="Learning rate"
                    value={config.train.lr}
                    onChange={(value) => updateTraining("lr", value)}
                    min={0.00001}
                    max={0.1}
                    step={0.0001}
                    hint="Optimizer step size."
                />
                <NumberField
                    label="Patience"
                    value={config.train.patience}
                    onChange={(value) => updateTraining("patience", value)}
                    min={1}
                    max={100}
                    hint="Early-stopping patience in epochs."
                />
                <SelectField
                    label="Scheduler"
                    value={config.train.schedulerType}
                    onChange={(value) => updateTraining("schedulerType", value)}
                    options={[
                        { value: "none", label: "None" },
                        { value: "step", label: "Step LR" },
                        { value: "cosine", label: "Cosine" },
                    ]}
                    hint="Learning-rate schedule exposed in TrainingConfig."
                />
                <ToggleField
                    label="Auto-track with MLTracker"
                    checked={config.train.autoTrack}
                    onChange={(value) => updateTraining("autoTrack", value)}
                    hint="Attach run metadata and artifacts."
                />
            </div>
        </Panel>
    );
}
