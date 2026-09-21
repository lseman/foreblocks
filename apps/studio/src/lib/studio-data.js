// ─── Paper ────────────────────────────────────────────────────────────────────
// Crisp light: warm neutral base, indigo/cyan accents, amber + emerald signals
export const PAPER_THEME = {
  name: "Paper",
  bg: "#f1f5f9",           // slate-100
  bgAlt: "#f8fafc",        // slate-50
  panel: "rgba(255,255,255,0.75)",
  panelSolid: "#ffffff",
  panelElevated: "#f0f4f8",
  border: "rgba(15,23,42,0.09)",
  borderStrong: "rgba(37,99,235,0.30)",
  text: "#0f172a",         // slate-900
  subtext: "#475569",      // slate-600
  muted: "#94a3b8",        // slate-400
  accent: "#2563eb",       // blue-600
  accentStrong: "#1d4ed8", // blue-700
  accentSoft: "rgba(37,99,235,0.08)",
  secondary: "#0891b2",    // cyan-600
  secondarySoft: "rgba(8,145,178,0.09)",
  warm: "#b45309",         // amber-700 — dark enough to read on white
  warmSoft: "rgba(180,83,9,0.08)",
  cool: "#047857",         // emerald-700
  coolSoft: "rgba(4,120,87,0.08)",
  // surface tokens — replace all rgba(255,255,255,X) hardcodes
  surface1: "rgba(15,23,42,0.02)",   // barely-there tint
  surface2: "rgba(15,23,42,0.04)",   // faint card bg
  surface3: "rgba(255,255,255,0.70)", // main card / panel
  surface4: "rgba(255,255,255,0.90)", // inputs, active chips
  surface5: "rgba(255,255,255,0.95)", // table headers, elevated
  codeBg: "#0d1526",
  codeTitle: "#bfdbfe",
  codeBodyText: "#e2eeff",
  codeGutter: "#4d6391",
  codeKeyword: "#60a5fa",
  codeString: "#6ee7b7",
  codeNumber: "#f9a8d4",
  codeComment: "#5a7298",
  codeFunction: "#fbbf24",
  codeClass: "#c4b5fd",
  codeOperator: "#7dd3fc",
  shadow: "0 1px 2px rgba(15,23,42,0.06), 0 4px 16px rgba(15,23,42,0.07), 0 16px 40px rgba(15,23,42,0.06)",
  glow: "0 0 0 1px rgba(37,99,235,0.10), 0 4px 20px rgba(37,99,235,0.12)",
  grid: "rgba(15,23,42,0.06)",
  isDark: false,
};

// ─── Dark ─────────────────────────────────────────────────────────────────────
// True near-black zinc: amber accent, cyan secondary, clean high-contrast
export const DARK_THEME = {
  name: "Dark",
  bg: "#09090b",           // zinc-950
  bgAlt: "#0f0f12",
  panel: "rgba(24,24,27,0.90)",    // zinc-900 tinted
  panelSolid: "#18181b",           // zinc-900
  panelElevated: "#1f1f23",        // zinc-850
  border: "rgba(255,255,255,0.08)",
  borderStrong: "rgba(251,191,36,0.32)",
  text: "#fafafa",         // zinc-50
  subtext: "#a1a1aa",      // zinc-400
  muted: "#52525b",        // zinc-600
  accent: "#fbbf24",       // amber-400
  accentStrong: "#f59e0b", // amber-500
  accentSoft: "rgba(251,191,36,0.11)",
  secondary: "#22d3ee",    // cyan-400
  secondarySoft: "rgba(34,211,238,0.10)",
  warm: "#fb923c",         // orange-400
  warmSoft: "rgba(251,146,60,0.11)",
  cool: "#34d399",         // emerald-400
  coolSoft: "rgba(52,211,153,0.10)",
  // surface tokens — all dark-mode aware
  surface1: "rgba(255,255,255,0.03)",
  surface2: "rgba(255,255,255,0.05)",
  surface3: "rgba(255,255,255,0.07)",
  surface4: "rgba(255,255,255,0.09)",
  surface5: "rgba(255,255,255,0.11)",
  codeBg: "#07070a",
  codeTitle: "#fde68a",
  codeBodyText: "#f4f4f5",
  codeGutter: "#3f3f46",
  codeKeyword: "#67e8f9",
  codeString: "#6ee7b7",
  codeNumber: "#f9a8d4",
  codeComment: "#52525b",
  codeFunction: "#fbbf24",
  codeClass: "#d8b4fe",
  codeOperator: "#7dd3fc",
  shadow: "0 1px 2px rgba(0,0,0,0.40), 0 8px 24px rgba(0,0,0,0.50), 0 24px 56px rgba(0,0,0,0.40)",
  glow: "0 0 0 1px rgba(251,191,36,0.14), 0 4px 24px rgba(251,191,36,0.10)",
  grid: "rgba(255,255,255,0.035)",
  isDark: true,
};

// ─── Glass ────────────────────────────────────────────────────────────────────
// Midnight indigo: violet/sky pair, frosted glass panels, neon accents
export const GLASS_THEME = {
  name: "Glass",
  bg: "#05080f",           // near black with blue cast
  bgAlt: "#070b15",
  panel: "rgba(15,22,42,0.55)",
  panelSolid: "rgba(12,18,36,0.98)",
  panelElevated: "rgba(22,34,62,0.85)",
  border: "rgba(148,163,184,0.09)",
  borderStrong: "rgba(139,92,246,0.38)",
  text: "#f1f5f9",         // slate-100
  subtext: "#94a3b8",      // slate-400
  muted: "#64748b",        // slate-500 — readable on dark glass
  accent: "#818cf8",       // indigo-400
  accentStrong: "#6366f1", // indigo-500
  accentSoft: "rgba(129,140,248,0.13)",
  secondary: "#38bdf8",    // sky-400
  secondarySoft: "rgba(56,189,248,0.12)",
  warm: "#f472b6",         // pink-400
  warmSoft: "rgba(244,114,182,0.12)",
  cool: "#34d399",         // emerald-400
  coolSoft: "rgba(52,211,153,0.11)",
  // surface tokens
  surface1: "rgba(148,163,184,0.04)",
  surface2: "rgba(148,163,184,0.07)",
  surface3: "rgba(148,163,184,0.09)",
  surface4: "rgba(148,163,184,0.13)",
  surface5: "rgba(148,163,184,0.17)",
  codeBg: "#04060e",
  codeTitle: "#c7d2fe",
  codeBodyText: "#f1f5f9",
  codeGutter: "#334155",
  codeKeyword: "#a5b4fc",
  codeString: "#67e8f9",
  codeNumber: "#f9a8d4",
  codeComment: "#475569",
  codeFunction: "#fde68a",
  codeClass: "#c4b5fd",
  codeOperator: "#7dd3fc",
  shadow: "0 1px 2px rgba(0,0,0,0.50), 0 8px 32px rgba(4,6,20,0.65), 0 32px 72px rgba(4,6,20,0.55)",
  glow: "0 0 0 1px rgba(129,140,248,0.20), 0 4px 28px rgba(129,140,248,0.18)",
  grid: "rgba(129,140,248,0.05)",
  isDark: true,
};

export const STEPS = [
  "Dataset",
  "Preprocess",
  "Architecture",
  "Training",
  "Blueprint",
];

export const STEP_COPY = [
  "Point the studio to a raw time series and set the train/validation split.",
  "Shape the TimeSeriesHandler pipeline around filtering, imputation, and feature generation.",
  "Choose between the stable direct baseline, recurrent seq2seq blocks, or the transformer stack.",
  "Tune TrainingConfig and the trainer path that foreBlocks actually exposes.",
  "Review the generated script, the active API stack, and the resulting experiment posture.",
];

export const FAMILY_META = {
  direct: {
    label: "Direct Baseline",
    strategy: "direct",
    modelType: "head_only",
    badge: "Fastest stable path",
    detail:
      "Uses a direct MLP head on top of TimeSeriesHandler windows. Best baseline for first integrations.",
  },
  lstm: {
    label: "LSTM Seq2Seq",
    strategy: "seq2seq",
    modelType: "lstm",
    badge: "Recurrent memory",
    detail:
      "ForecastingModel with LSTMEncoder/LSTMDecoder and optional external AttentionLayer.",
  },
  gru: {
    label: "GRU Seq2Seq",
    strategy: "seq2seq",
    modelType: "lstm",
    badge: "Lean recurrent",
    detail:
      "GRU blocks trade a lighter recurrence path for strong baseline efficiency on medium series.",
  },
  transformer: {
    label: "Transformer Seq2Seq",
    strategy: "transformer_seq2seq",
    modelType: "transformer",
    badge: "Long-range context",
    detail:
      "TransformerEncoder/TransformerDecoder with patching and internal attention modes from the foreBlocks transformer stack.",
  },
};

export function bestNheads(hiddenSize) {
  for (let candidate = 16; candidate >= 1; candidate -= 1) {
    if (hiddenSize % candidate === 0) {
      return candidate;
    }
  }
  return 1;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function ensureOddPositive(value, minimum = 3) {
  const normalized = Math.max(minimum, Math.round(Number.isFinite(value) ? value : minimum));
  return normalized % 2 === 1 ? normalized : normalized + 1;
}

function bestTransformerHeads(hiddenSize) {
  const preferred = [8, 6, 4, 12, 16, 3, 2, 1];
  for (const candidate of preferred) {
    if (hiddenSize % candidate === 0) {
      return candidate;
    }
  }
  return 1;
}

function chooseTransformerHiddenSize(analysis, patchingDiagnostics) {
  const count = analysis.count ?? 0;
  const patchingLabel = patchingDiagnostics?.patchingLabel ?? analysis.patchingLabel;
  const multiscaleLabel = patchingDiagnostics?.multiscaleLabel ?? analysis.multiscaleLabel;

  if (count >= 480 && patchingLabel === "good" && analysis.forecastabilityLabel === "high") {
    return 256;
  }
  if (count >= 220 || multiscaleLabel === "high" || analysis.forecastabilityLabel === "high") {
    return 192;
  }
  if (count >= 120 || patchingLabel === "moderate") {
    return 160;
  }
  return 128;
}

function chooseAttentionMode(analysis, patchingDiagnostics) {
  const multiscaleLabel = patchingDiagnostics?.multiscaleLabel ?? analysis.multiscaleLabel ?? "low";
  const patchingLabel = patchingDiagnostics?.patchingLabel ?? analysis.patchingLabel ?? "weak";
  const roughness = patchingDiagnostics?.roughness ?? 0.8;
  const count = analysis.count ?? 0;
  const horizon = analysis.recommendedHorizon ?? 12;
  const windowSize = analysis.suggestedLagWindow ?? 48;

  if (analysis.changePointCount >= 3 || roughness >= 1.05) {
    return horizon >= 18 || windowSize >= 72 ? "hybrid_gdn" : "gated_delta";
  }
  if (multiscaleLabel === "high") {
    return count >= 240 || horizon >= 18 ? "hybrid_kimi" : "hybrid";
  }
  if (multiscaleLabel === "moderate" && count >= 180) {
    return "hybrid";
  }
  if (analysis.seasonalityStabilityLabel === "stable" && (analysis.dominantAcfLag ?? 0) >= 24 && count >= 220) {
    return "kimi";
  }
  if (patchingLabel === "good" && analysis.forecastabilityLabel !== "low" && count >= 160) {
    return "linear";
  }
  return analysis.transformerAttentionMode ?? "standard";
}

function choosePatchGeometry(currentConfig, analysis, patchingDiagnostics) {
  const fallbackLength = clamp(Math.round((analysis.suggestedLagWindow ?? currentConfig.prep.windowSize ?? 48) / 4), 4, 32);
  const patchLen = clamp(
    patchingDiagnostics?.available
      ? (patchingDiagnostics.recommendedPatchLen || fallbackLength)
      : (currentConfig.model.patchLen ?? fallbackLength),
    4,
    32,
  );
  const patchStride = clamp(
    patchingDiagnostics?.available
      ? (patchingDiagnostics.recommendedStride || Math.max(1, Math.round(patchLen / 2)))
      : (currentConfig.model.patchStride ?? Math.max(1, Math.round(patchLen / 2))),
    1,
    patchLen,
  );
  const windowSize = analysis.suggestedLagWindow ?? currentConfig.prep.windowSize ?? patchLen;
  const patchPadEnd = windowSize < patchLen || Math.max(0, windowSize - patchLen) % Math.max(1, patchStride) !== 0;
  const patchEncoder = patchingDiagnostics?.available
    ? patchingDiagnostics.patchingLabel !== "weak" || windowSize >= 48
    : true;

  return {
    patchEncoder,
    patchLen,
    patchStride,
    patchPadEnd,
  };
}

export function createDefaultHeadConfig() {
  return {
    useRevIN: true,
    useDecomposition: false,
    decompositionKernelSize: 25,
    decompositionHiddenDim: 0,
    useMultiScaleConv: false,
    multiScaleNumScales: 5,
    multiScalePoolFactor: 2,
    multiScaleUseFftFilter: true,
    multiScaleFuseDropout: 0.0,
    multiScaleKeepInputResidual: true,
    multiScaleUseChannelMixer: true,
    multiScaleChannelMixerType: "mlp",
    multiScaleChannelMixerHiddenMult: 2.0,
    multiScaleChannelMixerDropout: 0.0,
    multiScaleChannelMixerResidual: true,
    multiScaleUseNhitsRefinement: false,
    multiScaleInterpolationMode: "linear",
    multiScaleRefinementHiddenMult: 2.0,
    multiScaleRefinementDropout: 0.0,
  };
}

export function applyBlueprint(currentConfig, analysis, patchingDiagnostics = null) {
  const family = "transformer";
  const hiddenSize = chooseTransformerHiddenSize(analysis, patchingDiagnostics);
  const currentHeads = currentConfig.model.heads ?? createDefaultHeadConfig();
  const patchGeometry = choosePatchGeometry(currentConfig, analysis, patchingDiagnostics);
  const attentionMode = chooseAttentionMode(analysis, patchingDiagnostics);
  const headCount = bestTransformerHeads(hiddenSize);
  const seasonalPeriod = analysis.dominantAcfLag ?? analysis.suggestedLagWindow ?? 24;
  const useMultiScaleHead = ["moderate", "high"].includes(patchingDiagnostics?.multiscaleLabel ?? analysis.multiscaleLabel);
  const useDecompositionHead = analysis.needsDetrend || analysis.seasonalityStabilityLabel === "stable" || seasonalPeriod >= 12;
  const useRevINHead = analysis.needsDiff
    || analysis.stationarityLabel !== "stationary"
    || analysis.volatilityLabel === "volatile"
    || analysis.outlierRate > 0.02
    || analysis.changePointCount > 0;
  const useMoeFfn = (analysis.count ?? 0) >= 720 && analysis.forecastabilityLabel === "high";
  const encoderLayers = analysis.count >= 320 ? 4 : 3;
  const decoderLayers = analysis.recommendedHorizon >= 18 || analysis.count >= 220 ? 3 : 2;
  const dropout = analysis.forecastabilityLabel === "low" || analysis.changePointCount > 0 ? 0.15 : 0.1;
  const decompositionKernelSize = ensureOddPositive(
    clamp(Math.round(seasonalPeriod), 9, 49),
    3,
  );
  const multiScaleNumScales = useMultiScaleHead
    ? (patchingDiagnostics?.multiscaleLabel === "high" ? 5 : 4)
    : currentHeads.multiScaleNumScales;
  const multiScaleUseNhitsRefinement = useMultiScaleHead
    && patchingDiagnostics?.multiscaleLabel === "high"
    && (analysis.changePointCount ?? 0) <= 1;

  return {
    ...currentConfig,
    data: {
      ...currentConfig.data,
      freq: analysis.frequencyHint,
    },
    prep: {
      ...currentConfig.prep,
      windowSize: analysis.suggestedLagWindow,
      horizon: analysis.recommendedHorizon,
      differencing: analysis.needsDiff,
      detrend: analysis.needsDetrend,
      applyFilter: analysis.needsFilter,
      generateTimeFeatures: analysis.useTimeFeatures,
      scalingMethod: analysis.scalingMethod,
      removeOutliers: analysis.outlierRate > 0.025,
      outlierMethod:
        analysis.outlierRate > 0.025 ? "iqr" : currentConfig.prep.outlierMethod,
      selfTune: true,
    },
    model: {
      ...currentConfig.model,
      family,
      hiddenSize,
      encLayers: encoderLayers,
      decLayers: decoderLayers,
      dropout,
      nheads: headCount,
      attentionMode,
      patchEncoder: patchGeometry.patchEncoder,
      patchLen: patchGeometry.patchLen,
      patchStride: patchGeometry.patchStride,
      patchPadEnd: patchGeometry.patchPadEnd,
      useAttentionModule: false,
      attentionMethod: currentConfig.model.attentionMethod,
      attentionBackend: "torch",
      ffDim: hiddenSize * 4,
      ffnType: useMoeFfn ? "moe" : "swiglu",
      normType: "rms",
      normStrategy: "pre_norm",
      useFinalNorm: true,
      revIn: useRevINHead,
      channelIndep: patchGeometry.patchEncoder,
      numExperts: useMoeFfn ? 4 : currentConfig.model.numExperts,
      topK: useMoeFfn ? 2 : currentConfig.model.topK,
      decNheads: headCount,
      encFfDim: hiddenSize * 4,
      decFfDim: hiddenSize * 4,
      encFfnType: useMoeFfn ? "moe" : "swiglu",
      decFfnType: useMoeFfn ? "moe" : "swiglu",
      encNumExperts: useMoeFfn ? 4 : currentConfig.model.encNumExperts,
      decNumExperts: useMoeFfn ? 4 : currentConfig.model.decNumExperts,
      encTopK: useMoeFfn ? 2 : currentConfig.model.encTopK,
      decTopK: useMoeFfn ? 2 : currentConfig.model.decTopK,
      patchDecoder: false,
      heads: {
        ...currentHeads,
        useRevIN: useRevINHead,
        useDecomposition: useDecompositionHead,
        useMultiScaleConv: useMultiScaleHead,
        decompositionKernelSize,
        multiScaleNumScales: useMultiScaleHead
          ? Math.max(currentHeads.multiScaleNumScales ?? multiScaleNumScales, multiScaleNumScales)
          : currentHeads.multiScaleNumScales,
        multiScaleUseFftFilter: useMultiScaleHead,
        multiScaleUseChannelMixer: useMultiScaleHead,
        multiScaleUseNhitsRefinement,
        multiScaleKeepInputResidual: useMultiScaleHead,
        multiScaleInterpolationMode: useMultiScaleHead && (analysis.recommendedHorizon ?? 12) >= 18 ? "linear" : currentHeads.multiScaleInterpolationMode,
      },
    },
    train: {
      ...currentConfig.train,
      lr: hiddenSize >= 256 ? 0.00015 : 0.0002,
      batchSize: hiddenSize >= 192 ? 24 : 32,
      patience: Math.max(8, Math.floor(currentConfig.train.epochs * 0.12)),
      weightDecay: 0.0001,
      schedulerType: "step",
      useAmp: false,
    },
  };
}

export const INITIAL_CONFIG = {
  data: {
    source: "csv",
    filename: "sample_reservoir.csv",
    target: "inflow_m3s",
    timestamp: "date",
    freq: "auto",
    trainSplit: 0.8,
    generator: {
      length: 240,
      baseline: 0,
      trendSlope: 0.0,
      seasonalityAmplitude: 10.0,
      seasonalityPeriod: 12,
      noiseStd: 1.0,
    },
  },
  prep: {
    windowSize: 72,
    horizon: 18,
    normalize: true,
    differencing: false,
    detrend: true,
    applyFilter: false,
    filterMethod: "savgol",
    applyEwt: false,
    selfTune: true,
    generateTimeFeatures: true,
    scalingMethod: "robust",
    removeOutliers: true,
    outlierMethod: "iqr",
    applyImputation: false,
    imputeMethod: "auto",
  },
  model: {
    family: "transformer",
    hiddenSize: 192,
    encLayers: 4,
    decLayers: 4,
    dropout: 0.1,
    ffDim: 768,
    nheads: 8,
    attentionMode: "standard",
    patchEncoder: true,
    patchLen: 16,
    patchStride: 8,
    patchPadEnd: false,
    useAttentionModule: false,
    attentionMethod: "mha",
    attentionBackend: "torch",
    ffnType: "swiglu",
    normType: "rms",
    normStrategy: "pre_norm",
    useFinalNorm: true,
    revIn: true,
    channelIndep: true,
    numExperts: 4,
    topK: 2,
    decNheads: 8,
    encFfDim: 768,
    decFfDim: 768,
    encFfnType: "swiglu",
    decFfnType: "swiglu",
    encNumExperts: 4,
    decNumExperts: 4,
    encTopK: 2,
    decTopK: 2,
    patchDecoder: false,
    heads: createDefaultHeadConfig(),
  },
  train: {
    epochs: 80,
    batchSize: 24,
    lr: 0.0002,
    patience: 10,
    weightDecay: 0.0001,
    schedulerType: "step",
    useAmp: false,
    autoTrack: false,
    experimentName: "studio_baseline",
  },
};
