export const PAPER_THEME = {
  name: "Paper",
  bg: "#eef3f8",
  bgAlt: "#f8fbff",
  panel: "rgba(255, 255, 255, 0.82)",
  panelSolid: "#ffffff",
  panelElevated: "#f3f7fc",
  border: "rgba(71, 85, 105, 0.13)",
  borderStrong: "rgba(59, 130, 246, 0.26)",
  text: "#162033",
  subtext: "#5b687d",
  muted: "#8b96a8",
  accent: "#3b82f6",
  accentStrong: "#2563eb",
  accentSoft: "rgba(59, 130, 246, 0.11)",
  secondary: "#6366f1",
  secondarySoft: "rgba(99, 102, 241, 0.12)",
  warm: "#0ea5a4",
  warmSoft: "rgba(14, 165, 164, 0.10)",
  cool: "#64748b",
  coolSoft: "rgba(100, 116, 139, 0.12)",
  codeBg: "#0d1320",
  codeTitle: "#c0d4f6",
  codeBodyText: "#dfebff",
  codeGutter: "#6f83a8",
  codeKeyword: "#7dd3fc",
  codeString: "#86efac",
  codeNumber: "#f9a8d4",
  codeComment: "#7c8aa5",
  codeFunction: "#fcd34d",
  codeClass: "#c4b5fd",
  codeOperator: "#93c5fd",
  shadow: "0 20px 48px rgba(15, 23, 42, 0.10)",
  glow: "0 0 0 1px rgba(59, 130, 246, 0.08), 0 18px 44px rgba(37, 99, 235, 0.08)",
  grid: "rgba(100, 116, 139, 0.14)",
  isDark: false,
};

export const DARK_THEME = {
  name: "Dark",
  bg: "#0e1726",
  bgAlt: "#0b1220",
  panel: "rgba(22, 36, 64, 0.85)",
  panelSolid: "#162240",
  panelElevated: "#1d2e50",
  border: "rgba(148, 163, 184, 0.11)",
  borderStrong: "rgba(96, 165, 250, 0.28)",
  text: "#daeaff",
  subtext: "#8099bb",
  muted: "#4a5e7a",
  accent: "#60a5fa",
  accentStrong: "#93c5fd",
  accentSoft: "rgba(96, 165, 250, 0.14)",
  secondary: "#818cf8",
  secondarySoft: "rgba(129, 140, 248, 0.14)",
  warm: "#2dd4bf",
  warmSoft: "rgba(45, 212, 191, 0.12)",
  cool: "#94a3b8",
  coolSoft: "rgba(148, 163, 184, 0.12)",
  codeBg: "#070d18",
  codeTitle: "#c0d4f6",
  codeBodyText: "#d8eaff",
  codeGutter: "#3d5275",
  codeKeyword: "#7dd3fc",
  codeString: "#86efac",
  codeNumber: "#f9a8d4",
  codeComment: "#3d5275",
  codeFunction: "#fcd34d",
  codeClass: "#c4b5fd",
  codeOperator: "#93c5fd",
  shadow: "0 20px 48px rgba(0, 0, 0, 0.48)",
  glow: "0 0 0 1px rgba(96, 165, 250, 0.12), 0 18px 44px rgba(37, 99, 235, 0.15)",
  grid: "rgba(100, 116, 139, 0.10)",
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
    filename: "sample_reservoir.csv",
    target: "inflow_m3s",
    timestamp: "date",
    freq: "auto",
    trainSplit: 0.8,
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
