import { FAMILY_META } from "./studio-data.js";

function pythonValue(value) {
  if (typeof value === "boolean") {
    return value ? "True" : "False";
  }
  if (typeof value === "number") {
    return String(value);
  }
  if (value == null) {
    return "None";
  }
  return `"${String(value)}"`;
}

function hasSelectedHeads(heads = {}) {
  return Boolean(
    heads.useRevIN
    || heads.useDecomposition
    || heads.useMultiScaleConv,
  );
}

function ensureOddPositive(value, minimum = 3) {
  const normalized = Math.max(minimum, Math.round(Number.isFinite(value) ? value : minimum));
  return normalized % 2 === 1 ? normalized : normalized + 1;
}

function buildImports(config) {
  const modelFamily = config.model.family;
  const heads = config.model.heads ?? {};
  const base = [
    "import numpy as np",
    "import pandas as pd",
    "import torch.nn as nn",
    "",
    "from foreblocks import (",
    "    ForecastingModel,",
    "    ModelConfig,",
    "    TimeSeriesHandler,",
    "    Trainer,",
    "    TrainingConfig,",
    "    create_dataloaders,",
  ];

  if (modelFamily === "lstm") {
    base.push("    LSTMEncoder,", "    LSTMDecoder,");
  }
  if (modelFamily === "gru") {
    base.push("    GRUEncoder,", "    GRUDecoder,");
  }
  if (modelFamily === "transformer") {
    base.push("    TransformerEncoder,", "    TransformerDecoder,");
  }
  if (modelFamily === "lstm" || modelFamily === "gru") {
    base.push("    AttentionLayer,");
  }

  base.push(")");

  if (hasSelectedHeads(heads)) {
    base.push("");
    base.push("from foreblocks.core.heads.head_helper import HeadComposer, HeadSpec");
    if (heads.useRevIN) {
      base.push("from foreblocks.core.heads.revin_head import RevINHead");
    }
    if (heads.useDecomposition) {
      base.push("from foreblocks.core.heads.decomposition_head import DecompositionBlock");
    }
    if (heads.useMultiScaleConv) {
      base.push("from foreblocks.core.heads.multiscale_conv_head import MultiScaleConvHead");
    }
  }

  return base.join("\n");
}

function buildHeadComposerSection(config) {
  const heads = config.model.heads ?? {};
  const lines = [];
  const decompositionHiddenDim = Number(heads.decompositionHiddenDim ?? 0);
  const decompositionKernelSize = ensureOddPositive(heads.decompositionKernelSize ?? 25, 3);

  if (!hasSelectedHeads(heads)) {
    return "";
  }

  lines.push("head_specs = []");

  if (heads.useRevIN) {
    lines.push(
      "head_specs.append(",
      "    HeadSpec(",
      "        head=RevINHead(feature_dim=X.shape[-1], affine=True),",
      '        name="revin",',
      '        combine="invert",',
      "    )",
      ")",
    );
  }

  if (heads.useDecomposition) {
    lines.push(
      "head_specs.append(",
      "    HeadSpec(",
      "        head=DecompositionBlock(",
      `            kernel_size=${decompositionKernelSize},`,
      "            feature_dim=X.shape[-1],",
      `            hidden_dim=${decompositionHiddenDim > 0 ? decompositionHiddenDim : "None"},`,
      "        ),",
      '        name="decomposition",',
      '        combine="add",',
      "    )",
      ")",
    );
  }

  if (heads.useMultiScaleConv) {
    lines.push(
      "head_specs.append(",
      "    HeadSpec(",
      "        head=MultiScaleConvHead(",
      "            feature_dim=X.shape[-1],",
      `            num_scales=${heads.multiScaleNumScales ?? 5},`,
      `            pool_factor=${heads.multiScalePoolFactor ?? 2},`,
      `            use_fft_filter=${pythonValue(heads.multiScaleUseFftFilter ?? true)},`,
      '            fft_init="identity",',
      `            fuse_dropout=${heads.multiScaleFuseDropout ?? 0.0},`,
      `            keep_input_residual=${pythonValue(heads.multiScaleKeepInputResidual ?? true)},`,
      `            use_channel_mixer=${pythonValue(heads.multiScaleUseChannelMixer ?? true)},`,
      `            channel_mixer_type=${pythonValue(heads.multiScaleChannelMixerType ?? "mlp")},`,
      `            channel_mixer_hidden_mult=${heads.multiScaleChannelMixerHiddenMult ?? 2.0},`,
      `            channel_mixer_dropout=${heads.multiScaleChannelMixerDropout ?? 0.0},`,
      `            channel_mixer_residual=${pythonValue(heads.multiScaleChannelMixerResidual ?? true)},`,
      `            use_nhits_style_refinement=${pythonValue(heads.multiScaleUseNhitsRefinement ?? false)},`,
      `            interpolation_mode=${pythonValue(heads.multiScaleInterpolationMode ?? "linear")},`,
      `            refinement_hidden_mult=${heads.multiScaleRefinementHiddenMult ?? 2.0},`,
      `            refinement_dropout=${heads.multiScaleRefinementDropout ?? 0.0},`,
      "        ),",
      '        name="multiscale_conv",',
      '        combine="none",',
      "    )",
      ")",
    );
  }

  lines.push("");
  lines.push("composer = HeadComposer(");
  lines.push("    specs=head_specs,");
  lines.push('    composer_mode="serial",');
  lines.push('    serial_none_merge="replace",');
  lines.push(")");
  lines.push("model.set_head_composer(composer)");

  return lines.join("\n");
}

function buildModelSection(config) {
  const { prep, model } = config;
  const encoderFfDim = model.encFfDim ?? model.ffDim;
  const decoderFfDim = model.decFfDim ?? model.ffDim;
  const encoderFfnType = model.encFfnType ?? model.ffnType ?? "swiglu";
  const decoderFfnType = model.decFfnType ?? model.ffnType ?? "swiglu";
  const encoderNumExperts = model.encNumExperts ?? model.numExperts ?? 4;
  const decoderNumExperts = model.decNumExperts ?? model.numExperts ?? 4;
  const encoderTopK = model.encTopK ?? model.topK ?? 2;
  const decoderTopK = model.decTopK ?? model.topK ?? 2;
  const encoderUseMoe = encoderFfnType === "moe";
  const decoderUseMoe = decoderFfnType === "moe";
  const encoderUseSwiglu = encoderFfnType !== "standard";
  const decoderUseSwiglu = decoderFfnType !== "standard";

  const modelConfigLines = [
    "model_cfg = ModelConfig(",
    `    model_type=${pythonValue(FAMILY_META[model.family].modelType)},`,
    "    input_size=X.shape[-1],",
    "    output_size=y.shape[-1],",
    `    hidden_size=${model.hiddenSize},`,
    `    seq_len=${prep.windowSize},`,
    `    target_len=${prep.horizon},`,
    `    strategy=${pythonValue(FAMILY_META[model.family].strategy)},`,
    `    dropout=${model.dropout},`,
    `    num_encoder_layers=${model.encLayers},`,
    `    num_decoder_layers=${model.decLayers},`,
  ];

  if (model.family === "transformer") {
    modelConfigLines.push(
      `    dim_feedforward=${encoderFfDim},`,
      `    nheads=${model.nheads},`,
    );
  }

  modelConfigLines.push(")");

  if (model.family === "direct") {
    return [
      modelConfigLines.join("\n"),
      "",
      "head = nn.Sequential(",
      "    nn.Flatten(),",
      "    nn.Linear(model_cfg.seq_len * model_cfg.input_size, model_cfg.hidden_size),",
      "    nn.GELU(),",
      "    nn.Dropout(model_cfg.dropout),",
      "    nn.Linear(model_cfg.hidden_size, model_cfg.target_len * model_cfg.output_size),",
      "    nn.Unflatten(1, (model_cfg.target_len, model_cfg.output_size)),",
      ")",
      "",
      "model = ForecastingModel(",
      "    head=head,",
      "    forecasting_strategy=model_cfg.strategy,",
      "    model_type=model_cfg.model_type,",
      "    target_len=model_cfg.target_len,",
      ")",
    ].join("\n");
  }

  const recurrentBlock =
    model.family === "lstm"
      ? [
          "encoder = LSTMEncoder(",
          "    input_size=model_cfg.input_size,",
          "    hidden_size=model_cfg.hidden_size,",
          "    num_layers=model_cfg.num_encoder_layers,",
          "    dropout=model_cfg.dropout,",
          ")",
          "",
          "decoder = LSTMDecoder(",
          "    input_size=model_cfg.output_size,",
          "    hidden_size=model_cfg.hidden_size,",
          "    output_size=model_cfg.output_size,",
          "    num_layers=model_cfg.num_decoder_layers,",
          "    dropout=model_cfg.dropout,",
          ")",
        ]
      : model.family === "gru"
        ? [
            "encoder = GRUEncoder(",
            "    input_size=model_cfg.input_size,",
            "    hidden_size=model_cfg.hidden_size,",
            "    num_layers=model_cfg.num_encoder_layers,",
            "    dropout=model_cfg.dropout,",
            ")",
            "",
            "decoder = GRUDecoder(",
            "    input_size=model_cfg.output_size,",
            "    hidden_size=model_cfg.hidden_size,",
            "    output_size=model_cfg.output_size,",
            "    num_layers=model_cfg.num_decoder_layers,",
            "    dropout=model_cfg.dropout,",
            ")",
          ]
        : [
            "encoder = TransformerEncoder(",
            "    input_size=model_cfg.input_size,",
            "    d_model=model_cfg.hidden_size,",
            "    nhead=model_cfg.nheads,",
            "    num_layers=model_cfg.num_encoder_layers,",
            "    dim_feedforward=model_cfg.dim_feedforward,",
            "    dropout=model_cfg.dropout,",
            `    attention_mode=${pythonValue(model.attentionMode)},`,
            `    norm_strategy=${pythonValue(model.normStrategy ?? "pre_norm")},`,
            `    custom_norm=${pythonValue(model.normType ?? "rms")},`,
            `    use_final_norm=${pythonValue(model.useFinalNorm ?? true)},`,
            `    use_swiglu=${pythonValue(encoderUseSwiglu)},`,
            `    use_moe=${pythonValue(encoderUseMoe)},`,
            `    num_experts=${encoderNumExperts},`,
            `    top_k=${encoderTopK},`,
            `    patch_encoder=${pythonValue(model.patchEncoder)},`,
            `    patch_decoder=${pythonValue(false)},`,
            `    patch_len=${model.patchLen},`,
            `    patch_stride=${model.patchStride},`,
            `    patch_pad_end=${pythonValue(model.patchPadEnd ?? true)},`,
            ")",
            "",
            "decoder = TransformerDecoder(",
            "    input_size=model_cfg.output_size,",
            "    output_size=model_cfg.output_size,",
            "    d_model=model_cfg.hidden_size,",
            `    nhead=${model.decNheads ?? model.nheads},`,
            "    num_layers=model_cfg.num_decoder_layers,",
            `    dim_feedforward=${decoderFfDim},`,
            "    dropout=model_cfg.dropout,",
            `    norm_strategy=${pythonValue(model.normStrategy ?? "pre_norm")},`,
            `    custom_norm=${pythonValue(model.normType ?? "rms")},`,
            `    use_final_norm=${pythonValue(model.useFinalNorm ?? true)},`,
            `    use_swiglu=${pythonValue(decoderUseSwiglu)},`,
            `    use_moe=${pythonValue(decoderUseMoe)},`,
            `    num_experts=${decoderNumExperts},`,
            `    top_k=${decoderTopK},`,
            `    patch_decoder=${pythonValue(model.patchDecoder ?? false)},`,
            `    patch_pad_end=${pythonValue(model.patchPadEnd ?? true)},`,
            "    informer_like=False,",
            ")",
          ];

  const attentionBlock =
    model.family !== "transformer" && model.useAttentionModule
      ? [
          "",
          "attention = AttentionLayer(",
          "    decoder_hidden_size=model_cfg.hidden_size,",
          "    encoder_hidden_size=model_cfg.hidden_size,",
          `    method=${pythonValue(model.attentionMethod)},`,
          `    attention_backend=${pythonValue(model.attentionBackend)},`,
          "    nhead=4,",
          "    dropout=model_cfg.dropout,",
          ")",
        ]
      : [];

  return [
    modelConfigLines.join("\n"),
    "",
    ...recurrentBlock,
    ...attentionBlock,
    "",
    "model = ForecastingModel(",
    "    encoder=encoder,",
    "    decoder=decoder,",
    ...(model.family !== "transformer" && model.useAttentionModule
      ? ["    attention_module=attention,"]
      : []),
    "    forecasting_strategy=model_cfg.strategy,",
    `    model_type=${pythonValue(FAMILY_META[model.family].modelType)},`,
    "    target_len=model_cfg.target_len,",
    "    output_size=model_cfg.output_size,",
    "    hidden_size=model_cfg.hidden_size,",
    ...(model.family === "transformer"
      ? ["    label_len=model_cfg.target_len // 2,"]
      : ["    teacher_forcing_ratio=0.5,"]),
    ")",
  ].join("\n");
}

function mdCell(source) {
  return { cell_type: "markdown", metadata: {}, source };
}

function codeCell(source) {
  return { cell_type: "code", execution_count: null, metadata: {}, outputs: [], source };
}

export function buildNotebook(config, datasetInfo = null) {
  const code = buildPythonPipeline(config, datasetInfo);
  const lines = code.split("\n");

  const MARKERS = [
    { key: "data",  pattern: /^DATA_PATH\s*=/ },
    { key: "prep",  pattern: /^# Preprocess with the stable/ },
    { key: "model", pattern: /^model_cfg\s*=\s*ModelConfig/ },
    { key: "train", pattern: /^train_cfg\s*=\s*TrainingConfig/ },
  ];

  const boundaries = [0];
  const sectionKeys = ["imports"];

  for (let i = 0; i < lines.length; i++) {
    for (const marker of MARKERS) {
      if (marker.pattern.test(lines[i])) {
        boundaries.push(i);
        sectionKeys.push(marker.key);
        break;
      }
    }
  }
  boundaries.push(lines.length);

  const sections = sectionKeys.map((key, idx) => ({
    key,
    code: lines.slice(boundaries[idx], boundaries[idx + 1]).join("\n").trim(),
  }));

  const SECTION_HEADERS = {
    imports: "## Imports",
    data: "## Data Loading",
    prep: "## Preprocessing",
    model: "## Model Definition",
    train: "## Training",
  };

  const { prep, model } = config;
  const cells = [
    mdCell(
      `# foreBlocks Forecast Pipeline\n\nGenerated by foreBlocks Studio.\n\n` +
      `**Model family:** ${model.family}  \n` +
      `**Window:** ${prep.windowSize}  \n` +
      `**Horizon:** ${prep.horizon}  \n` +
      `**Scaling:** ${prep.scalingMethod}`,
    ),
  ];

  for (const section of sections) {
    if (!section.code) continue;
    cells.push(mdCell(SECTION_HEADERS[section.key] ?? `## ${section.key}`));
    cells.push(codeCell(section.code));
  }

  return JSON.stringify(
    {
      nbformat: 4,
      nbformat_minor: 5,
      metadata: {
        kernelspec: { display_name: "Python 3", language: "python", name: "python3" },
        language_info: { name: "python", pygments_lexer: "ipython3", version: "3.10.0" },
      },
      cells,
    },
    null,
    2,
  );
}

export function buildPythonPipeline(config, datasetInfo = null) {
  const { data, prep, train } = config;

  const trainingConfigLines = [
    "train_cfg = TrainingConfig(",
    `    num_epochs=${train.epochs},`,
    `    learning_rate=${train.lr},`,
    `    batch_size=${train.batchSize},`,
    `    patience=${train.patience},`,
    `    weight_decay=${train.weightDecay},`,
    `    use_amp=${pythonValue(train.useAmp)},`,
    `    scheduler_type=${train.schedulerType === "none" ? "None" : pythonValue(train.schedulerType)},`,
    `    experiment_name=${pythonValue(train.experimentName)},`,
    ")",
  ];

  return [
    buildImports(config),
    "",
    ...(datasetInfo?.kind === "upload"
      ? [
          `# Dataset source: uploaded browser file ${datasetInfo.name}`,
          "# Replace DATA_PATH with the real saved path before running this script.",
        ]
      : datasetInfo?.kind === "path"
        ? [`# Dataset source: ${datasetInfo.name}`]
        : []),
    `DATA_PATH = ${pythonValue(data.filename)}`,
    "",
    "# Load a raw [T, D] series from disk",
    "frame = pd.read_csv(DATA_PATH)",
    `timestamps = pd.to_datetime(frame[${pythonValue(data.timestamp)}])`,
    `raw = frame[[${pythonValue(data.target)}]].to_numpy(dtype=\"float32\")`,
    "",
    "# Preprocess with the stable foreBlocks pipeline",
    "pre = TimeSeriesHandler(",
    `    window_size=${prep.windowSize},`,
    `    horizon=${prep.horizon},`,
    `    normalize=${pythonValue(prep.normalize)},`,
    `    differencing=${pythonValue(prep.differencing)},`,
    `    detrend=${pythonValue(prep.detrend)},`,
    `    apply_filter=${pythonValue(prep.applyFilter)},`,
    `    filter_method=${pythonValue(prep.filterMethod)},`,
    `    apply_ewt=${pythonValue(prep.applyEwt)},`,
    `    self_tune=${pythonValue(prep.selfTune)},`,
    `    generate_time_features=${pythonValue(prep.generateTimeFeatures)},`,
    `    scaling_method=${pythonValue(prep.scalingMethod)},`,
    `    remove_outliers=${pythonValue(prep.removeOutliers)},`,
    `    outlier_method=${pythonValue(prep.outlierMethod)},`,
    `    apply_imputation=${pythonValue(prep.applyImputation)},`,
    `    impute_method=${pythonValue(prep.imputeMethod)},`,
    "    verbose=True,",
    ")",
    "",
    "X, y, processed, time_features = pre.fit_transform(raw, time_stamps=timestamps)",
    `split_idx = int(len(X) * ${data.trainSplit})`,
    "X_train, X_val = X[:split_idx], X[split_idx:]",
    "y_train, y_val = y[:split_idx], y[split_idx:]",
    "tf_train = time_features[:split_idx] if time_features is not None else None",
    "tf_val = time_features[split_idx:] if time_features is not None else None",
    "",
    "train_loader, val_loader = create_dataloaders(",
    "    X_train,",
    "    y_train,",
    "    X_val,",
    "    y_val,",
    `    batch_size=${train.batchSize},`,
    "    time_feat_train=tf_train,",
    "    time_feat_val=tf_val,",
    ")",
    "",
    buildModelSection(config),
    ...(buildHeadComposerSection(config)
      ? ["", "# Optional preprocessing head stack", buildHeadComposerSection(config)]
      : []),
    "",
    trainingConfigLines.join("\n"),
    "",
    "trainer = Trainer(",
    "    model,",
    "    config=train_cfg,",
    `    auto_track=${pythonValue(train.autoTrack)},`,
    ")",
    "",
    "history = trainer.train(train_loader, val_loader)",
    'print("last_train_loss:", history.train_losses[-1])',
    "if history.val_losses:",
    '    print("last_val_loss:", history.val_losses[-1])',
    'print("best_val_loss:", trainer.best_val_loss)',
  ].join("\n");
}
