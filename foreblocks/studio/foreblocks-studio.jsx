import { useEffect, useMemo, useState } from "react";

import "./foreblocks-studio.css";

import { Panel } from "./components/base.jsx";
import { StudioLogo } from "./components/studio-logo.jsx";
import {
    AutomationPanel,
    BacktestingPanel,
    CodePanel,
    DiagnosticsPanel,
    EmdPanel,
    EemdPanel,
    EwtPanel,
    VmdPanel,
    ExplorePanel,
    FeatureLabPanel,
    FilterPanel,
    IntermittencyPanel,
    NarrativePanel,
    OutlierPanel,
    PatchingPanel,
    OverviewPanel,
    RegimePanel,
    StationarityPanel,
    VolatilityPanel,
} from "./components/panels.jsx";
import {
    ArchitectureStep,
    DatasetStep,
    PreprocessStep,
    TrainingStep,
} from "./components/steps.jsx";
import {
    applyBlueprint,
    DARK_THEME,
    FAMILY_META,
    GLASS_THEME,
    INITIAL_CONFIG,
    PAPER_THEME,
    STEP_COPY,
    STEPS,
} from "./lib/studio-data.js";
import { inspectCsvText, parseSeriesCsvText } from "./lib/csv-loader.js";
import { buildNotebook, buildPythonPipeline } from "./lib/pipeline.js";

function createEmptyDatasetSummary() {
    return {
        rowCount: 0,
        columnCount: 0,
        missingCellCount: 0,
        targetColumn: "",
        timeColumn: "",
        validObservationCount: 0,
        missingTargetCount: 0,
        invalidTargetCount: 0,
        missingTimestampCount: 0,
        droppedRowCount: 0,
        exogenousCount: 0,
    };
}

function fallbackAnalysis(activeFamily) {
    return {
        count: 0,
        noisy: false,
        trendLabel: "computing",
        stationarityLabel: "computing",
        automationSummary: "Running stationarity tests and automation checks in the background worker.",
        automationFlags: [],
        adfStatistic: "0.000",
        adfCriticalValue: "-2.86",
        adfRejectUnitRoot: false,
        adfLagOrder: 0,
        kpssStatistic: "0.000",
        kpssCriticalValue: "0.463",
        kpssRejectStationarity: false,
        kpssBandwidth: 0,
        dominantPacfLag: 0,
        dominantAcfLag: 0,
        suggestedLagWindow: INITIAL_CONFIG.prep.windowSize,
        recommendedHorizon: INITIAL_CONFIG.prep.horizon,
        recommendedFamily: activeFamily,
        scalingMethod: INITIAL_CONFIG.prep.scalingMethod,
        frequencyHint: INITIAL_CONFIG.data.freq,
        pacfThreshold: "0.000",
        pacfPeaks: [],
        acfPeaks: [],
        needsDetrend: false,
        needsDiff: false,
        needsFilter: false,
        useTimeFeatures: INITIAL_CONFIG.prep.generateTimeFeatures,
        outlierRate: 0,
        transformerAttentionMode: "standard",
        changePointCount: 0,
        changePointLabel: "computing",
        residualWhitenessLabel: "computing",
        residualWhitenessPValue: 1,
        seasonalityStabilityLabel: "computing",
        exogenousStabilityLabel: "computing",
        forecastabilityScore: 0,
        forecastabilityLabel: "computing",
        spectralEntropy: 1,
        complexityLabel: "unknown",
        patchingLabel: "computing",
        patchingScore: 0,
        recommendedPatchLen: 0,
        recommendedPatchStride: 0,
        multiscaleLabel: "unknown",
    };
}

function buildExperimentNarrative({ config, analysis, automationDiagnostics, benchmark, spectralDiagnostics, forecastabilityDiagnostics, datasetSummary, familyMeta }) {
    if (!automationDiagnostics) {
        return {
            summary: "The studio is still assembling the experiment rationale from the active diagnostics.",
            reasons: [],
            tags: [],
        };
    }

    const reasons = [
        {
            title: "Backbone choice",
            body: `${familyMeta.label} stays fixed as the blueprint backbone, while the diagnostic layer guides transformer details such as attention mode, patch geometry, and optional preprocessing heads. The current baseline ladder is led by ${benchmark?.winner?.label ?? "the current baseline"}.`,
        },
        {
            title: "Temporal framing",
            body: `The script uses window ${config.prep.windowSize} and horizon ${config.prep.horizon}, anchored by PACF ${analysis.dominantPacfLag}, ACF ${analysis.dominantAcfLag}, and the recommended context ${analysis.suggestedLagWindow}.`,
        },
        {
            title: "Signal structure",
            body: `Stationarity is ${analysis.stationarityLabel}, seasonality is ${analysis.seasonalityStabilityLabel}, forecastability is ${analysis.forecastabilityLabel} with entropy ${analysis.spectralEntropy}, and patching fit is ${analysis.patchingLabel}.`,
        },
        {
            title: "Data posture",
            body: `Readiness is ${automationDiagnostics.readinessLabel} (${automationDiagnostics.readinessScore}/100), data quality is ${automationDiagnostics.dataQualityScore}/100, and completeness is ${datasetSummary.rowCount > 0 ? ((datasetSummary.validObservationCount / datasetSummary.rowCount) * 100).toFixed(1) : "0.0"}%.`,
        },
    ];

    if (spectralDiagnostics?.available) {
        reasons.push({
            title: "Seasonal evidence",
            body: `The dominant spectral period is ${spectralDiagnostics.dominantPeriod} with ${(spectralDiagnostics.normalizedPeakPower * 100).toFixed(1)}% normalized power, which supports the current seasonal framing.`,
        });
    }

    if (forecastabilityDiagnostics?.available && forecastabilityDiagnostics.label === "low") {
        reasons.push({
            title: "Complexity guardrail",
            body: `Forecastability is low, so the automation layer down-ranks unnecessarily complex backbones and biases toward simpler, more stable baselines.`,
        });
    }

    return {
        summary: `This blueprint was selected as the most defensible compromise between data quality, signal structure, backtest evidence, and automation confidence. The generated script mirrors those decisions directly in preprocessing, model family, and training defaults.`,
        reasons,
        tags: [
            `readiness ${automationDiagnostics.readinessLabel}`,
            `forecastability ${analysis.forecastabilityLabel}`,
            `family ${familyMeta.label}`,
            `window ${config.prep.windowSize}`,
            `horizon ${config.prep.horizon}`,
        ],
    };
}

function formatSeries(data) {
    return data.map((point, index) => ({
        ...point,
        month: point.timestamp
            ? (() => {
                const parsed = Date.parse(point.timestamp);
                if (!Number.isNaN(parsed)) {
                    return new Date(parsed).toLocaleDateString(undefined, {
                        month: "short",
                        day: "numeric",
                    });
                }
                return point.timestamp;
            })()
            : `T${String(index + 1).padStart(3, "0")}`,
    }));
}

function buildThemeStyle(theme) {
    return {
        "--bg": theme.bg,
        "--bg-alt": theme.bgAlt,
        "--panel": theme.panel,
        "--panel-solid": theme.panelSolid,
        "--panel-elevated": theme.panelElevated,
        "--border": theme.border,
        "--border-strong": theme.borderStrong,
        "--text": theme.text,
        "--subtext": theme.subtext,
        "--muted": theme.muted,
        "--accent": theme.accent,
        "--accent-strong": theme.accentStrong,
        "--accent-soft": theme.accentSoft,
        "--secondary": theme.secondary,
        "--secondary-soft": theme.secondarySoft,
        "--warm": theme.warm,
        "--warm-soft": theme.warmSoft,
        "--cool": theme.cool,
        "--cool-soft": theme.coolSoft,
        "--grid": theme.grid,
        "--shadow": theme.shadow,
        "--glow": theme.glow,
        "--code-bg": theme.codeBg,
        "--code-title": theme.codeTitle,
        "--code-body-text": theme.codeBodyText,
        "--code-gutter": theme.codeGutter,
        "--code-keyword": theme.codeKeyword,
        "--code-string": theme.codeString,
        "--code-number": theme.codeNumber,
        "--code-comment": theme.codeComment,
        "--code-function": theme.codeFunction,
        "--code-class": theme.codeClass,
        "--code-operator": theme.codeOperator,
        "--overlay-opacity": theme.isDark ? 0.08 : 0.14,
        "--cool-border": theme.coolSoft,
        "--warm-border": theme.warmSoft,
        "--button-text": "#f8fbff",
        colorScheme: theme.isDark ? "dark" : "light",
    };
}

function BlueprintStep({ analysis, config, familyMeta, onApplyBlueprint, onReset }) {
    return (
        <Panel
            title="Blueprint Review"
            kicker="Step 05"
            actions={
                <div className="topbar-actions">
                    <button className="ghost-button" type="button" onClick={onReset}>
                        Reset
                    </button>
                    <button className="solid-button" type="button" onClick={onApplyBlueprint}>
                        Re-apply recommendations
                    </button>
                </div>
            }
        >
            <div className="summary-card-grid">
                <div className="summary-card">
                    <div className="summary-card-title">Blueprint backbone</div>
                    <div className="architecture-name">{familyMeta.label}</div>
                    <div className="summary-card-copy">
                        Blueprint recommendations tune the transformer stack instead of switching model families.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Attention / patching</div>
                    <div className="architecture-name">{config.model.attentionMode}</div>
                    <div className="summary-card-copy">
                        {config.model.patchEncoder
                            ? `Patch encoder on with L ${config.model.patchLen} and S ${config.model.patchStride}.`
                            : "Patch encoder disabled for the current transformer layout."}
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Window / horizon</div>
                    <div className="architecture-name">
                        {config.prep.windowSize} / {config.prep.horizon}
                    </div>
                    <div className="summary-card-copy">
                        Temporal framing configured for TimeSeriesHandler.
                    </div>
                </div>
            </div>
        </Panel>
    );
}

export default function ForeblocksStudio() {
    const [activeStep, setActiveStep] = useState(0);
    const [config, setConfig] = useState(INITIAL_CONFIG);
    const [isCodeModalOpen, setIsCodeModalOpen] = useState(false);
    const [activeExploreTab, setActiveExploreTab] = useState("series");
    const [activeRightRailTab, setActiveRightRailTab] = useState("explore");
    const [activePrepLabSubgroup, setActivePrepLabSubgroup] = useState("outliers");
    const [activeEmdLikeSubgroup, setActiveEmdLikeSubgroup] = useState("emd");
    const [activeRegimeSubgroup, setActiveRegimeSubgroup] = useState("signal");
    const [uploadedDataset, setUploadedDataset] = useState(null);
    const [themeKey, setThemeKey] = useState(() => {
        if (typeof window !== "undefined") {
            return localStorage.getItem("studioTheme") || "glass";
        }
        return "dark";
    });
    const [changePointMethod, setChangePointMethod] = useState("segmentation");
    const [outlierPreviewMethod, setOutlierPreviewMethod] = useState(INITIAL_CONFIG.prep.outlierMethod);
    const [filterPreviewSettings, setFilterPreviewSettings] = useState({
        applyFilter: false,
        filterMethod: "none",
        filterWindow: 9,
        filterPolyorder: 2,
        lowessFraction: 0.12,
        detrend: false,
        detrenderMethod: "none",
        detrenderOrder: 2,
    });
    const [datasetState, setDatasetState] = useState({
        status: "loading",
        series: [],
        errorMessage: "",
        sourceLabel: INITIAL_CONFIG.data.filename,
        points: 0,
        headers: [],
        covariates: [],
        summary: createEmptyDatasetSummary(),
    });
    const [emdOptions, setEmdOptions] = useState({
        maxImfs: 6,
        maxSifts: 80,
        siftThreshold: 0.03,
    });
    const [eemdOptions, setEemdOptions] = useState({
        maxImfs: 6,
        ensembleSize: 8,
        noiseStdRatio: 0.15,
        siftThreshold: 0.08,
        maxSifts: 80,
    });
    const [ewtOptions, setEwtOptions] = useState({
        maxBands: 5,
        smoothingWindow: 7,
        detectThreshold: 0.05,
        gamma: 0.1,
    });
    const [vmdOptions, setVmdOptions] = useState({
        modeCount: 4,
        alpha: 2000,
        tolerance: 1e-7,
        maxIterations: 500,
    });

    const [diagnosticsState, setDiagnosticsState] = useState({
        status: "loading",
        analysis: null,
        acfCurve: [],
        pacfCurve: [],
        decomposition: null,
        benchmark: null,
        exogenousScreening: [],
        grangerDiagnostics: null,
        changePoints: null,
        residualDiagnostics: null,
        stabilityDiagnostics: null,
        spectralDiagnostics: null,
        forecastabilityDiagnostics: null,
        patchingDiagnostics: null,
        emdDiagnostics: null,
        eemdDiagnostics: null,
        ewtDiagnostics: null,
        vmdDiagnostics: null,
        calendarDiagnostics: null,
        intermittencyDiagnostics: null,
        volatilityDiagnostics: null,
        automationDiagnostics: null,
        errorMessage: "",
    });

    const theme = useMemo(() => {
        const themes = {
            dark: DARK_THEME,
            paper: PAPER_THEME,
            glass: GLASS_THEME,
        };
        return themes[themeKey] ?? DARK_THEME;
    }, [themeKey]);
    const isDarkMode = theme.isDark ?? false;
    const themeStyle = useMemo(() => buildThemeStyle(theme), [theme]);
    const seriesData = useMemo(
        () => formatSeries(datasetState.series.slice(-Math.min(datasetState.series.length, 240))),
        [datasetState.series],
    );
    const regimeSeriesData = useMemo(
        () => formatSeries(datasetState.series),
        [datasetState.series],
    );
    const familyMeta = FAMILY_META[config.model.family];
    const analysis = diagnosticsState.analysis ?? fallbackAnalysis(config.model.family);
    const datasetInfo = useMemo(() => {
        if (uploadedDataset && uploadedDataset.name === config.data.filename) {
            return { kind: "upload", name: uploadedDataset.name };
        }
        return { kind: "path", name: config.data.filename };
    }, [config.data.filename, uploadedDataset]);
    const code = useMemo(() => buildPythonPipeline(config, datasetInfo), [config, datasetInfo]);
    const notebook = useMemo(() => buildNotebook(config, datasetInfo), [config, datasetInfo]);
    const experimentNarrative = useMemo(() => buildExperimentNarrative({
        config,
        analysis,
        automationDiagnostics: diagnosticsState.automationDiagnostics,
        benchmark: diagnosticsState.benchmark,
        spectralDiagnostics: diagnosticsState.spectralDiagnostics,
        forecastabilityDiagnostics: diagnosticsState.forecastabilityDiagnostics,
        datasetSummary: datasetState.summary,
        familyMeta,
    }), [config, analysis, diagnosticsState.automationDiagnostics, diagnosticsState.benchmark, diagnosticsState.spectralDiagnostics, diagnosticsState.forecastabilityDiagnostics, datasetState.summary, familyMeta]);

    useEffect(() => {
        if (!isCodeModalOpen) {
            return undefined;
        }

        const previousOverflow = document.body.style.overflow;
        const handleKeyDown = (event) => {
            if (event.key === "Escape") {
                if (isCodeModalOpen) setIsCodeModalOpen(false);
            }
        };

        document.body.style.overflow = "hidden";
        window.addEventListener("keydown", handleKeyDown);

        return () => {
            document.body.style.overflow = previousOverflow;
            window.removeEventListener("keydown", handleKeyDown);
        };
    }, [isCodeModalOpen]);

    useEffect(() => {
        if (typeof window !== "undefined") {
            try {
                localStorage.setItem("studioTheme", themeKey);
            } catch {
                // ignore storage failures
            }
        }
    }, [themeKey]);

    useEffect(() => {
        let cancelled = false;

        const loadDataset = async () => {
            setDatasetState((current) => ({
                ...current,
                status: "loading",
                errorMessage: "",
                sourceLabel: uploadedDataset?.name === config.data.filename
                    ? uploadedDataset.name
                    : config.data.filename,
            }));

            try {
                let csvText = "";
                let sourceLabel = config.data.filename;

                if (uploadedDataset && uploadedDataset.name === config.data.filename) {
                    csvText = uploadedDataset.text;
                    sourceLabel = uploadedDataset.name;
                } else {
                    const response = await fetch(config.data.filename, { cache: "no-store" });
                    if (!response.ok) {
                        throw new Error(
                            `Could not fetch ${config.data.filename}. Upload a CSV or use a browser-reachable relative path.`,
                        );
                    }
                    csvText = await response.text();
                }

                const parsedDataset = parseSeriesCsvText(csvText, {
                    targetColumn: config.data.target,
                    timeColumn: config.data.timestamp,
                });

                if (!cancelled) {
                    setDatasetState({
                        status: "ready",
                        series: parsedDataset.series,
                        errorMessage: "",
                        sourceLabel,
                        points: parsedDataset.series.length,
                        headers: parsedDataset.headers,
                        covariates: parsedDataset.covariates,
                        summary: parsedDataset.summary,
                    });
                }
            } catch (error) {
                if (!cancelled) {
                    setDatasetState({
                        status: "error",
                        series: [],
                        errorMessage: error instanceof Error ? error.message : "Dataset loading failed.",
                        sourceLabel: config.data.filename,
                        points: 0,
                        headers: [],
                        covariates: [],
                        summary: createEmptyDatasetSummary(),
                    });
                }
            }
        };

        loadDataset();

        return () => {
            cancelled = true;
        };
    }, [config.data.filename, config.data.target, config.data.timestamp, uploadedDataset]);

    useEffect(() => {
        if (datasetState.status !== "ready") {
            setDiagnosticsState({
                status: datasetState.status === "error" ? "error" : "loading",
                analysis: null,
                acfCurve: [],
                pacfCurve: [],
                decomposition: null,
                benchmark: null,
                exogenousScreening: [],
                grangerDiagnostics: null,
                changePoints: null,
                residualDiagnostics: null,
                stabilityDiagnostics: null,
                spectralDiagnostics: null,
                forecastabilityDiagnostics: null,
                patchingDiagnostics: null,
                emdDiagnostics: null,
                eemdDiagnostics: null,
                ewtDiagnostics: null,
                calendarDiagnostics: null,
                intermittencyDiagnostics: null,
                volatilityDiagnostics: null,
                automationDiagnostics: null,
                errorMessage: datasetState.status === "error" ? datasetState.errorMessage : "",
            });
            return undefined;
        }

        const worker = new Worker(new URL("./lib/diagnostics.worker.js", import.meta.url), {
            type: "module",
        });

        worker.onmessage = (event) => {
            const { data } = event;

            if (data?.type === "diagnostics-ready") {
                setDiagnosticsState({
                    status: "ready",
                    analysis: data.payload.analysis,
                    acfCurve: data.payload.acfCurve,
                    pacfCurve: data.payload.pacfCurve,
                    decomposition: data.payload.decomposition,
                    benchmark: data.payload.benchmark,
                    exogenousScreening: data.payload.exogenousScreening,
                    grangerDiagnostics: data.payload.grangerDiagnostics,
                    changePoints: data.payload.changePoints,
                    residualDiagnostics: data.payload.residualDiagnostics,
                    stabilityDiagnostics: data.payload.stabilityDiagnostics,
                    spectralDiagnostics: data.payload.spectralDiagnostics,
                    forecastabilityDiagnostics: data.payload.forecastabilityDiagnostics,
                    patchingDiagnostics: data.payload.patchingDiagnostics,
                    emdDiagnostics: data.payload.emdDiagnostics,
                    eemdDiagnostics: data.payload.eemdDiagnostics,
                    ewtDiagnostics: data.payload.ewtDiagnostics,
                    vmdDiagnostics: data.payload.vmdDiagnostics,
                    calendarDiagnostics: data.payload.calendarDiagnostics,
                    intermittencyDiagnostics: data.payload.intermittencyDiagnostics,
                    volatilityDiagnostics: data.payload.volatilityDiagnostics,
                    automationDiagnostics: data.payload.automationDiagnostics,
                    errorMessage: "",
                });
            }

            if (data?.type === "diagnostics-error") {
                setDiagnosticsState((current) => ({
                    ...current,
                    status: "error",
                    errorMessage: data.payload,
                }));
            }
        };

        worker.postMessage({
            type: "compute-diagnostics",
            series: datasetState.series,
            covariates: datasetState.covariates,
            horizon: config.prep.horizon,
            windowSize: config.prep.windowSize,
            changePointMethod,
            datasetSummary: datasetState.summary,
            emdOptions,
            eemdOptions,
            ewtOptions,
            vmdOptions,
        });

        return () => {
            worker.terminate();
        };
    }, [
        datasetState.status,
        datasetState.series,
        datasetState.covariates,
        datasetState.errorMessage,
        config.prep.horizon,
        config.prep.windowSize,
        changePointMethod,
        emdOptions,
        eemdOptions,
        ewtOptions,
        vmdOptions,
    ]);

    const blueprintSummary = diagnosticsState.status === "ready"
        ? `${familyMeta.label} with ${config.prep.scalingMethod} scaling, ${config.prep.windowSize}-step context, and ${config.train.epochs} training epochs. Stationarity tests classify the current signal as ${analysis.stationarityLabel}, the benchmark ladder is led by ${diagnosticsState.benchmark?.winner?.label ?? "the current baseline"}, automation readiness is ${diagnosticsState.automationDiagnostics?.readinessLabel ?? "pending"}, and PACF suggests lag ${analysis.dominantPacfLag} with a recommended context window near ${analysis.suggestedLagWindow}. The current configuration mirrors the stable foreBlocks pipeline: TimeSeriesHandler, create_dataloaders, ForecastingModel, and Trainer.`
        : datasetState.status === "error"
            ? `Dataset loading failed, so only the configuration shell is available. Fix the CSV source and the worker will recompute real lag diagnostics.`
            : `${familyMeta.label} with ${config.prep.scalingMethod} scaling, ${config.prep.windowSize}-step context, and ${config.train.epochs} training epochs. Background diagnostics are still computing ACF, PACF, and stationarity recommendations for the loaded series.`;

    const applyRecommendedBlueprint = () => {
        if (diagnosticsState.status !== "ready") {
            return;
        }
        setConfig((current) => {
            const next = applyBlueprint(current, analysis, diagnosticsState.patchingDiagnostics);
            const hasMissingValues = datasetState.summary.missingTargetCount > 0 || datasetState.summary.missingTimestampCount > 0;

            return {
                ...next,
                prep: {
                    ...next.prep,
                    applyImputation: hasMissingValues,
                    imputeMethod: hasMissingValues ? "linear" : next.prep.imputeMethod,
                },
            };
        });
    };

    const applyPreprocessingRecommendations = () => {
        if (diagnosticsState.status !== "ready") {
            return;
        }

        setConfig((current) => {
            const hasMissingValues = datasetState.summary.missingTargetCount > 0 || datasetState.summary.missingTimestampCount > 0;

            return {
                ...current,
                data: {
                    ...current.data,
                    freq: analysis.frequencyHint,
                },
                prep: {
                    ...current.prep,
                    differencing: analysis.needsDiff,
                    detrend: analysis.needsDetrend,
                    applyFilter: analysis.needsFilter,
                    generateTimeFeatures: analysis.useTimeFeatures,
                    scalingMethod: analysis.scalingMethod,
                    removeOutliers: analysis.outlierRate > 0.025,
                    outlierMethod: analysis.outlierRate > 0.025 ? "iqr" : current.prep.outlierMethod,
                    applyImputation: hasMissingValues,
                    imputeMethod: hasMissingValues ? "linear" : current.prep.imputeMethod,
                },
            };
        });
    };

    const applyLagWindow = (windowSize) => {
        setConfig((current) => ({
            ...current,
            prep: {
                ...current.prep,
                windowSize,
                horizon: Math.max(6, Math.min(24, Math.round(windowSize / 4))),
            },
        }));
    };

    const applyPatchGeometry = (patchingDiagnostics) => {
        if (!patchingDiagnostics?.available) {
            return;
        }

        setConfig((current) => ({
            ...current,
            model: {
                ...current.model,
                family: "transformer",
                patchEncoder: true,
                patchLen: patchingDiagnostics.recommendedPatchLen,
                patchStride: patchingDiagnostics.recommendedStride,
            },
        }));
        setActiveStep(2);
    };

    const openWorkflowTarget = (action) => {
        if (["intermittency", "volatility", "patching"].includes(action?.targetTab)) {
            setActivePrepLabSubgroup(action.targetTab);
            setActiveRightRailTab("prep");
            return;
        }

        if (action?.targetTab) {
            setActiveExploreTab(action.targetTab);
        }

        if (action?.targetSection === "backtesting") {
            setActiveRightRailTab("validation");
            return;
        }

        if (action?.targetSection === "regime") {
            setActiveRightRailTab("regime");
            setActiveRegimeSubgroup("stability");
            return;
        }

        if (action?.targetSection === "explore") {
            setActiveRightRailTab("explore");
            return;
        }

        if (action?.targetSection === "automation") {
            setActiveRightRailTab("automation");
            return;
        }

        if (action?.targetSection === "stationarity") {
            setActiveRightRailTab("regime");
            setActiveRegimeSubgroup("signal");
            return;
        }

        setActiveRightRailTab("overview");
    };

    const runAutomationAction = (action) => {
        if (!action) {
            return;
        }

        if (action.key === "apply_blueprint") {
            applyRecommendedBlueprint();
            return;
        }

        if (action.key === "auto_prep") {
            applyPreprocessingRecommendations();
            return;
        }

        if ((action.key === "apply_window" || action.key === "apply_spectral_window") && action.window) {
            applyLagWindow(action.window);
            openWorkflowTarget(action);
            return;
        }

        openWorkflowTarget(action);
    };

    const resetConfig = () => {
        setConfig(INITIAL_CONFIG);
        setUploadedDataset(null);
        setChangePointMethod("segmentation");
        setOutlierPreviewMethod(INITIAL_CONFIG.prep.outlierMethod);
        setFilterPreviewSettings({
            applyFilter: false,
            filterMethod: "none",
            filterWindow: 9,
            filterPolyorder: 2,
            lowessFraction: 0.12,
            detrend: false,
            detrenderMethod: "none",
            detrenderOrder: 2,
        });
    };

    const handleFileLoad = async (file) => {
        const text = await file.text();
        let detectedTarget = config.data.target;
        let detectedTimestamp = config.data.timestamp;

        try {
            const inspection = inspectCsvText(text);
            detectedTarget = inspection.suggestions.targetColumn || detectedTarget;
            detectedTimestamp = inspection.suggestions.timeColumn || detectedTimestamp;
        } catch {
            // Let the main dataset load path surface parse errors.
        }

        setUploadedDataset({ name: file.name, text });
        setConfig((current) => ({
            ...current,
            data: {
                ...current.data,
                filename: file.name,
                target: detectedTarget,
                timestamp: detectedTimestamp,
            },
        }));
    };

    let stepPanel = (
        <DatasetStep
            config={config}
            setConfig={setConfig}
            datasetState={datasetState}
            onFileLoad={handleFileLoad}
        />
    );

    if (activeStep === 1) {
        stepPanel = <PreprocessStep config={config} setConfig={setConfig} />;
    }
    if (activeStep === 2) {
        stepPanel = (
            <ArchitectureStep
                config={config}
                setConfig={setConfig}
                familyMeta={familyMeta}
                analysis={analysis}
                patchingDiagnostics={diagnosticsState.patchingDiagnostics}
            />
        );
    }
    if (activeStep === 3) {
        stepPanel = <TrainingStep config={config} setConfig={setConfig} />;
    }
    if (activeStep === 4) {
        stepPanel = (
            <BlueprintStep
                analysis={analysis}
                config={config}
                familyMeta={familyMeta}
                onApplyBlueprint={applyRecommendedBlueprint}
                onReset={resetConfig}
            />
        );
    }

    return (
        <div className="studio-root" style={themeStyle} data-color-scheme={isDarkMode ? "dark" : "light"}>
            <header className="topbar">
                <div className="brand">
                    <div className="brand-mark" aria-hidden="true">
                        <StudioLogo />
                    </div>
                    <div className="brand-copy">
                        <h1>foreBlocks Studio</h1>
                        <p>
                            Scientific workspace for shaping preprocessing, selecting stable
                            forecasting backbones, and generating runnable foreBlocks pipelines.
                        </p>
                    </div>
                </div>
                <div className="topbar-actions">
                    <span className={`status-badge${datasetState.status === "ready" ? " status-badge-ready" : datasetState.status === "error" ? " status-badge-error" : ""}`}>
                        <span className={`status-dot status-dot-${datasetState.status === "ready" ? "ready" : datasetState.status === "error" ? "error" : "loading"}`} />
                        Dataset
                    </span>
                    <span className={`status-badge${diagnosticsState.status === "ready" ? " status-badge-ready" : diagnosticsState.status === "error" ? " status-badge-error" : ""}`}>
                        <span className={`status-dot status-dot-${diagnosticsState.status === "ready" ? "ready" : diagnosticsState.status === "error" ? "error" : "loading"}`} />
                        Diagnostics
                    </span>
                    <select
                        className="theme-select"
                        value={themeKey}
                        style={{ color: "white" }}
                        onChange={(event) => setThemeKey(event.target.value)}
                        aria-label="Choose Studio theme"
                    >
                        <option value="dark">Night</option>
                        <option value="glass">Glass</option>
                        <option value="paper">Paper</option>
                    </select>
                    <button
                        className="theme-toggle"
                        type="button"
                        title={isDarkMode ? "Switch to light theme" : "Switch to dark theme"}
                        aria-label={isDarkMode ? "Switch to light theme" : "Switch to dark theme"}
                        onClick={() => setThemeKey((current) => (current === "dark" ? "paper" : "dark"))}
                    >
                        {isDarkMode ? "☀" : "◑"}
                    </button>
                    <button
                        className="ghost-button"
                        type="button"
                        onClick={() => setIsCodeModalOpen(true)}
                    >
                        Generate Code
                    </button>
                    <button
                        className="solid-button"
                        type="button"
                        onClick={applyRecommendedBlueprint}
                        disabled={diagnosticsState.status !== "ready"}
                    >
                        Apply Blueprint
                    </button>
                </div>
            </header>

            <main className="workspace">
                <section className="left-rail">
                    <section className="panel">
                        <div className="config-step-row">
                            {STEPS.map((step, index) => (
                                <button
                                    key={step}
                                    className={`config-step-chip ${activeStep === index ? "config-step-chip-active" : ""
                                        }`}
                                    type="button"
                                    onClick={() => setActiveStep(index)}
                                >
                                    {step}
                                </button>
                            ))}
                        </div>

                        {stepPanel}
                    </section>
                </section>

                <section className="right-rail">
                    <div className="right-panel-tab-row">
                        {[
                            { key: "explore", label: "Exploration" },
                            { key: "overview", label: "Overview" },
                            { key: "prep", label: "Preparation Lab" },
                            { key: "emd-like", label: "EMD-like" },
                            { key: "regime", label: "Regime & Stability" },
                            { key: "automation", label: "Automation Center" },
                            { key: "validation", label: "Validation" },
                        ].map((tab) => (
                            <button
                                key={tab.key}
                                type="button"
                                className={`right-panel-tab ${activeRightRailTab === tab.key ? "right-panel-tab-active" : ""}`}
                                onClick={() => setActiveRightRailTab(tab.key)}
                            >
                                {tab.label}
                            </button>
                        ))}
                    </div>

                    <div className="overview-stack">
                        {activeRightRailTab === "overview" ? (
                            <>
                                <OverviewPanel
                                    analysis={analysis}
                                    config={config}
                                    familyMeta={familyMeta}
                                    blueprintSummary={blueprintSummary}
                                    datasetSummary={datasetState.summary}
                                    automationDiagnostics={diagnosticsState.automationDiagnostics}
                                    benchmark={diagnosticsState.benchmark}
                                    spectralDiagnostics={diagnosticsState.spectralDiagnostics}
                                    forecastabilityDiagnostics={diagnosticsState.forecastabilityDiagnostics}
                                    experimentNarrative={experimentNarrative}
                                    onRunAction={runAutomationAction}
                                />
                                <DiagnosticsPanel
                                    status={diagnosticsState.status}
                                    errorMessage={diagnosticsState.errorMessage}
                                    analysis={analysis}
                                    spectralDiagnostics={diagnosticsState.spectralDiagnostics}
                                    forecastabilityDiagnostics={diagnosticsState.forecastabilityDiagnostics}
                                    automationDiagnostics={diagnosticsState.automationDiagnostics}
                                    residualDiagnostics={diagnosticsState.residualDiagnostics}
                                    benchmark={diagnosticsState.benchmark}
                                    datasetSummary={datasetState.summary}
                                />
                            </>
                        ) : null}

                        {activeRightRailTab === "explore" ? (
                            <ExplorePanel
                                data={seriesData}
                                status={datasetState.status}
                                datasetSummary={datasetState.summary}
                                headers={datasetState.headers}
                                covariates={datasetState.covariates}
                                targetColumn={config.data.target}
                                timestampColumn={config.data.timestamp}
                                analysis={analysis}
                                acfCurve={diagnosticsState.acfCurve}
                                pacfCurve={diagnosticsState.pacfCurve}
                                decomposition={diagnosticsState.decomposition}
                                exogenousScreening={diagnosticsState.exogenousScreening}
                                grangerDiagnostics={diagnosticsState.grangerDiagnostics}
                                spectralDiagnostics={diagnosticsState.spectralDiagnostics}
                                calendarDiagnostics={diagnosticsState.calendarDiagnostics}
                                onApplyLagWindow={applyLagWindow}
                                activeTabOverride={activeExploreTab}
                                onActiveTabChange={setActiveExploreTab}
                                errorMessage={datasetState.errorMessage}
                            />
                        ) : null}

                        {activeRightRailTab === "prep" ? (
                            <>
                                <section className="panel automation-shell">
                                    <div className="panel-head automation-shell-head">
                                        <div>
                                            <div className="panel-kicker">Preparation workspace</div>
                                            <h3 className="panel-title">Preparation Lab</h3>
                                        </div>
                                    </div>

                                    <div className="prep-lab-subgroup-row">
                                        {[
                                            { key: "outliers", label: "Outliers" },
                                            { key: "filtering", label: "Filtering" },
                                            { key: "features", label: "Feature Lab" },
                                            { key: "intermittency", label: "Intermittency" },
                                            { key: "volatility", label: "Volatility" },
                                            { key: "patching", label: "Patching" },
                                        ].map((group) => (
                                            <button
                                                key={group.key}
                                                type="button"
                                                className={`automation-subgroup-tab ${activePrepLabSubgroup === group.key ? "automation-subgroup-tab-active" : ""}`}
                                                onClick={() => setActivePrepLabSubgroup(group.key)}
                                            >
                                                {group.label}
                                            </button>
                                        ))}
                                    </div>

                                    <p className="lead-copy automation-shell-copy">
                                        This workspace collects diagnostics that directly change preprocessing, tokenization, and target transformation decisions before training.
                                    </p>
                                </section>

                                {activePrepLabSubgroup === "features" ? (
                                    <FeatureLabPanel
                                        status={datasetState.status}
                                        errorMessage={datasetState.errorMessage}
                                        data={regimeSeriesData}
                                        covariates={datasetState.covariates}
                                        targetColumn={config.data.target}
                                        analysis={analysis}
                                        decomposition={diagnosticsState.decomposition}
                                    />
                                ) : null}

                                {activePrepLabSubgroup === "outliers" ? (
                                    <OutlierPanel
                                        status={datasetState.status}
                                        errorMessage={datasetState.errorMessage}
                                        data={regimeSeriesData}
                                        targetColumn={config.data.target}
                                        previewMethod={outlierPreviewMethod}
                                        onPreviewMethodChange={setOutlierPreviewMethod}
                                        pipelineMethod={config.prep.outlierMethod}
                                        pipelineEnabled={config.prep.removeOutliers}
                                        onSyncFromPreprocessing={() => setOutlierPreviewMethod(config.prep.outlierMethod)}
                                    />
                                ) : null}

                                {activePrepLabSubgroup === "filtering" ? (
                                    <FilterPanel
                                        status={datasetState.status}
                                        errorMessage={datasetState.errorMessage}
                                        data={seriesData}
                                        settings={filterPreviewSettings}
                                        onSettingsChange={setFilterPreviewSettings}
                                    />
                                ) : null}

                                {activePrepLabSubgroup === "intermittency" ? (
                                    <IntermittencyPanel
                                        status={diagnosticsState.status}
                                        errorMessage={diagnosticsState.errorMessage}
                                        intermittencyDiagnostics={diagnosticsState.intermittencyDiagnostics}
                                    />
                                ) : null}

                                {activePrepLabSubgroup === "volatility" ? (
                                    <VolatilityPanel
                                        status={diagnosticsState.status}
                                        errorMessage={diagnosticsState.errorMessage}
                                        volatilityDiagnostics={diagnosticsState.volatilityDiagnostics}
                                    />
                                ) : null}

                                {activePrepLabSubgroup === "patching" ? (
                                    <PatchingPanel
                                        status={diagnosticsState.status}
                                        errorMessage={diagnosticsState.errorMessage}
                                        patchingDiagnostics={diagnosticsState.patchingDiagnostics}
                                        activeWindow={config.prep.windowSize}
                                        onApplyPatchGeometry={applyPatchGeometry}
                                    />
                                ) : null}
                            </>
                        ) : null}

                        {activeRightRailTab === "emd-like" ? (
                            <>
                                <section className="panel automation-shell">
                                    <div className="panel-head automation-shell-head">
                                        <div>
                                            <div className="panel-kicker">EMD-like diagnostics</div>
                                            <h3 className="panel-title">EMD-like</h3>
                                        </div>
                                    </div>

                                    <div className="prep-lab-subgroup-row">
                                        {[
                                            { key: "emd", label: "EMD" },
                                            { key: "eemd", label: "EEMD" },
                                            { key: "ewt", label: "EWT" },
                                            { key: "vmd", label: "VMD" },
                                        ].map((group) => (
                                            <button
                                                key={group.key}
                                                type="button"
                                                className={`automation-subgroup-tab ${activeEmdLikeSubgroup === group.key ? "automation-subgroup-tab-active" : ""}`}
                                                onClick={() => setActiveEmdLikeSubgroup(group.key)}
                                            >
                                                {group.label}
                                            </button>
                                        ))}
                                    </div>

                                    <p className="lead-copy automation-shell-copy">
                                        These diagnostics expose intrinsic decomposition methods for the current series. Use them to compare adaptive, ensemble, wavelet, and variational mode decompositions in one place.
                                    </p>
                                </section>

                                {activeEmdLikeSubgroup === "emd" ? (
                                    <EmdPanel
                                        status={diagnosticsState.status}
                                        errorMessage={diagnosticsState.errorMessage}
                                        emdDiagnostics={diagnosticsState.emdDiagnostics}
                                        emdOptions={emdOptions}
                                        onOptionsChange={setEmdOptions}
                                        seriesData={regimeSeriesData}
                                    />
                                ) : null}

                                {activeEmdLikeSubgroup === "eemd" ? (
                                    <EemdPanel
                                        status={diagnosticsState.status}
                                        errorMessage={diagnosticsState.errorMessage}
                                        eemdDiagnostics={diagnosticsState.eemdDiagnostics}
                                        eemdOptions={eemdOptions}
                                        onOptionsChange={setEemdOptions}
                                        seriesData={regimeSeriesData}
                                    />
                                ) : null}

                                {activeEmdLikeSubgroup === "ewt" ? (
                                    <EwtPanel
                                        status={diagnosticsState.status}
                                        errorMessage={diagnosticsState.errorMessage}
                                        ewtDiagnostics={diagnosticsState.ewtDiagnostics}
                                        ewtOptions={ewtOptions}
                                        onOptionsChange={setEwtOptions}
                                        seriesData={regimeSeriesData}
                                    />
                                ) : null}

                                {activeEmdLikeSubgroup === "vmd" ? (
                                    <VmdPanel
                                        status={diagnosticsState.status}
                                        errorMessage={diagnosticsState.errorMessage}
                                        vmdDiagnostics={diagnosticsState.vmdDiagnostics}
                                        vmdOptions={vmdOptions}
                                        onOptionsChange={setVmdOptions}
                                        seriesData={regimeSeriesData}
                                    />
                                ) : null}
                            </>
                        ) : null}

                        {activeRightRailTab === "automation" ? (
                            <>
                                <section className="panel automation-shell">
                                    <div className="panel-head automation-shell-head">
                                        <div>
                                            <div className="panel-kicker">Automation workflow</div>
                                            <h3 className="panel-title">Automation Center</h3>
                                        </div>
                                    </div>

                                    <p className="lead-copy automation-shell-copy">
                                        This workspace is now action-focused: use it to execute the queued recommendations that jump into the relevant diagnostic workspaces.
                                    </p>
                                </section>

                                <AutomationPanel
                                    status={diagnosticsState.status}
                                    errorMessage={diagnosticsState.errorMessage}
                                    automationDiagnostics={diagnosticsState.automationDiagnostics}
                                    analysis={analysis}
                                    onRunAction={runAutomationAction}
                                />
                            </>
                        ) : null}

                        {activeRightRailTab === "validation" ? (
                            <BacktestingPanel
                                benchmark={diagnosticsState.benchmark}
                                residualDiagnostics={diagnosticsState.residualDiagnostics}
                                status={diagnosticsState.status}
                                errorMessage={diagnosticsState.errorMessage}
                            />
                        ) : null}

                        {activeRightRailTab === "regime" ? (
                            <>
                                <section className="panel automation-shell">
                                    <div className="panel-head automation-shell-head">
                                        <div>
                                            <div className="panel-kicker">Diagnostic detail</div>
                                            <h3 className="panel-title">Regime & Stability</h3>
                                        </div>
                                    </div>

                                    <div className="automation-subgroup-row">
                                        {[
                                            { key: "signal", label: "Signal Checks" },
                                            { key: "stability", label: "Structural Stability" },
                                        ].map((group) => (
                                            <button
                                                key={group.key}
                                                type="button"
                                                className={`automation-subgroup-tab ${activeRegimeSubgroup === group.key ? "automation-subgroup-tab-active" : ""}`}
                                                onClick={() => setActiveRegimeSubgroup(group.key)}
                                            >
                                                {group.label}
                                            </button>
                                        ))}
                                    </div>

                                    <p className="lead-copy automation-shell-copy">
                                        Signal stationarity and structural stability are grouped here because both are diagnostic evidence about whether the observed process is stable enough to support the current modeling assumptions.
                                    </p>
                                </section>

                                {activeRegimeSubgroup === "signal" ? (
                                    <StationarityPanel
                                        analysis={analysis}
                                        status={diagnosticsState.status}
                                        errorMessage={diagnosticsState.errorMessage}
                                        onApplyPreprocessing={applyPreprocessingRecommendations}
                                        hasMissingValues={datasetState.summary.missingTargetCount > 0 || datasetState.summary.missingTimestampCount > 0}
                                    />
                                ) : null}

                                {activeRegimeSubgroup === "stability" ? (
                                    <RegimePanel
                                        status={diagnosticsState.status}
                                        errorMessage={diagnosticsState.errorMessage}
                                        changePoints={diagnosticsState.changePoints}
                                        stabilityDiagnostics={diagnosticsState.stabilityDiagnostics}
                                        seriesData={regimeSeriesData}
                                        changePointMethod={changePointMethod}
                                        onChangePointMethod={setChangePointMethod}
                                    />
                                ) : null}
                            </>
                        ) : null}
                    </div>
                </section>
            </main>

            {isCodeModalOpen ? (
                <div
                    className="modal-overlay"
                    role="presentation"
                    onClick={() => setIsCodeModalOpen(false)}
                >
                    <div
                        className="modal-dialog modal-code-dialog"
                        role="dialog"
                        aria-modal="true"
                        aria-label="Generated foreBlocks Script"
                        onClick={(event) => event.stopPropagation()}
                    >
                        <button
                            className="modal-close"
                            type="button"
                            aria-label="Close generated code"
                            onClick={() => setIsCodeModalOpen(false)}
                        >
                            Close
                        </button>
                        <NarrativePanel narrative={experimentNarrative} />
                        <CodePanel code={code} notebook={notebook} />
                    </div>
                </div>
            ) : null}
        </div>
    );
}
