import { create } from "zustand";
import { persist } from "zustand/middleware";
import { INITIAL_CONFIG } from "./studio-data.js";

export const DEFAULT_FILTER_PREVIEW_SETTINGS = {
    applyFilter: false,
    filterMethod: "none",
    filterWindow: 9,
    filterPolyorder: 2,
    lowessFraction: 0.12,
    detrend: false,
    detrenderMethod: "none",
    detrenderOrder: 2,
};

export function createEmptyDatasetSummary() {
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

export function createDefaultDatasetState() {
    return {
        status: "loading",
        series: [],
        errorMessage: "",
        sourceLabel: INITIAL_CONFIG.data.filename,
        points: 0,
        headers: [],
        covariates: [],
        summary: createEmptyDatasetSummary(),
    };
}

export function createDefaultDiagnosticsState() {
    return {
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
    };
}

const getInitialThemeKey = () => {
    if (typeof window !== "undefined") {
        return localStorage.getItem("studioTheme") || "paper";
    }
    return "paper";
};

const createUiSlice = (set) => ({
    activeStep: 0,
    isCodeModalOpen: false,
    activeExploreTab: "series",
    activeRightRailTab: "explore",
    activePrepLabSubgroup: "outliers",
    activeEmdLikeSubgroup: "emd",
    activeRegimeSubgroup: "signal",
    themeKey: getInitialThemeKey(),
    setActiveStep: (value) => set({ activeStep: value }),
    setIsCodeModalOpen: (value) => set({ isCodeModalOpen: value }),
    setActiveExploreTab: (value) => set({ activeExploreTab: value }),
    setActiveRightRailTab: (value) => set({ activeRightRailTab: value }),
    setActivePrepLabSubgroup: (value) => set({ activePrepLabSubgroup: value }),
    setActiveEmdLikeSubgroup: (value) => set({ activeEmdLikeSubgroup: value }),
    setActiveRegimeSubgroup: (value) => set({ activeRegimeSubgroup: value }),
    setThemeKey: (value) => set({ themeKey: value }),
    resetUi: () => set({
        activeStep: 0,
        activeExploreTab: "series",
        activeRightRailTab: "explore",
        activePrepLabSubgroup: "outliers",
        activeEmdLikeSubgroup: "emd",
        activeRegimeSubgroup: "signal",
    }),
});

const createAnalysisSlice = (set) => ({
    config: INITIAL_CONFIG,
    configHistory: [INITIAL_CONFIG],
    historyIndex: 0,
    canUndo: false,
    canRedo: false,
    setConfig: (value, options = { recordHistory: true }) => set((state) => {
        const nextConfig = typeof value === "function" ? value(state.config) : value;
        const currentSerialized = JSON.stringify(state.config);
        const nextSerialized = JSON.stringify(nextConfig);

        if (currentSerialized === nextSerialized) {
            return {};
        }

        if (!options.recordHistory) {
            return {
                config: nextConfig,
                canUndo: state.historyIndex > 0,
                canRedo: state.historyIndex < state.configHistory.length - 1,
            };
        }

        const nextHistory = [...state.configHistory.slice(0, state.historyIndex + 1), nextConfig];
        const nextIndex = nextHistory.length - 1;

        return {
            config: nextConfig,
            configHistory: nextHistory,
            historyIndex: nextIndex,
            canUndo: nextIndex > 0,
            canRedo: false,
        };
    }),
    undoConfig: () => set((state) => {
        if (state.historyIndex <= 0) {
            return {};
        }
        const nextIndex = state.historyIndex - 1;
        return {
            config: state.configHistory[nextIndex],
            historyIndex: nextIndex,
            canUndo: nextIndex > 0,
            canRedo: true,
        };
    }),
    redoConfig: () => set((state) => {
        if (state.historyIndex >= state.configHistory.length - 1) {
            return {};
        }
        const nextIndex = state.historyIndex + 1;
        return {
            config: state.configHistory[nextIndex],
            historyIndex: nextIndex,
            canUndo: true,
            canRedo: nextIndex < state.configHistory.length - 1,
        };
    }),
    resetConfig: () => set({
        config: INITIAL_CONFIG,
        configHistory: [INITIAL_CONFIG],
        historyIndex: 0,
        canUndo: false,
        canRedo: false,
        uploadedDataset: null,
        changePointMethod: "segmentation",
        outlierPreviewMethod: INITIAL_CONFIG.prep.outlierMethod,
        filterPreviewSettings: DEFAULT_FILTER_PREVIEW_SETTINGS,
    }),
});

const createDatasetSlice = (set) => ({
    uploadedDataset: null,
    datasetState: createDefaultDatasetState(),
    setUploadedDataset: (value) => set({ uploadedDataset: value }),
    setDatasetState: (value) => set((state) => ({
        datasetState: typeof value === "function" ? value(state.datasetState) : value,
    })),
    resetDataset: () => set({
        uploadedDataset: null,
        datasetState: createDefaultDatasetState(),
    }),
});

const createDiagnosticsSlice = (set) => ({
    changePointMethod: "segmentation",
    outlierPreviewMethod: INITIAL_CONFIG.prep.outlierMethod,
    filterPreviewSettings: DEFAULT_FILTER_PREVIEW_SETTINGS,
    emdOptions: {
        maxImfs: 6,
        maxSifts: 80,
        siftThreshold: 0.03,
    },
    eemdOptions: {
        maxImfs: 6,
        ensembleSize: 8,
        noiseStdRatio: 0.15,
        siftThreshold: 0.08,
        maxSifts: 80,
    },
    ewtOptions: {
        maxBands: 5,
        smoothingWindow: 7,
        detectThreshold: 0.05,
        gamma: 0.1,
    },
    vmdOptions: {
        modeCount: 4,
        alpha: 2000,
        tolerance: 1e-7,
        maxIterations: 500,
    },
    diagnosticsState: createDefaultDiagnosticsState(),
    setChangePointMethod: (value) => set({ changePointMethod: value }),
    setOutlierPreviewMethod: (value) => set({ outlierPreviewMethod: value }),
    setFilterPreviewSettings: (value) => set((state) => ({
        filterPreviewSettings: typeof value === "function"
            ? value(state.filterPreviewSettings)
            : value,
    })),
    setEmdOptions: (value) => set((state) => ({
        emdOptions: typeof value === "function" ? value(state.emdOptions) : value,
    })),
    setEemdOptions: (value) => set((state) => ({
        eemdOptions: typeof value === "function" ? value(state.eemdOptions) : value,
    })),
    setEwtOptions: (value) => set((state) => ({
        ewtOptions: typeof value === "function" ? value(state.ewtOptions) : value,
    })),
    setVmdOptions: (value) => set((state) => ({
        vmdOptions: typeof value === "function" ? value(state.vmdOptions) : value,
    })),
    setDiagnosticsState: (value) => set((state) => ({
        diagnosticsState: typeof value === "function"
            ? value(state.diagnosticsState)
            : value,
    })),
    resetDiagnostics: () => set({
        changePointMethod: "segmentation",
        outlierPreviewMethod: INITIAL_CONFIG.prep.outlierMethod,
        filterPreviewSettings: DEFAULT_FILTER_PREVIEW_SETTINGS,
        emdOptions: {
            maxImfs: 6,
            maxSifts: 80,
            siftThreshold: 0.03,
        },
        eemdOptions: {
            maxImfs: 6,
            ensembleSize: 8,
            noiseStdRatio: 0.15,
            siftThreshold: 0.08,
            maxSifts: 80,
        },
        ewtOptions: {
            maxBands: 5,
            smoothingWindow: 7,
            detectThreshold: 0.05,
            gamma: 0.1,
        },
        vmdOptions: {
            modeCount: 4,
            alpha: 2000,
            tolerance: 1e-7,
            maxIterations: 500,
        },
        diagnosticsState: createDefaultDiagnosticsState(),
    }),
});

export const selectUi = (state) => ({
    activeStep: state.activeStep,
    setActiveStep: state.setActiveStep,
    isCodeModalOpen: state.isCodeModalOpen,
    setIsCodeModalOpen: state.setIsCodeModalOpen,
    activeExploreTab: state.activeExploreTab,
    setActiveExploreTab: state.setActiveExploreTab,
    activeRightRailTab: state.activeRightRailTab,
    setActiveRightRailTab: state.setActiveRightRailTab,
    activePrepLabSubgroup: state.activePrepLabSubgroup,
    setActivePrepLabSubgroup: state.setActivePrepLabSubgroup,
    activeEmdLikeSubgroup: state.activeEmdLikeSubgroup,
    setActiveEmdLikeSubgroup: state.setActiveEmdLikeSubgroup,
    activeRegimeSubgroup: state.activeRegimeSubgroup,
    setActiveRegimeSubgroup: state.setActiveRegimeSubgroup,
    themeKey: state.themeKey,
    setThemeKey: state.setThemeKey,
    resetUi: state.resetUi,
});

export const selectDataset = (state) => ({
    uploadedDataset: state.uploadedDataset,
    setUploadedDataset: state.setUploadedDataset,
    datasetState: state.datasetState,
    setDatasetState: state.setDatasetState,
    resetDataset: state.resetDataset,
});

export const selectDiagnostics = (state) => ({
    changePointMethod: state.changePointMethod,
    setChangePointMethod: state.setChangePointMethod,
    outlierPreviewMethod: state.outlierPreviewMethod,
    setOutlierPreviewMethod: state.setOutlierPreviewMethod,
    filterPreviewSettings: state.filterPreviewSettings,
    setFilterPreviewSettings: state.setFilterPreviewSettings,
    emdOptions: state.emdOptions,
    setEmdOptions: state.setEmdOptions,
    eemdOptions: state.eemdOptions,
    setEemdOptions: state.setEemdOptions,
    ewtOptions: state.ewtOptions,
    setEwtOptions: state.setEwtOptions,
    vmdOptions: state.vmdOptions,
    setVmdOptions: state.setVmdOptions,
    diagnosticsState: state.diagnosticsState,
    setDiagnosticsState: state.setDiagnosticsState,
    resetDiagnostics: state.resetDiagnostics,
});

export const selectAnalysis = (state) => ({
    config: state.config,
    setConfig: state.setConfig,
    resetConfig: state.resetConfig,
});

export const useStudioStore = create(
    persist(
        (set) => ({
            ...createUiSlice(set),
            ...createAnalysisSlice(set),
            ...createDatasetSlice(set),
            ...createDiagnosticsSlice(set),
            loadWorkspace: (workspace) => set((state) => ({
                config: workspace.config ?? state.config,
                configHistory: [workspace.config ?? state.config],
                historyIndex: 0,
                canUndo: false,
                canRedo: false,
                uploadedDataset: workspace.uploadedDataset ?? state.uploadedDataset,
                datasetState: workspace.datasetState ?? state.datasetState,
                diagnosticsState: workspace.diagnosticsState ?? state.diagnosticsState,
                themeKey: workspace.themeKey ?? state.themeKey,
                activeStep: workspace.activeStep ?? state.activeStep,
                activeExploreTab: workspace.activeExploreTab ?? state.activeExploreTab,
                activeRightRailTab: workspace.activeRightRailTab ?? state.activeRightRailTab,
                activePrepLabSubgroup: workspace.activePrepLabSubgroup ?? state.activePrepLabSubgroup,
                activeEmdLikeSubgroup: workspace.activeEmdLikeSubgroup ?? state.activeEmdLikeSubgroup,
                activeRegimeSubgroup: workspace.activeRegimeSubgroup ?? state.activeRegimeSubgroup,
            })),
        }),
        {
            name: "foreblocks-studio",
            partialize: (state) => ({
                themeKey: state.themeKey,
                uploadedDataset: state.uploadedDataset,
                datasetState: state.datasetState,
                activeStep: state.activeStep,
                activeExploreTab: state.activeExploreTab,
                activeRightRailTab: state.activeRightRailTab,
                activePrepLabSubgroup: state.activePrepLabSubgroup,
                activeEmdLikeSubgroup: state.activeEmdLikeSubgroup,
                activeRegimeSubgroup: state.activeRegimeSubgroup,
            }),
            getStorage: () =>
                typeof window !== "undefined" ? localStorage : undefined,
        },
    ),
);
