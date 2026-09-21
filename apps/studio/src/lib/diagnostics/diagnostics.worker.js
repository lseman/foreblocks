import { computeSeriesDiagnostics } from "./diagnostics-core.js";

self.onmessage = (event) => {
  const { data } = event;

  if (!data || data.type !== "compute-diagnostics") {
    return;
  }

  try {
    const result = computeSeriesDiagnostics({
      series: data.series,
      covariates: data.covariates,
      horizon: data.horizon,
      windowSize: data.windowSize,
      changePointMethod: data.changePointMethod,
      datasetSummary: data.datasetSummary,
      emdOptions: data.emdOptions,
      eemdOptions: data.eemdOptions,
      ewtOptions: data.ewtOptions,
      vmdOptions: data.vmdOptions,
    });
    self.postMessage({ type: "diagnostics-ready", payload: result });
  } catch (error) {
    self.postMessage({
      type: "diagnostics-error",
      payload: error instanceof Error ? error.message : "Diagnostics worker failed.",
    });
  }
};
