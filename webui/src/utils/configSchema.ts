import type {
  ConfigFieldSchema,
  ConfigFieldType,
  NodeTypeDef,
} from "../types/types";

export interface ResolvedConfigField {
  key: string;
  label: string;
  type: ConfigFieldType;
  section: string;
  description?: string;
  options?: Array<string | number | boolean>;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  advanced: boolean;
  required: boolean;
}

const KEY_OPTIONS: Record<string, string[]> = {
  forecasting_strategy: ["seq2seq", "direct", "recursive", "multioutput"],
  model_type: [
    "informer-like",
    "transformer",
    "lstm",
    "gru",
    "tcn",
    "timesnet",
    "nbeats",
    "autoformer",
    "fedformer",
  ],
  optimizer: ["adam", "adamw", "sgd", "rmsprop"],
  loss: ["mse", "mae", "huber", "smooth_l1"],
  scheduler_type: ["none", "step", "cosine", "plateau"],
  activation: ["relu", "gelu", "silu", "tanh", "sigmoid"],
  freq: ["s", "min", "h", "d", "w", "m", "q", "y"],
};

const INTEGER_KEY_HINT = /(layers|heads|epoch|batch|patience|len|size|steps|window|horizon|stride|patch|topk|dim|channels|workers)$/i;
const NUMBER_KEY_HINT = /(dropout|rate|lr|learning|weight|alpha|beta|gamma|delta|eps|epsilon)$/i;
const PATH_KEY_HINT = /(path|file|dir|folder|checkpoint|ckpt|save)$/i;

const SECTION_RULES: Array<{ re: RegExp; section: string }> = [
  { re: /(d_|n_|layers|dropout|heads|embed|stride|patch|channels|kernel)/i, section: "Model Architecture" },
  { re: /(seq|len|pred|label|target|horizon|window)/i, section: "Sequence Settings" },
  { re: /(rate|batch|epoch|optimizer|loss|amp|patience|scheduler|weight_decay|lr)/i, section: "Training Params" },
  { re: /(feature|scale|freq|data|file|path|column|csv|time)/i, section: "Data Settings" },
];

const DEFAULT_SECTION_ORDER = [
  "Model Architecture",
  "Sequence Settings",
  "Training Params",
  "Data Settings",
  "General",
  "Advanced",
];

export const SECTION_ORDER = DEFAULT_SECTION_ORDER;

function humanizeLabel(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^./, (m) => m.toUpperCase());
}

function inferSection(key: string): string {
  const k = key.toLowerCase();
  const found = SECTION_RULES.find((r) => r.re.test(k));
  return found?.section || "General";
}

function inferTypeFromValue(key: string, value: any): ConfigFieldType {
  if (typeof value === "boolean") return "boolean";
  if (typeof value === "number") return Number.isInteger(value) ? "integer" : "number";
  if (Array.isArray(value)) return "array";
  if (value && typeof value === "object") return "json";
  if (typeof value === "string") {
    if (KEY_OPTIONS[key.toLowerCase()]) return "enum";
    return "string";
  }
  if (value == null) {
    const lk = key.toLowerCase();
    if (KEY_OPTIONS[lk]) return "enum";
    if (INTEGER_KEY_HINT.test(lk)) return "integer";
    if (NUMBER_KEY_HINT.test(lk)) return "number";
    return "string";
  }
  return "string";
}

function normalizeOptions(
  key: string,
  currentValue: any,
  schema?: ConfigFieldSchema,
): Array<string | number | boolean> | undefined {
  const keyOpts = KEY_OPTIONS[key.toLowerCase()] || [];
  const raw = schema?.options || schema?.enum || keyOpts;
  if (!raw || raw.length === 0) return undefined;
  const options = [...raw];
  if (
    currentValue != null &&
    (typeof currentValue === "string" ||
      typeof currentValue === "number" ||
      typeof currentValue === "boolean") &&
    !options.some((x) => x === currentValue)
  ) {
    options.unshift(currentValue);
  }
  return options;
}

function resolveFieldType(
  key: string,
  value: any,
  schema?: ConfigFieldSchema,
): ConfigFieldType {
  const schemaType = schema?.type || schema?.kind;
  const options = normalizeOptions(key, value, schema);
  if (schemaType) return schemaType;
  const inferred = inferTypeFromValue(key, value);
  if (options && options.length > 0 && inferred === "string") return "enum";
  return inferred;
}

function resolveSingleField(
  key: string,
  value: any,
  schema?: ConfigFieldSchema,
): ResolvedConfigField {
  const type = resolveFieldType(key, value, schema);
  const options = normalizeOptions(key, value, schema);
  const section = schema?.section || inferSection(key);
  const placeholder =
    schema?.placeholder || (PATH_KEY_HINT.test(key.toLowerCase()) ? "Enter path..." : undefined);

  return {
    key,
    label: schema?.label || humanizeLabel(key),
    type,
    section: schema?.advanced ? "Advanced" : section,
    description: schema?.description,
    options,
    min: schema?.min,
    max: schema?.max,
    step:
      schema?.step != null
        ? schema.step
        : type === "integer"
          ? 1
          : type === "number"
            ? 0.0001
            : undefined,
    placeholder,
    advanced: Boolean(schema?.advanced),
    required: Boolean(schema?.required),
  };
}

export function resolveConfigFields(
  nodeTypeDef: NodeTypeDef | null | undefined,
  config: Record<string, any> | null | undefined,
): ResolvedConfigField[] {
  const safeConfig = config || {};
  const schemaMap = nodeTypeDef?.config_schema || {};
  const fromSchema: ResolvedConfigField[] = Object.keys(schemaMap).map((key) =>
    resolveSingleField(key, safeConfig[key], schemaMap[key]),
  );

  const schemaKeys = new Set(Object.keys(schemaMap));
  const extraKeys = Object.keys(safeConfig).filter((key) => !schemaKeys.has(key));
  const inferred = extraKeys
    .sort((a, b) => a.localeCompare(b))
    .map((key) => resolveSingleField(key, safeConfig[key]));

  return [...fromSchema, ...inferred];
}

export function sortSections(sections: string[]): string[] {
  const order = new Map(SECTION_ORDER.map((x, i) => [x, i]));
  return [...sections].sort((a, b) => {
    const ai = order.has(a) ? (order.get(a) as number) : Number.MAX_SAFE_INTEGER;
    const bi = order.has(b) ? (order.get(b) as number) : Number.MAX_SAFE_INTEGER;
    if (ai !== bi) return ai - bi;
    return a.localeCompare(b);
  });
}
