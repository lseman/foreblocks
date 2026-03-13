
export type NodeID = string;
export type PortName = string;

export type ConfigFieldType =
  | "string"
  | "number"
  | "integer"
  | "boolean"
  | "enum"
  | "json"
  | "array";

export interface ConfigFieldSchema {
  type?: ConfigFieldType;
  kind?: ConfigFieldType;
  label?: string;
  description?: string;
  options?: Array<string | number | boolean>;
  enum?: Array<string | number | boolean>;
  section?: string;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  advanced?: boolean;
  required?: boolean;
}

export interface NodeTypeDef {
  name?: string;
  color?: string;             // Tailwind-like bg/gradient class
  inputs?: string[];
  outputs?: string[];
  subtypes?: string[];
  config?: Record<string, any>;
  config_schema?: Record<string, ConfigFieldSchema>;
  py?: {
    ctor?: string;
    var_prefix?: string;
    role?: string; // e.g. 'trainer'
    imports?: string[];
    bind?: {
      args?: (string | number | boolean)[];
      kwargs?: Record<string, string | number | boolean>;
    };
    output_map?: Record<string, string>;
  };
}

export interface NodeData {
  id: NodeID;
  type: string;
  subtype: string | null;
  position: { x: number; y: number };
  config: Record<string, any>;
}

export interface Connection {
  id: string;
  from: NodeID;
  fromPort: PortName;
  to: NodeID;
  toPort: PortName;
}

export type NodeTypeMap = Record<string, NodeTypeDef>;
export type CategoryMap = Record<string, string[]>;

export type ExecutionStateMap = Record<NodeID, "running" | "success" | "error" | undefined>;

export type ResultKind = "plot" | "image" | "metrics" | "table" | "json" | "array" | "text";

export interface ExecutionResult {
  type: ResultKind;
  data: any;
}

export type ExecutionResultsMap = Record<NodeID, ExecutionResult | undefined>;

export interface WorkflowIssue {
  code: string;
  message: string;
  nodeId?: NodeID;
  connectionId?: string;
}

export interface WorkflowPreflightResult {
  errors: WorkflowIssue[];
  warnings: WorkflowIssue[];
}
