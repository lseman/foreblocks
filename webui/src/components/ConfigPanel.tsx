import React, { useState, useMemo, useEffect } from "react";
import { ChevronRight, Settings2, Sliders, PlayCircle, Database, Box, Cpu, Info } from "lucide-react";
import { useStore } from "../store/store";
import { motion, AnimatePresence } from "framer-motion";
import {
    resolveConfigFields,
    sortSections,
    type ResolvedConfigField,
} from "../utils/configSchema";

type HeadBlockStage = "serial" | "parallel";
type HeadBlockCombine = "invert" | "add" | "none";
type HeadBlockAlphaMode = "off" | "gate" | "soft";
type HeadBlockAlphaMix = "blend" | "residual";

interface HeadBlockConfig {
    head_node_id: string;
    name?: string;
    stage: HeadBlockStage;
    combine: HeadBlockCombine;
    add_project: boolean;
    alpha_mode: HeadBlockAlphaMode;
    alpha_mix_style: HeadBlockAlphaMix;
    alpha_init: number;
    alpha_trainable: boolean;
    weight_carry: boolean;
}

export const ConfigPanel: React.FC<{ variant?: "overlay" | "dock" }> = ({ variant = "overlay" }) => {
    const {
        nodes,
        connections,
        selectedNodeId,
        nodeTypes,
        showConfigPanel,
        setSelectedNodeId,
        setShowConfigPanel,
        updateNodes,
    } = useStore();

    const [collapsedSections, setCollapsedSections] = useState<Record<string, boolean>>({});
    const [jsonDrafts, setJsonDrafts] = useState<Record<string, string>>({});
    const [jsonErrors, setJsonErrors] = useState<Record<string, string>>({});

    const selectedNode = nodes.find((n) => n.id === selectedNodeId);

    const typeDef = useMemo(() =>
        selectedNode ? nodeTypes[selectedNode.type] : null
        , [selectedNode, nodeTypes]);

    const effectiveConfig = useMemo(
        () => ({
            ...(typeDef?.config || {}),
            ...(selectedNode?.config || {}),
        }),
        [selectedNode?.config, typeDef?.config],
    );

    useEffect(() => {
        setJsonDrafts({});
        setJsonErrors({});
    }, [selectedNodeId]);

    const updateNodeConfig = (key: string, value: any) => {
        updateNodes((ns) =>
            ns.map((n) =>
                n.id === selectedNodeId
                    ? { ...n, config: { ...n.config, [key]: value } }
                    : n
            )
        );
    };

    const updateNodeSubtype = (subtype: string) => {
        updateNodes((ns) =>
            ns.map((n) =>
                n.id === selectedNodeId ? { ...n, subtype } : n
            )
        );
    };

    const toggleSection = (section: string) => {
        setCollapsedSections(prev => ({ ...prev, [section]: !prev[section] }));
    };

    const fields = useMemo(
        () => resolveConfigFields(typeDef, effectiveConfig),
        [effectiveConfig, typeDef],
    );

    const isHeadComposer = selectedNode?.type === "head_composer";

    const headCandidates = useMemo(() => {
        return nodes.filter((n) => {
            if (!selectedNode || n.id === selectedNode.id) return false;
            const outputs = nodeTypes[n.type]?.outputs || [];
            return outputs.some((out) => String(out).toLowerCase().includes("head"));
        });
    }, [nodes, nodeTypes, selectedNode]);

    const defaultHeadNodeId = headCandidates[0]?.id || "";

    const headBlocks = useMemo<HeadBlockConfig[]>(() => {
        const raw = selectedNode?.config?.head_blocks;
        if (!Array.isArray(raw)) return [];
        return raw
            .filter((x) => x && typeof x === "object")
            .map((x: any) => ({
                head_node_id: String(x.head_node_id || ""),
                name: x.name == null ? "" : String(x.name),
                stage: x.stage === "parallel" ? "parallel" : "serial",
                combine: x.combine === "invert" || x.combine === "add" ? x.combine : "none",
                add_project: x.add_project !== false,
                alpha_mode: x.alpha_mode === "gate" || x.alpha_mode === "soft" ? x.alpha_mode : "off",
                alpha_mix_style: x.alpha_mix_style === "residual" ? "residual" : "blend",
                alpha_init: typeof x.alpha_init === "number" ? x.alpha_init : 0,
                alpha_trainable: x.alpha_trainable !== false,
                weight_carry: x.weight_carry !== false,
            }));
    }, [selectedNode?.config?.head_blocks]);

    const updateHeadBlocks = (next: HeadBlockConfig[]) => {
        updateNodeConfig("head_blocks", next);
    };

    const addHeadBlock = () => {
        const base: HeadBlockConfig = {
            head_node_id: defaultHeadNodeId,
            name: "",
            stage: "serial",
            combine: "none",
            add_project: true,
            alpha_mode: "off",
            alpha_mix_style: "blend",
            alpha_init: 0,
            alpha_trainable: true,
            weight_carry: true,
        };
        updateHeadBlocks([...headBlocks, base]);
    };

    const patchHeadBlock = (idx: number, patch: Partial<HeadBlockConfig>) => {
        const next = headBlocks.map((b, i) => (i === idx ? { ...b, ...patch } : b));
        updateHeadBlocks(next);
    };

    const removeHeadBlock = (idx: number) => {
        updateHeadBlocks(headBlocks.filter((_, i) => i !== idx));
    };

    const autofillHeadBlocksFromWiring = () => {
        if (!selectedNode) return;
        const incoming = connections.filter((c) => c.to === selectedNode.id);
        const existingByHeadId = new Map<string, HeadBlockConfig>();
        for (const block of headBlocks) {
            if (block.head_node_id) {
                existingByHeadId.set(block.head_node_id, block);
            }
        }

        const isHeadNode = (nodeId: string) => {
            const src = nodes.find((n) => n.id === nodeId);
            if (!src) return false;
            const outputs = nodeTypes[src.type]?.outputs || [];
            return outputs.some((out) => String(out).toLowerCase().includes("head"));
        };

        const orderedUniqueSourceIds: string[] = [];
        const stageBySource: Record<string, HeadBlockStage> = {};

        for (const c of incoming) {
            if (!isHeadNode(c.from)) continue;
            if (!orderedUniqueSourceIds.includes(c.from)) {
                orderedUniqueSourceIds.push(c.from);
            }
            const port = String(c.toPort || "").toLowerCase();
            if (port.includes("parallel")) stageBySource[c.from] = "parallel";
            else if (port.includes("serial") && !stageBySource[c.from]) stageBySource[c.from] = "serial";
            else if (!stageBySource[c.from]) stageBySource[c.from] = "serial";
        }

        const next: HeadBlockConfig[] = orderedUniqueSourceIds.map((srcId, idx) => {
            const stage = stageBySource[srcId] || "serial";
            const prev = existingByHeadId.get(srcId);
            if (prev) {
                return {
                    ...prev,
                    head_node_id: srcId,
                    stage,
                    name: prev.name || `${srcId}_${stage}_${idx + 1}`,
                };
            }
            return {
                head_node_id: srcId,
                name: `${srcId}_${stage}_${idx + 1}`,
                stage,
                combine: "none",
                add_project: true,
                alpha_mode: "off",
                alpha_mix_style: "blend",
                alpha_init: 0,
                alpha_trainable: true,
                weight_carry: true,
            };
        });

        if (next.length > 0) {
            updateHeadBlocks(next);
        }
    };

    const grouped = useMemo(() => {
        const g: Record<string, ResolvedConfigField[]> = {};
        fields.forEach((field) => {
            if (!g[field.section]) g[field.section] = [];
            g[field.section].push(field);
        });
        const sections = sortSections(Object.keys(g));
        return { sections, groups: g };
    }, [fields]);

    const coerceSelectValue = (
        raw: string,
        options?: Array<string | number | boolean>,
    ): string | number | boolean => {
        if (!options || options.length === 0) return raw;
        const match = options.find((opt) => String(opt) === raw);
        return match ?? raw;
    };

    const getJsonDraft = (
        key: string,
        current: any,
        type: "json" | "array",
    ): string => {
        const existing = jsonDrafts[key];
        if (existing != null) return existing;
        const fallback = current ?? (type === "array" ? [] : {});
        try {
            return JSON.stringify(fallback, null, 2);
        } catch {
            return type === "array" ? "[]" : "{}";
        }
    };

    const updateJsonField = (
        key: string,
        raw: string,
        type: "json" | "array",
    ) => {
        setJsonDrafts((prev) => ({ ...prev, [key]: raw }));
        try {
            const parsed = JSON.parse(raw);
            if (type === "array" && !Array.isArray(parsed)) {
                setJsonErrors((prev) => ({ ...prev, [key]: "Expected a JSON array." }));
                return;
            }
            updateNodeConfig(key, parsed);
            setJsonErrors((prev) => {
                const next = { ...prev };
                delete next[key];
                return next;
            });
        } catch {
            setJsonErrors((prev) => ({ ...prev, [key]: "Invalid JSON." }));
        }
    };

    const docked = variant === "dock";
    const isVisible = docked ? Boolean(selectedNode) : showConfigPanel;

    if (!selectedNode || !isVisible) return null;

    return (
        <AnimatePresence>
            {isVisible && (
                <motion.div
                    initial={docked ? false : { x: "100%" }}
                    animate={docked ? { opacity: 1 } : { x: 0 }}
                    exit={docked ? { opacity: 0 } : { x: "100%" }}
                    className={
                        docked
                            ? "h-full w-full bg-transparent flex flex-col"
                            : "fixed top-0 right-0 z-[60] flex h-screen w-[400px] flex-col border-l border-white/5 bg-neutral-900/40 backdrop-blur-3xl shadow-[-20px_0_50px_rgba(0,0,0,0.5)]"
                    }
                >
                    {/* Header */}
                    <div className={`flex items-center justify-between border-b border-white/5 px-6 py-6 ${docked ? "bg-[#0b0f17]" : ""}`}>
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
                                <Settings2 size={20} />
                            </div>
                            <div>
                                <h3 className="font-bold text-neutral-100 text-sm">{typeDef?.name || selectedNode.type}</h3>
                                <p className="text-[10px] text-neutral-500 uppercase tracking-widest font-bold">Configuration</p>
                            </div>
                        </div>
                        <button
                            onClick={() => {
                                setShowConfigPanel(false);
                                if (docked) {
                                    setSelectedNodeId(null);
                                }
                            }}
                            className="w-8 h-8 rounded-lg hover:bg-white/5 flex items-center justify-center text-neutral-500 hover:text-neutral-100 transition-colors"
                        >
                            <ChevronRight size={20} />
                        </button>
                    </div>

                    <div className={`flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-neutral-800 p-6 space-y-8 ${docked ? "bg-[#0b0f17]" : ""}`}>
                        {/* Subtype selector if applicable */}
                        {typeDef?.subtypes && typeDef.subtypes.length > 0 && (
                            <div className="space-y-3">
                                <div className="flex items-center gap-2 text-neutral-400">
                                    <Box size={14} />
                                    <span className="text-[10px] font-bold uppercase tracking-widest">Model Variant</span>
                                </div>
                                <select
                                    value={selectedNode.subtype || ""}
                                    onChange={(e) => updateNodeSubtype(e.target.value)}
                                    className="w-full bg-neutral-800/50 border border-white/5 rounded-xl px-4 py-3 text-sm text-neutral-200 focus:outline-none focus:ring-2 focus:ring-blue-500/30 transition-all"
                                >
                                    {typeDef.subtypes.map((subtype: string) => (
                                        <option key={subtype} value={subtype} className="bg-neutral-900">{subtype}</option>
                                    ))}
                                </select>
                            </div>
                        )}

                        {/* Property Grid */}
                        <div className="space-y-4">
                            {isHeadComposer && (
                                <div className="border border-purple-500/20 rounded-2xl overflow-hidden bg-purple-500/[0.04]">
                                    <div className="px-4 py-3 border-b border-purple-500/20 flex items-center justify-between">
                                        <div>
                                            <div className="text-xs font-bold text-purple-200">Head Blocks</div>
                                            <div className="text-[10px] text-purple-300/70">Compose heads as serial/parallel sub-blocks</div>
                                        </div>
                                        <button
                                            onClick={addHeadBlock}
                                            className="px-3 py-1.5 rounded-lg bg-purple-500/20 hover:bg-purple-500/30 text-purple-200 text-[11px] font-bold"
                                        >
                                            Add Block
                                        </button>
                                        <button
                                            onClick={autofillHeadBlocksFromWiring}
                                            className="px-3 py-1.5 rounded-lg bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-200 text-[11px] font-bold"
                                        >
                                            Autofill from Wiring
                                        </button>
                                    </div>
                                    <div className="p-4 space-y-3">
                                        {headBlocks.length === 0 && (
                                            <div className="text-[11px] text-neutral-400">
                                                No head blocks yet. Add a block and choose a head node.
                                            </div>
                                        )}
                                        {headBlocks.map((blk, idx) => (
                                            <div key={`${blk.head_node_id}-${idx}`} className="rounded-xl border border-white/10 bg-white/[0.02] p-3 space-y-2.5">
                                                <div className="flex items-center justify-between">
                                                    <div className="text-[11px] font-bold text-neutral-300">Block {idx + 1}</div>
                                                    <button
                                                        onClick={() => removeHeadBlock(idx)}
                                                        className="text-[10px] px-2 py-1 rounded-md bg-rose-500/10 text-rose-300 hover:bg-rose-500/20"
                                                    >
                                                        Remove
                                                    </button>
                                                </div>

                                                <div className="grid grid-cols-2 gap-2">
                                                    <select
                                                        value={blk.head_node_id}
                                                        onChange={(e) => patchHeadBlock(idx, { head_node_id: e.target.value })}
                                                        className="col-span-2 bg-neutral-800/30 border border-white/10 rounded-lg px-3 py-2 text-xs text-neutral-200"
                                                    >
                                                        {headCandidates.length === 0 && (
                                                            <option value="">No head nodes available</option>
                                                        )}
                                                        {headCandidates.map((hn) => (
                                                            <option key={hn.id} value={hn.id}>
                                                                {hn.id} · {nodeTypes[hn.type]?.name || hn.type}
                                                            </option>
                                                        ))}
                                                    </select>

                                                    <select
                                                        value={blk.stage}
                                                        onChange={(e) => patchHeadBlock(idx, { stage: e.target.value as HeadBlockStage })}
                                                        className="bg-neutral-800/30 border border-white/10 rounded-lg px-3 py-2 text-xs text-neutral-200"
                                                    >
                                                        <option value="serial">serial</option>
                                                        <option value="parallel">parallel</option>
                                                    </select>

                                                    <select
                                                        value={blk.combine}
                                                        onChange={(e) => patchHeadBlock(idx, { combine: e.target.value as HeadBlockCombine })}
                                                        className="bg-neutral-800/30 border border-white/10 rounded-lg px-3 py-2 text-xs text-neutral-200"
                                                    >
                                                        <option value="none">none</option>
                                                        <option value="add">add</option>
                                                        <option value="invert">invert</option>
                                                    </select>

                                                    <select
                                                        value={blk.alpha_mode}
                                                        onChange={(e) => patchHeadBlock(idx, { alpha_mode: e.target.value as HeadBlockAlphaMode })}
                                                        className="bg-neutral-800/30 border border-white/10 rounded-lg px-3 py-2 text-xs text-neutral-200"
                                                    >
                                                        <option value="off">alpha off</option>
                                                        <option value="gate">alpha gate</option>
                                                        <option value="soft">alpha soft</option>
                                                    </select>

                                                    <select
                                                        value={blk.alpha_mix_style}
                                                        onChange={(e) => patchHeadBlock(idx, { alpha_mix_style: e.target.value as HeadBlockAlphaMix })}
                                                        className="bg-neutral-800/30 border border-white/10 rounded-lg px-3 py-2 text-xs text-neutral-200"
                                                    >
                                                        <option value="blend">blend</option>
                                                        <option value="residual">residual</option>
                                                    </select>

                                                    <input
                                                        type="text"
                                                        value={blk.name || ""}
                                                        placeholder="Optional spec name"
                                                        onChange={(e) => patchHeadBlock(idx, { name: e.target.value })}
                                                        className="col-span-2 bg-neutral-800/30 border border-white/10 rounded-lg px-3 py-2 text-xs text-neutral-200"
                                                    />
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {grouped.sections.map((groupName) => (
                                <div key={groupName} className="border border-white/5 rounded-2xl overflow-hidden bg-white/[0.02]">
                                    <button
                                        onClick={() => toggleSection(groupName)}
                                        className="w-full flex items-center justify-between px-4 py-3 hover:bg-white/5 transition-colors text-left"
                                    >
                                        <div className="flex items-center gap-2.5">
                                            {groupName === "Model Architecture" && <Cpu size={14} className="text-purple-400" />}
                                            {groupName === "Sequence Settings" && <Sliders size={14} className="text-blue-400" />}
                                            {groupName === "Training Params" && <PlayCircle size={14} className="text-emerald-400" />}
                                            {groupName === "Data Settings" && <Database size={14} className="text-amber-400" />}
                                            <span className="text-xs font-bold text-neutral-300">{groupName}</span>
                                        </div>
                                        <ChevronRight size={16} className={`text-neutral-600 transition-transform duration-300 ${collapsedSections[groupName] ? "" : "rotate-90"}`} />
                                    </button>

                                    {!collapsedSections[groupName] && (
                                        <div className="p-4 space-y-4 border-t border-white/5">
                                            {grouped.groups[groupName].map((field) => {
                                                const value = effectiveConfig[field.key];
                                                return (
                                                    <div key={field.key} className="space-y-1.5">
                                                    <div className="flex items-center justify-between px-1">
                                                        <label className="text-[10px] font-bold text-neutral-500 uppercase tracking-wider capitalize">
                                                                {field.label}
                                                                {field.required && <span className="text-rose-400 ml-1">*</span>}
                                                        </label>
                                                    </div>
                                                        {field.description && (
                                                            <div className="flex items-start gap-1.5 text-[10px] text-neutral-500 px-1">
                                                                <Info size={12} className="mt-[1px] shrink-0" />
                                                                <span>{field.description}</span>
                                                            </div>
                                                        )}

                                                        {field.type === "boolean" ? (
                                                        <button
                                                                onClick={() => updateNodeConfig(field.key, !Boolean(value))}
                                                            className={`
                                                                flex items-center justify-between w-full px-4 py-2.5 rounded-xl border transition-all text-sm
                                                                    ${Boolean(value) ? 'bg-blue-500/10 border-blue-500/30 text-blue-400' : 'bg-neutral-800/30 border-white/5 text-neutral-500'}
                                                            `}
                                                        >
                                                                <span className="font-medium">{Boolean(value) ? "Enabled" : "Disabled"}</span>
                                                                <div className={`w-2 h-2 rounded-full ${Boolean(value) ? 'bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.5)]' : 'bg-neutral-600'}`} />
                                                        </button>
                                                        ) : field.type === "enum" && (field.options?.length || 0) > 0 ? (
                                                            <select
                                                                value={value == null ? "" : String(value)}
                                                                onChange={(e) =>
                                                                    updateNodeConfig(
                                                                        field.key,
                                                                        coerceSelectValue(e.target.value, field.options),
                                                                    )
                                                                }
                                                                className="w-full bg-neutral-800/30 border border-white/5 rounded-xl px-4 py-2.5 text-sm text-neutral-200 focus:outline-none focus:ring-2 focus:ring-blue-500/20 transition-all hover:bg-neutral-800/50"
                                                            >
                                                                {!field.required && (
                                                                    <option value="" className="bg-neutral-900">
                                                                        (empty)
                                                                    </option>
                                                                )}
                                                                {field.options!.map((option) => (
                                                                    <option
                                                                        key={String(option)}
                                                                        value={String(option)}
                                                                        className="bg-neutral-900"
                                                                    >
                                                                        {String(option)}
                                                                    </option>
                                                                ))}
                                                            </select>
                                                        ) : field.type === "integer" || field.type === "number" ? (
                                                            <input
                                                                type="number"
                                                                value={
                                                                    typeof value === "number"
                                                                        ? value
                                                                        : value == null
                                                                            ? ""
                                                                            : Number(value)
                                                                }
                                                                min={field.min}
                                                                max={field.max}
                                                                step={field.step}
                                                                onChange={(e) => {
                                                                    const raw = e.target.value;
                                                                    if (raw.trim() === "") {
                                                                        updateNodeConfig(field.key, field.required ? 0 : null);
                                                                        return;
                                                                    }
                                                                    const parsed =
                                                                        field.type === "integer"
                                                                            ? parseInt(raw, 10)
                                                                            : parseFloat(raw);
                                                                    if (!Number.isNaN(parsed)) {
                                                                        updateNodeConfig(field.key, parsed);
                                                                    }
                                                                }}
                                                                className="w-full bg-neutral-800/30 border border-white/5 rounded-xl px-4 py-2.5 text-sm text-neutral-200 focus:outline-none focus:ring-2 focus:ring-blue-500/20 transition-all hover:bg-neutral-800/50"
                                                            />
                                                        ) : field.type === "json" || field.type === "array" ? (
                                                            <div className="space-y-1.5">
                                                                <textarea
                                                                    value={getJsonDraft(field.key, value, field.type)}
                                                                    onChange={(e) =>
                                                                        updateJsonField(field.key, e.target.value, field.type)
                                                                    }
                                                                    rows={5}
                                                                    className="w-full bg-neutral-800/30 border border-white/5 rounded-xl px-4 py-2.5 text-xs font-mono text-neutral-200 focus:outline-none focus:ring-2 focus:ring-blue-500/20 transition-all hover:bg-neutral-800/50"
                                                                />
                                                                {jsonErrors[field.key] && (
                                                                    <div className="text-[10px] text-rose-400 px-1">
                                                                        {jsonErrors[field.key]}
                                                                    </div>
                                                                )}
                                                            </div>
                                                        ) : (
                                                        <input
                                                                type="text"
                                                                value={
                                                                    typeof value === "string"
                                                                        ? value
                                                                        : value == null
                                                                            ? ""
                                                                            : String(value)
                                                                }
                                                                placeholder={field.placeholder}
                                                                onChange={(e) => updateNodeConfig(field.key, e.target.value)}
                                                            className="w-full bg-neutral-800/30 border border-white/5 rounded-xl px-4 py-2.5 text-sm text-neutral-200 focus:outline-none focus:ring-2 focus:ring-blue-500/20 transition-all hover:bg-neutral-800/50"
                                                        />
                                                        )}
                                                </div>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="p-6 border-t border-white/5 bg-neutral-950/20">
                        <button
                            onClick={() => setSelectedNodeId(null)}
                            className="w-full h-11 rounded-xl bg-neutral-800 hover:bg-neutral-700 text-neutral-300 text-xs font-bold uppercase tracking-wider transition-all"
                        >
                            Deselect Node
                        </button>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};
