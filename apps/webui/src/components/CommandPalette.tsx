import React, { useState, useEffect, useRef } from 'react';
import { Search, Package, ArrowRight, Command } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '../store/store';

interface CommandPaletteProps {
    onAddNode: (type: string) => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({ onAddNode }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [query, setQuery] = useState('');
    const [selectedIndex, setSelectedIndex] = useState(0);
    const { nodeTypes, nodeCategories } = useStore();
    const inputRef = useRef<HTMLInputElement>(null);

    const flatNodes = Object.entries(nodeCategories).flatMap(([category, types]) =>
        types.map(type => ({
            type,
            category,
            name: nodeTypes[type]?.name || type,
            color: nodeTypes[type]?.color
        }))
    );

    const filteredNodes = flatNodes.filter(n =>
        n.name.toLowerCase().includes(query.toLowerCase()) ||
        n.category.toLowerCase().includes(query.toLowerCase())
    );

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'p') {
                e.preventDefault();
                setIsOpen(prev => !prev);
            }
            if (e.key === 'Escape') {
                setIsOpen(false);
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);

    useEffect(() => {
        if (isOpen) {
            setSelectedIndex(0);
            setTimeout(() => inputRef.current?.focus(), 10);
        } else {
            setQuery('');
        }
    }, [isOpen]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (filteredNodes.length === 0) {
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp' || e.key === 'Enter') {
                e.preventDefault();
            }
            return;
        }

        if (e.key === 'ArrowDown') {
            setSelectedIndex(prev => (prev + 1) % filteredNodes.length);
        } else if (e.key === 'ArrowUp') {
            setSelectedIndex(prev => (prev - 1 + filteredNodes.length) % filteredNodes.length);
        } else if (e.key === 'Enter') {
            if (filteredNodes[selectedIndex]) {
                onAddNode(filteredNodes[selectedIndex].type);
                setIsOpen(false);
            }
        }
    };

    const selectNode = (type: string) => {
        onAddNode(type);
        setIsOpen(false);
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={() => setIsOpen(false)}
                        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[100]"
                    />
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: -20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: -20 }}
                        className="fixed top-[20%] left-1/2 -translate-x-1/2 w-full max-w-xl bg-neutral-900 border border-neutral-800 rounded-2xl shadow-2xl z-[101] overflow-hidden"
                    >
                        <div className="flex items-center px-4 border-b border-neutral-800">
                            <Search className="w-5 h-5 text-neutral-500" />
                            <input
                                ref={inputRef}
                                value={query}
                                onChange={e => setQuery(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder="Search nodes..."
                                className="w-full h-14 bg-transparent border-none focus:ring-0 text-neutral-100 placeholder:text-neutral-600 text-base px-3"
                            />
                            <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-neutral-800 border border-neutral-700 text-[10px] text-neutral-400 font-mono">
                                <Command size={10} />
                                <span>P</span>
                            </div>
                        </div>

                        <div className="max-h-[400px] overflow-y-auto p-2 scrollbar-thin scrollbar-thumb-neutral-800">
                            {filteredNodes.length > 0 ? (
                                filteredNodes.map((node, index) => (
                                    <div
                                        key={`${node.type}-${index}`}
                                        onClick={() => selectNode(node.type)}
                                        onMouseEnter={() => setSelectedIndex(index)}
                                        className={`
                      flex items-center justify-between px-3 py-3 rounded-xl cursor-pointer transition-all duration-200
                      ${index === selectedIndex ? 'bg-blue-500/10 border border-blue-500/30' : 'hover:bg-neutral-800 border border-transparent'}
                    `}
                                    >
                                        <div className="flex items-center gap-3">
                                            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${node.color || 'bg-neutral-800'} shadow-lg`}>
                                                <Package className="w-5 h-5 text-white/90" />
                                            </div>
                                            <div>
                                                <div className="text-neutral-100 font-medium text-sm">{node.name}</div>
                                                <div className="text-neutral-500 text-[11px] uppercase tracking-wider font-semibold">{node.category}</div>
                                            </div>
                                        </div>
                                        {index === selectedIndex && (
                                            <div className="flex items-center gap-2 text-blue-400">
                                                <span className="text-[10px] font-medium uppercase">Add Node</span>
                                                <ArrowRight size={14} />
                                            </div>
                                        )}
                                    </div>
                                ))
                            ) : (
                                <div className="flex flex-col items-center justify-center py-12 text-neutral-500">
                                    <Package className="w-12 h-12 mb-3 opacity-20" />
                                    <p>No nodes found matching "{query}"</p>
                                </div>
                            )}
                        </div>

                        <div className="px-4 py-3 bg-neutral-950/50 border-t border-neutral-800 flex items-center justify-between text-[11px] text-neutral-500">
                            <div className="flex gap-4">
                                <span className="flex items-center gap-1"><kbd className="px-1.5 py-0.5 rounded bg-neutral-800 border border-neutral-700 text-[9px]">Enter</kbd> to select</span>
                                <span className="flex items-center gap-1"><kbd className="px-1.5 py-0.5 rounded bg-neutral-800 border border-neutral-700 text-[9px]">↑↓</kbd> to navigate</span>
                            </div>
                            <span>{filteredNodes.length} results</span>
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
};
