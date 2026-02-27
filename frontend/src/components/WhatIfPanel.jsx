import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
    LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';
import { fetchWhatIf } from '../api/shadowmap';

function InterventionCurve({ curves }) {
    if (!curves || Object.keys(curves).length === 0) return null;

    const chartConfig = {
        buildings: { label: 'Buildings Added', color: '#f97316', unit: '' },
        trees: { label: 'Trees Planted', color: '#10b981', unit: '' },
        albedo: { label: 'Albedo Increase (%)', color: '#06b6d4', unit: '%' },
    };

    return (
        <div className="mt-4 space-y-3">
            <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                Intervention Curves
            </h4>
            {Object.entries(curves).map(([type, data]) => {
                if (!data || data.length === 0) return null;
                const config = chartConfig[type];
                return (
                    <div key={type} className="bg-dark-600 rounded-lg p-3 border border-dark-400">
                        <p className="text-[10px] text-gray-500 mb-2">{config.label}</p>
                        <ResponsiveContainer width="100%" height={100}>
                            <LineChart data={data} margin={{ top: 2, right: 5, left: -15, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3f" />
                                <XAxis
                                    dataKey="value"
                                    tick={{ fontSize: 9, fill: '#64748b' }}
                                    axisLine={{ stroke: '#363c50' }}
                                    tickLine={false}
                                />
                                <YAxis
                                    tick={{ fontSize: 9, fill: '#64748b' }}
                                    axisLine={false}
                                    tickLine={false}
                                    tickFormatter={(v) => `${v > 0 ? '+' : ''}${v}°`}
                                    domain={['auto', 'auto']}
                                />
                                <Tooltip
                                    contentStyle={{
                                        background: '#1a1d27',
                                        border: '1px solid #363c50',
                                        borderRadius: 6,
                                        fontSize: 10,
                                        color: '#e2e8f0',
                                    }}
                                    formatter={(val) => [`${val > 0 ? '+' : ''}${val}°C`, 'ΔTemp']}
                                    labelFormatter={(val) => `${config.label}: ${val}${config.unit}`}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="delta_temp"
                                    stroke={config.color}
                                    strokeWidth={2}
                                    dot={false}
                                    activeDot={{ r: 3, fill: config.color }}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                );
            })}
        </div>
    );
}

export default function WhatIfPanel({ blockData }) {
    const [buildings, setBuildings] = useState(0);
    const [trees, setTrees] = useState(0);
    const [albedo, setAlbedo] = useState(0);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const debounceRef = useRef(null);

    const runSimulation = useCallback(async (b, t, a) => {
        if (!blockData) return;
        if (b === 0 && t === 0 && a === 0) {
            setResult(null);
            return;
        }

        setLoading(true);
        try {
            const data = await fetchWhatIf({
                block_id: blockData.block_id,
                delta_buildings: b,
                delta_trees: t,
                delta_albedo: a,
            });
            setResult(data);
        } catch (err) {
            console.error('What-If simulation failed:', err);
        } finally {
            setLoading(false);
        }
    }, [blockData]);

    useEffect(() => {
        if (debounceRef.current) clearTimeout(debounceRef.current);
        debounceRef.current = setTimeout(() => {
            runSimulation(buildings, trees, albedo);
        }, 300);
        return () => {
            if (debounceRef.current) clearTimeout(debounceRef.current);
        };
    }, [buildings, trees, albedo, runSimulation]);

    useEffect(() => {
        setBuildings(0);
        setTrees(0);
        setAlbedo(0);
        setResult(null);
    }, [blockData?.block_id]);

    if (!blockData) {
        return (
            <div className="p-4 text-center">
                <div className="text-4xl mb-3 opacity-50">🔬</div>
                <p className="text-sm text-gray-400">
                    Select a block first, then simulate urban interventions
                </p>
            </div>
        );
    }

    const deltaTemp = result ? result.delta_temp : 0;
    const deltaColor = deltaTemp < 0 ? 'text-emerald-400' : deltaTemp > 0 ? 'text-red-400' : 'text-gray-400';
    const deltaArrow = deltaTemp < 0 ? '↓' : deltaTemp > 0 ? '↑' : '→';

    return (
        <div className="p-4 space-y-4 overflow-y-auto max-h-[calc(100vh-160px)]">
            <div>
                <h3 className="text-sm font-semibold text-white mb-0.5">
                    What-If Simulator
                </h3>
                <p className="text-[10px] text-gray-500">
                    Simulate urban interventions for {blockData.block_name}
                </p>
            </div>

            {/* Sliders */}
            <div className="space-y-4">
                {/* Buildings Slider */}
                <div className="slider-container">
                    <div className="flex justify-between items-center mb-1">
                        <label className="text-xs text-gray-400 flex items-center gap-1.5">
                            <span>🏢</span> Add Buildings
                        </label>
                        <span className="text-xs font-mono text-amber-400 bg-dark-600 px-2 py-0.5 rounded">
                            +{buildings}
                        </span>
                    </div>
                    <input
                        type="range"
                        min={0}
                        max={50}
                        value={buildings}
                        onChange={(e) => setBuildings(parseInt(e.target.value))}
                    />
                    <div className="flex justify-between text-[9px] text-gray-600 mt-0.5">
                        <span>0</span><span>25</span><span>50</span>
                    </div>
                </div>

                {/* Trees Slider */}
                <div className="slider-container">
                    <div className="flex justify-between items-center mb-1">
                        <label className="text-xs text-gray-400 flex items-center gap-1.5">
                            <span>🌳</span> Plant Trees
                        </label>
                        <span className="text-xs font-mono text-emerald-400 bg-dark-600 px-2 py-0.5 rounded">
                            +{trees}
                        </span>
                    </div>
                    <input
                        type="range"
                        min={0}
                        max={500}
                        step={10}
                        value={trees}
                        onChange={(e) => setTrees(parseInt(e.target.value))}
                    />
                    <div className="flex justify-between text-[9px] text-gray-600 mt-0.5">
                        <span>0</span><span>250</span><span>500</span>
                    </div>
                </div>

                {/* Albedo Slider */}
                <div className="slider-container">
                    <div className="flex justify-between items-center mb-1">
                        <label className="text-xs text-gray-400 flex items-center gap-1.5">
                            <span>☀️</span> Roof Albedo
                        </label>
                        <span className="text-xs font-mono text-cyan-400 bg-dark-600 px-2 py-0.5 rounded">
                            +{albedo}%
                        </span>
                    </div>
                    <input
                        type="range"
                        min={0}
                        max={50}
                        value={albedo}
                        onChange={(e) => setAlbedo(parseInt(e.target.value))}
                    />
                    <div className="flex justify-between text-[9px] text-gray-600 mt-0.5">
                        <span>0%</span><span>25%</span><span>50%</span>
                    </div>
                </div>
            </div>

            {/* Result Display */}
            {(result || loading) && (
                <div className="bg-dark-600 rounded-xl p-4 border border-dark-400 panel-transition animate-fade-in">
                    {loading ? (
                        <div className="flex items-center justify-center gap-2 py-3">
                            <div className="w-4 h-4 border-2 border-amber-500 border-t-transparent rounded-full animate-spin"></div>
                            <span className="text-xs text-gray-400">Simulating...</span>
                        </div>
                    ) : result && (
                        <>
                            {/* Before → After */}
                            <div className="flex items-center justify-center gap-3 mb-3">
                                <div className="text-center">
                                    <p className="text-[9px] text-gray-500 uppercase">Before</p>
                                    <p className="text-lg font-bold text-gray-300">{result.baseline_lst}°C</p>
                                </div>
                                <div className="text-2xl animate-pulse">{deltaArrow}</div>
                                <div className="text-center">
                                    <p className="text-[9px] text-gray-500 uppercase">After</p>
                                    <p className="text-lg font-bold text-white">{result.predicted_lst}°C</p>
                                </div>
                            </div>

                            {/* Delta */}
                            <div className="text-center mb-3">
                                <span className={`text-3xl font-black ${deltaColor}`}>
                                    {deltaTemp > 0 ? '+' : ''}{deltaTemp}°C
                                </span>
                                <p className="text-[10px] text-gray-500 mt-1">
                                    CI: {result.confidence_interval.lower}°C – {result.confidence_interval.upper}°C
                                </p>
                            </div>

                            {/* Narrative */}
                            {result.narrative_explanation && (
                                <div className="bg-dark-800 rounded-lg p-3 border border-dark-400">
                                    <p className="text-xs text-gray-300 leading-relaxed">
                                        {result.narrative_explanation}
                                    </p>
                                </div>
                            )}

                            {/* Intervention Curves */}
                            <InterventionCurve curves={result.intervention_curves} />
                        </>
                    )}
                </div>
            )}
        </div>
    );
}
