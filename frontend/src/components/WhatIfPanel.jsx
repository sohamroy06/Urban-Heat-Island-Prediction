import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
    LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';
import { fetchWhatIf } from '../api/shadowmap';

function InterventionCurve({ curves }) {
    if (!curves || Object.keys(curves).length === 0) return null;

    const chartConfig = {
        buildings: { label: 'Buildings Added', color: 'var(--tw-colors-surface-tint)', unit: '' },
        trees: { label: 'Trees Planted', color: 'var(--tw-colors-secondary)', unit: '' },
        albedo: { label: 'Albedo Increase (%)', color: 'var(--tw-colors-primary)', unit: '%' },
    };

    return (
        <div className="flex flex-col gap-4 mt-2">
            <h3 className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-widest">
                Intervention Curves
            </h3>
            {Object.entries(curves).map(([type, data]) => {
                if (!data || data.length === 0) return null;
                const config = chartConfig[type];
                return (
                    <div key={type} className="h-28 w-full bg-surface-container rounded-lg border border-white/5 p-4 relative overflow-hidden flex flex-col">
                        <span className="font-body-md text-body-md text-on-surface-variant text-[11px] mb-2">{config.label}</span>
                        <div className="flex-1 min-h-0">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={data} margin={{ top: 2, right: 5, left: -15, bottom: 0 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                    <XAxis
                                        dataKey="value"
                                        tick={{ fontSize: 9, fill: 'var(--tw-colors-on-surface-variant)' }}
                                        axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                                        tickLine={false}
                                    />
                                    <YAxis
                                        tick={{ fontSize: 9, fill: 'var(--tw-colors-on-surface-variant)' }}
                                        axisLine={false}
                                        tickLine={false}
                                        tickFormatter={(v) => `${v > 0 ? '+' : ''}${v}°`}
                                        domain={['auto', 'auto']}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            background: 'var(--tw-colors-surface-container-high)',
                                            border: '1px solid rgba(255,255,255,0.1)',
                                            borderRadius: 6,
                                            fontSize: 10,
                                            color: 'var(--tw-colors-on-surface)',
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
            <div className="p-card-padding text-center">
                <div className="text-4xl mb-3 opacity-50"><span className="material-symbols-outlined text-[48px] text-on-surface-variant">science</span></div>
                <p className="font-body-md text-on-surface-variant">
                    Select a block first, then simulate urban interventions
                </p>
            </div>
        );
    }

    const deltaTemp = result ? result.delta_temp : 0;
    const deltaColor = deltaTemp < 0 ? 'text-secondary' : deltaTemp > 0 ? 'text-error' : 'text-on-surface-variant';
    const deltaArrow = deltaTemp < 0 ? 'arrow_downward' : deltaTemp > 0 ? 'arrow_upward' : 'trending_flat';

    return (
        <div className="p-card-padding flex flex-col gap-8 overflow-y-auto max-h-[calc(100vh-160px)]">
            <div className="flex flex-col gap-1">
                <h2 className="font-headline-md text-headline-md text-on-surface">What-If Simulator</h2>
                <p className="font-body-md text-body-md text-on-surface-variant text-[13px]">
                    Simulate urban interventions for {blockData.block_name}
                </p>
            </div>

            {/* Sliders Section */}
            <div className="flex flex-col gap-8">
                {/* Buildings Slider */}
                <div className="flex flex-col gap-3 slider-container">
                    <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2 text-on-surface">
                            <span className="material-symbols-outlined text-[18px] text-surface-tint" style={{ fontVariationSettings: "'FILL' 1" }}>apartment</span>
                            <span className="font-body-md text-body-md text-[14px]">Add Buildings</span>
                        </div>
                        <span className="font-numeric-data text-numeric-data text-surface-tint text-[14px] px-2 py-0.5 bg-surface-tint/10 rounded">
                            +{buildings}
                        </span>
                    </div>
                    <input
                        type="range"
                        min={0}
                        max={50}
                        value={buildings}
                        onChange={(e) => setBuildings(parseInt(e.target.value))}
                        style={{ '--tw-colors-primary': 'var(--tw-colors-surface-tint)' }}
                    />
                    <div className="flex justify-between font-label-caps text-label-caps text-on-surface-variant text-[10px] mt-1">
                        <span>0</span><span>25</span><span>50</span>
                    </div>
                </div>

                {/* Trees Slider */}
                <div className="flex flex-col gap-3 slider-container">
                    <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2 text-on-surface">
                            <span className="material-symbols-outlined text-[18px] text-secondary" style={{ fontVariationSettings: "'FILL' 1" }}>park</span>
                            <span className="font-body-md text-body-md text-[14px]">Plant Trees</span>
                        </div>
                        <span className="font-numeric-data text-numeric-data text-secondary text-[14px] px-2 py-0.5 bg-secondary/10 rounded">
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
                        style={{ '--tw-colors-primary': 'var(--tw-colors-secondary)' }}
                    />
                    <div className="flex justify-between font-label-caps text-label-caps text-on-surface-variant text-[10px] mt-1">
                        <span>0</span><span>250</span><span>500</span>
                    </div>
                </div>

                {/* Albedo Slider */}
                <div className="flex flex-col gap-3 slider-container">
                    <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2 text-on-surface">
                            <span className="material-symbols-outlined text-[18px] text-[#0ea5e9]" style={{ fontVariationSettings: "'FILL' 1" }}>wb_sunny</span>
                            <span className="font-body-md text-body-md text-[14px]">Roof Albedo</span>
                        </div>
                        <span className="font-numeric-data text-numeric-data text-[#0ea5e9] text-[14px] px-2 py-0.5 bg-[#0ea5e9]/10 rounded">
                            +{albedo}%
                        </span>
                    </div>
                    <input
                        type="range"
                        min={0}
                        max={50}
                        value={albedo}
                        onChange={(e) => setAlbedo(parseInt(e.target.value))}
                        style={{ '--tw-colors-primary': '#0ea5e9' }}
                    />
                    <div className="flex justify-between font-label-caps text-label-caps text-on-surface-variant text-[10px] mt-1">
                        <span>0%</span><span>25%</span><span>50%</span>
                    </div>
                </div>
            </div>

            {/* Result Display */}
            {(result || loading) && (
                <div className="bg-surface-container rounded-xl border border-white/10 p-6 flex flex-col items-center gap-6 mt-4 relative overflow-hidden backdrop-blur-md panel-transition animate-fade-in">
                    {/* Subtle glow effect */}
                    <div className={`absolute top-0 left-1/2 -translate-x-1/2 w-32 h-32 rounded-full blur-3xl pointer-events-none ${deltaTemp > 0 ? 'bg-error/10' : deltaTemp < 0 ? 'bg-secondary/10' : 'bg-white/5'}`}></div>
                    
                    {loading ? (
                        <div className="flex flex-col items-center justify-center gap-3 py-8 z-10">
                            <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
                            <span className="font-label-caps text-label-caps text-on-surface-variant">Simulating...</span>
                        </div>
                    ) : result && (
                        <>
                            {/* Before → After */}
                            <div className="flex items-center justify-center gap-8 w-full z-10">
                                <div className="flex flex-col items-center">
                                    <span className="font-label-caps text-label-caps text-on-surface-variant mb-1">Before</span>
                                    <span className="font-numeric-data text-numeric-data text-on-surface text-[22px]">{result.baseline_lst}°C</span>
                                </div>
                                <span className="material-symbols-outlined text-on-surface-variant text-3xl font-light">{deltaArrow}</span>
                                <div className="flex flex-col items-center">
                                    <span className="font-label-caps text-label-caps text-on-surface-variant mb-1">After</span>
                                    <span className="font-numeric-data text-numeric-data text-on-surface text-[22px]">{result.predicted_lst}°C</span>
                                </div>
                            </div>

                            {/* Delta */}
                            <div className="flex flex-col items-center gap-1 z-10">
                                <span className={`font-display-lg text-display-lg text-[42px] leading-none ${deltaColor}`}>
                                    {deltaTemp > 0 ? '+' : ''}{deltaTemp}°C
                                </span>
                                <span className="font-label-caps text-label-caps text-on-surface-variant mt-2 text-[10px]">
                                    CI: {result.confidence_interval.lower}°C - {result.confidence_interval.upper}°C
                                </span>
                            </div>

                            {/* Narrative */}
                            {result.narrative_explanation && (
                                <div className="bg-surface-container-low rounded-lg p-4 border border-white/5 z-10 mt-2 w-full">
                                    <p className="font-body-md text-body-md text-on-surface-variant text-[13px] leading-relaxed">
                                        {result.narrative_explanation}
                                    </p>
                                </div>
                            )}

                            {/* Intervention Curves */}
                            <div className="w-full z-10">
                                <InterventionCurve curves={result.intervention_curves} />
                            </div>
                        </>
                    )}
                </div>
            )}
        </div>
    );
}
