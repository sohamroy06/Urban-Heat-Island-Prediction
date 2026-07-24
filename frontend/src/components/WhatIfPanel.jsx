import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';
import { fetchWhatIf, fetchWhatIfCityWide } from '../api/shadowmap';
import AnimatedNumber from './AnimatedNumber';
import { BuildingIcon, TreeIcon, SunIcon, FlaskIcon, SparklesIcon } from './Icons';

function InterventionCurve({ curves }) {
    if (!curves || Object.keys(curves).length === 0) return null;

    const chartConfig = {
        buildings: { label: 'Buildings Added', color: '#fb923c', unit: '' },
        trees: { label: 'Trees Planted', color: '#34d399', unit: '' },
        albedo: { label: 'Albedo Increase (%)', color: '#38bdf8', unit: '%' },
    };

    return (
        <div className="mt-4 space-y-3">
            <h4 className="text-[10px] font-semibold text-ink-400 uppercase tracking-wider">Intervention Curves</h4>
            {Object.entries(curves).map(([type, data]) => {
                if (!data || data.length === 0) return null;
                const config = chartConfig[type];
                return (
                    <div key={type} className="bg-ink-800 rounded-lg p-3 border border-ink-600">
                        <p className="text-[10px] text-ink-400 mb-2">{config.label}</p>
                        <ResponsiveContainer width="100%" height={100}>
                            <LineChart data={data} margin={{ top: 2, right: 5, left: -15, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#262220" />
                                <XAxis dataKey="value" tick={{ fontSize: 9, fill: '#665a52' }} axisLine={{ stroke: '#332d2a' }} tickLine={false} />
                                <YAxis
                                    tick={{ fontSize: 9, fill: '#665a52' }}
                                    axisLine={false}
                                    tickLine={false}
                                    tickFormatter={(v) => `${v > 0 ? '+' : ''}${v}°`}
                                    domain={['auto', 'auto']}
                                />
                                <Tooltip
                                    contentStyle={{ background: '#1c1917', border: '1px solid #332d2a', borderRadius: 8, fontSize: 10, color: '#e8e0d8' }}
                                    formatter={(val) => [`${val > 0 ? '+' : ''}${val}°C`, 'ΔTemp']}
                                    labelFormatter={(val) => `${config.label}: ${val}${config.unit}`}
                                />
                                <Line type="monotone" dataKey="delta_temp" stroke={config.color} strokeWidth={2} dot={false} activeDot={{ r: 3, fill: config.color }} animationDuration={500} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                );
            })}
        </div>
    );
}

function InterventionSliders({ buildings, setBuildings, trees, setTrees, albedo, setAlbedo }) {
    return (
        <div className="space-y-4">
            <SliderRow icon={BuildingIcon} label="Add Buildings" value={buildings} onChange={setBuildings} max={50} step={1} colorClass="text-amber-300" fmt={(v) => `+${v}`} marks={['0', '25', '50']} />
            <SliderRow icon={TreeIcon} label="Plant Trees" value={trees} onChange={setTrees} max={500} step={10} colorClass="text-emerald-300" fmt={(v) => `+${v}`} marks={['0', '250', '500']} />
            <SliderRow icon={SunIcon} label="Roof Albedo" value={albedo} onChange={setAlbedo} max={50} step={1} colorClass="text-sky-300" fmt={(v) => `+${v}%`} marks={['0%', '25%', '50%']} />
        </div>
    );
}

function SliderRow({ icon: Icon, label, value, onChange, max, step, colorClass, fmt, marks }) {
    return (
        <div className="slider-container">
            <div className="flex justify-between items-center mb-1">
                <label className="text-xs text-ink-300 flex items-center gap-1.5">
                    <Icon width={13} height={13} /> {label}
                </label>
                <span className={`text-xs font-mono bg-ink-800 px-2 py-0.5 rounded ${colorClass}`}>{fmt(value)}</span>
            </div>
            <input type="range" min={0} max={max} step={step} value={value} onChange={(e) => onChange(parseInt(e.target.value, 10))} />
            <div className="flex justify-between text-[9px] text-ink-500 mt-0.5">
                {marks.map((m) => <span key={m}>{m}</span>)}
            </div>
        </div>
    );
}

function ModeToggle({ mode, setMode }) {
    return (
        <div className="grid grid-cols-2 gap-1 p-1 bg-ink-800 rounded-lg border border-ink-600">
            {['single', 'citywide'].map((m) => (
                <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={`relative py-1.5 text-[11px] font-semibold rounded-md transition-colors flex items-center justify-center gap-1.5 ${mode === m ? 'text-ink-950' : 'text-ink-300 hover:text-ink-100'
                        }`}
                >
                    {mode === m && (
                        <motion.div layoutId="whatif-mode-bg" className="absolute inset-0 bg-signal-400 rounded-md -z-10" transition={{ type: 'spring', stiffness: 400, damping: 30 }} />
                    )}
                    {m === 'single' ? <FlaskIcon width={12} height={12} /> : <SparklesIcon width={12} height={12} />}
                    {m === 'single' ? 'Single Block' : 'City-Wide'}
                </button>
            ))}
        </div>
    );
}

function SingleBlockMode({ blockData }) {
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
            const data = await fetchWhatIf({ block_id: blockData.block_id, delta_buildings: b, delta_trees: t, delta_albedo: a });
            setResult(data);
        } catch (err) {
            console.error('What-If simulation failed:', err);
        } finally {
            setLoading(false);
        }
    }, [blockData]);

    useEffect(() => {
        if (debounceRef.current) clearTimeout(debounceRef.current);
        debounceRef.current = setTimeout(() => runSimulation(buildings, trees, albedo), 300);
        return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
    }, [buildings, trees, albedo, runSimulation]);

    useEffect(() => {
        setBuildings(0); setTrees(0); setAlbedo(0); setResult(null);
    }, [blockData?.block_id]);

    if (!blockData) {
        return (
            <div className="p-6 text-center flex flex-col items-center gap-3">
                <FlaskIcon width={28} height={28} className="text-ink-500" />
                <p className="text-sm text-ink-400 max-w-[200px]">Select a block first, then simulate urban interventions</p>
            </div>
        );
    }

    const deltaTemp = result ? result.delta_temp : 0;
    const deltaColor = deltaTemp < 0 ? 'text-emerald-400' : deltaTemp > 0 ? 'text-red-400' : 'text-ink-400';

    return (
        <div className="space-y-4">
            <div>
                <h3 className="font-display text-sm font-semibold text-ink-100 mb-0.5">What-If Simulator</h3>
                <p className="text-[10px] text-ink-400">Simulate urban interventions for {blockData.block_name}</p>
            </div>

            <InterventionSliders buildings={buildings} setBuildings={setBuildings} trees={trees} setTrees={setTrees} albedo={albedo} setAlbedo={setAlbedo} />

            <AnimatePresence>
                {(result || loading) && (
                    <motion.div
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -6 }}
                        className="bg-ink-800 rounded-xl p-4 border border-ink-600"
                    >
                        {loading ? (
                            <div className="flex items-center justify-center gap-2 py-3">
                                <div className="w-4 h-4 border-2 border-signal-400 border-t-transparent rounded-full animate-spin" />
                                <span className="text-xs text-ink-300">Simulating…</span>
                            </div>
                        ) : result && (
                            <>
                                <div className="flex items-center justify-center gap-3 mb-3">
                                    <div className="text-center">
                                        <p className="text-[9px] text-ink-500 uppercase">Before</p>
                                        <p className="text-lg font-mono font-semibold text-ink-300">{result.baseline_lst}°C</p>
                                    </div>
                                    <div className="text-xl text-ink-500">→</div>
                                    <div className="text-center">
                                        <p className="text-[9px] text-ink-500 uppercase">After</p>
                                        <p className="text-lg font-mono font-semibold text-ink-100">{result.predicted_lst}°C</p>
                                    </div>
                                </div>

                                <div className="text-center mb-3">
                                    <AnimatedNumber
                                        value={deltaTemp}
                                        decimals={2}
                                        prefix={deltaTemp > 0 ? '+' : ''}
                                        suffix="°C"
                                        className={`text-3xl font-display font-semibold ${deltaColor}`}
                                    />
                                    <p className="text-[10px] text-ink-500 mt-1 font-mono">
                                        CI: {result.confidence_interval.lower}°C – {result.confidence_interval.upper}°C
                                    </p>
                                </div>

                                {result.narrative_explanation && (
                                    <div className="bg-ink-900 rounded-lg p-3 border border-ink-700">
                                        <p className="text-xs text-ink-300 leading-relaxed">{result.narrative_explanation}</p>
                                    </div>
                                )}

                                <InterventionCurve curves={result.intervention_curves} />
                            </>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

function CityWideMode({ onCityWideResult, onCityWideReset }) {
    const [buildings, setBuildings] = useState(0);
    const [trees, setTrees] = useState(100);
    const [albedo, setAlbedo] = useState(0);
    const [scope, setScope] = useState('all');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const run = async () => {
        setLoading(true);
        try {
            const data = await fetchWhatIfCityWide({
                delta_buildings: buildings,
                delta_trees: trees,
                delta_albedo: albedo,
                scope,
                top_n: 20,
            });
            setResult(data);
            onCityWideResult(data.block_deltas);
        } catch (err) {
            console.error('City-wide What-If failed:', err);
        } finally {
            setLoading(false);
        }
    };

    const reset = () => {
        setResult(null);
        onCityWideReset();
    };

    const deltaMean = result?.delta_city_mean_lst ?? 0;
    const deltaColor = deltaMean < 0 ? 'text-emerald-400' : deltaMean > 0 ? 'text-red-400' : 'text-ink-400';

    return (
        <div className="space-y-4">
            <div>
                <h3 className="font-display text-sm font-semibold text-ink-100 mb-0.5">City-Wide Simulator</h3>
                <p className="text-[10px] text-ink-400">Apply one intervention across many wards at once</p>
            </div>

            <div className="grid grid-cols-2 gap-1 p-1 bg-ink-800 rounded-lg border border-ink-600">
                {[['all', 'All Wards'], ['hottest', 'Top 20 Hottest']].map(([val, label]) => (
                    <button
                        key={val}
                        onClick={() => setScope(val)}
                        className={`py-1.5 text-[11px] font-semibold rounded-md transition-colors ${scope === val ? 'bg-ink-700 text-signal-300' : 'text-ink-400 hover:text-ink-200'
                            }`}
                    >
                        {label}
                    </button>
                ))}
            </div>

            <InterventionSliders buildings={buildings} setBuildings={setBuildings} trees={trees} setTrees={setTrees} albedo={albedo} setAlbedo={setAlbedo} />

            <button
                onClick={run}
                disabled={loading}
                className="w-full py-2 rounded-lg bg-signal-500 hover:bg-signal-600 disabled:opacity-50 text-white text-xs font-semibold transition-colors flex items-center justify-center gap-2"
            >
                {loading ? (
                    <span className="w-3.5 h-3.5 border-2 border-white/60 border-t-transparent rounded-full animate-spin" />
                ) : (
                    <SparklesIcon width={13} height={13} />
                )}
                Run City-Wide Simulation
            </button>

            <AnimatePresence>
                {result && (
                    <motion.div
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -6 }}
                        className="bg-ink-800 rounded-xl p-4 border border-ink-600 space-y-3"
                    >
                        <div className="text-center">
                            <p className="text-[9px] text-ink-500 uppercase mb-1">City Mean Shift · {result.n_blocks_affected} blocks</p>
                            <div className="flex items-center justify-center gap-3">
                                <span className="text-sm font-mono text-ink-400">{result.baseline_city_stats.city_mean_lst}°C</span>
                                <span className="text-ink-500">→</span>
                                <span className="text-sm font-mono text-ink-100">{result.new_city_stats.city_mean_lst}°C</span>
                            </div>
                            <AnimatedNumber
                                value={deltaMean}
                                decimals={2}
                                prefix={deltaMean > 0 ? '+' : ''}
                                suffix="°C"
                                className={`block text-2xl font-display font-semibold mt-1 ${deltaColor}`}
                            />
                        </div>
                        <div className="bg-ink-900 rounded-lg p-3 border border-ink-700">
                            <p className="text-xs text-ink-300 leading-relaxed">{result.narrative_explanation}</p>
                        </div>
                        <button
                            onClick={reset}
                            className="w-full py-1.5 rounded-lg border border-ink-600 text-ink-300 hover:text-ink-100 hover:border-ink-500 text-[11px] font-semibold transition-colors"
                        >
                            Reset map to baseline
                        </button>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

export default function WhatIfPanel({ blockData, onCityWideResult, onCityWideReset }) {
    const [mode, setMode] = useState('single');

    return (
        <div className="p-4 space-y-4 lg:overflow-y-auto lg:max-h-[calc(100vh-160px)]">
            <ModeToggle mode={mode} setMode={setMode} />
            <AnimatePresence>
                {mode === 'single' ? (
                    <motion.div key="single" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
                        <SingleBlockMode blockData={blockData} />
                    </motion.div>
                ) : (
                    <motion.div key="citywide" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
                        <CityWideMode onCityWideResult={onCityWideResult} onCityWideReset={onCityWideReset} />
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
