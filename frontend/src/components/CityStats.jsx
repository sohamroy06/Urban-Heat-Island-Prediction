import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    PieChart, Pie, Cell, ResponsiveContainer, Tooltip,
} from 'recharts';
import { fetchCityStats, fetchModelInfo } from '../api/shadowmap';
import AnimatedNumber from './AnimatedNumber';
import { SkylineIcon, SearchIcon, FlameIcon, DropletIcon, ChevronIcon } from './Icons';

const RISK_COLORS = {
    critical: '#991b1b',
    hot: '#f97316',
    moderate: '#eab308',
    cool: '#0ea5e9',
};

function DonutChart({ categories }) {
    if (!categories) return null;

    const data = [
        { name: 'Critical', value: categories.critical.count, color: RISK_COLORS.critical },
        { name: 'Hot', value: categories.hot.count, color: RISK_COLORS.hot },
        { name: 'Moderate', value: categories.moderate.count, color: RISK_COLORS.moderate },
        { name: 'Cool', value: categories.cool.count, color: RISK_COLORS.cool },
    ].filter((d) => d.value > 0);

    return (
        <div className="relative">
            <ResponsiveContainer width="100%" height={150}>
                <PieChart>
                    <Pie
                        data={data}
                        cx="50%"
                        cy="50%"
                        innerRadius={42}
                        outerRadius={62}
                        paddingAngle={3}
                        dataKey="value"
                        startAngle={90}
                        endAngle={-270}
                        animationDuration={600}
                    >
                        {data.map((entry, idx) => (
                            <Cell key={idx} fill={entry.color} stroke="none" />
                        ))}
                    </Pie>
                    <Tooltip
                        contentStyle={{
                            background: '#1c1917',
                            border: '1px solid #332d2a',
                            borderRadius: 8,
                            fontSize: 11,
                            color: '#e8e0d8',
                        }}
                        formatter={(val, name) => [`${val} blocks`, name]}
                    />
                </PieChart>
            </ResponsiveContainer>
            <div className="flex flex-wrap justify-center gap-x-3 gap-y-1 mt-1">
                {data.map((d) => (
                    <div key={d.name} className="flex items-center gap-1.5">
                        <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: d.color }} />
                        <span className="text-[10px] text-ink-300">{d.name} ({d.value})</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

function ModelInfoPanel({ modelInfo }) {
    if (!modelInfo) return null;
    const importances = modelInfo.feature_importances || {};
    const sorted = Object.entries(importances).sort((a, b) => b[1] - a[1]);

    return (
        <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2">
                <MetricTile label="Repeated CV R²" value={modelInfo.repeated_cv_r2_mean?.toFixed(2)} color="text-emerald-400" />
                <MetricTile label="RMSE" value={`${modelInfo.rmse?.toFixed(2)}°C`} color="text-signal-300" />
                <MetricTile label="MAE" value={`${modelInfo.mae?.toFixed(2)}°C`} color="text-amber-300" />
                <MetricTile label="Spatial CV R²" value={modelInfo.spatial_cv_r2?.toFixed(2)} color="text-rose-300" />
            </div>

            <div>
                <h4 className="text-[10px] font-semibold text-ink-400 uppercase tracking-wider mb-2">
                    Feature Importance
                </h4>
                <div className="space-y-1.5">
                    {sorted.map(([feature, pct]) => (
                        <div key={feature}>
                            <div className="flex justify-between mb-0.5">
                                <span className="text-[10px] text-ink-300 truncate">{feature.replace(/_/g, ' ')}</span>
                                <span className="text-[10px] text-ink-400 font-mono">{pct.toFixed(1)}%</span>
                            </div>
                            <div className="h-1.5 bg-ink-700 rounded-full overflow-hidden">
                                <motion.div
                                    className="h-full rounded-full bg-gradient-to-r from-signal-500 to-signal-300"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${Math.min(pct, 100)}%` }}
                                    transition={{ duration: 0.5, ease: 'easeOut' }}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="text-[10px] text-ink-500 pt-1 border-t border-ink-700">
                {modelInfo.model_type} · train {modelInfo.n_train} · test {modelInfo.n_test}
            </div>
        </div>
    );
}

function MetricTile({ label, value, color }) {
    return (
        <div className="bg-ink-800 rounded-lg px-3 py-2 border border-ink-600 text-center">
            <div className="text-[9px] text-ink-400 uppercase tracking-wide">{label}</div>
            <div className={`text-sm font-semibold font-mono ${color}`}>{value ?? '—'}</div>
        </div>
    );
}

function WardSearch({ geojsonData, onSelectBlock }) {
    const [query, setQuery] = useState('');
    const [focused, setFocused] = useState(false);

    const results = useMemo(() => {
        if (!query.trim() || !geojsonData) return [];
        const q = query.trim().toLowerCase();
        return geojsonData.features
            .filter((f) => {
                const p = f.properties;
                return (p.block_name || '').toLowerCase().includes(q) || (p.ward || '').toLowerCase().includes(q);
            })
            .slice(0, 6);
    }, [query, geojsonData]);

    return (
        <div className="relative">
            <div className="relative">
                <SearchIcon width={14} height={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-400" />
                <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onFocus={() => setFocused(true)}
                    onBlur={() => setTimeout(() => setFocused(false), 120)}
                    placeholder="Search a ward or block…"
                    className="w-full bg-ink-800 border border-ink-600 rounded-lg pl-8 pr-3 py-2 text-xs text-ink-100 placeholder:text-ink-400 focus:outline-none focus:border-signal-500 transition-colors"
                />
            </div>
            <AnimatePresence>
                {focused && results.length > 0 && (
                    <motion.div
                        initial={{ opacity: 0, y: -4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -4 }}
                        transition={{ duration: 0.12 }}
                        className="absolute z-30 mt-1 w-full bg-ink-800 border border-ink-600 rounded-lg overflow-hidden shadow-xl"
                    >
                        {results.map((f) => (
                            <button
                                key={f.properties.block_id}
                                onClick={() => { onSelectBlock(f.properties.block_id); setQuery(''); }}
                                className="w-full text-left px-3 py-2 text-xs text-ink-200 hover:bg-ink-700 transition-colors flex items-center justify-between"
                            >
                                <span className="truncate">{f.properties.block_name}</span>
                                <span className="text-ink-500 text-[10px] ml-2 flex-shrink-0">{f.properties.ward}</span>
                            </button>
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

function BlockRow({ block, idx, colorClass, onSelectBlock, delay }) {
    return (
        <motion.button
            initial={{ opacity: 0, x: -6 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay, duration: 0.25 }}
            whileHover={{ x: 2 }}
            onClick={() => onSelectBlock && onSelectBlock(block.block_id)}
            className="w-full flex items-center justify-between bg-ink-800 hover:bg-ink-700 rounded-lg px-3 py-2 border border-ink-600 transition-colors cursor-pointer text-left"
        >
            <div className="flex items-center gap-2 min-w-0">
                <span className={`text-[11px] font-mono font-semibold w-4 ${colorClass}`}>{idx + 1}</span>
                <div className="min-w-0">
                    <p className="text-xs text-ink-100 truncate">{block.block_name}</p>
                    <p className="text-[9px] text-ink-400">{block.ward}</p>
                </div>
            </div>
            <div className="text-right flex-shrink-0 ml-2">
                <p className={`text-xs font-mono font-semibold ${colorClass}`}>{block.predicted_lst}°C</p>
            </div>
        </motion.button>
    );
}

export default function CityStats({ geojsonData, onSelectBlock }) {
    const [cityStats, setCityStats] = useState(null);
    const [modelInfo, setModelInfo] = useState(null);
    const [showModel, setShowModel] = useState(false);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        async function loadData() {
            try {
                const [stats, info] = await Promise.all([fetchCityStats(), fetchModelInfo()]);
                setCityStats(stats);
                setModelInfo(info);
            } catch (err) {
                console.error('Failed to load city stats:', err);
            } finally {
                setLoading(false);
            }
        }
        loadData();
    }, []);

    if (loading) {
        return (
            <div className="p-4 flex items-center gap-3">
                <div className="w-4 h-4 border-2 border-signal-400 border-t-transparent rounded-full animate-spin" />
                <span className="text-sm text-ink-300">Loading city stats…</span>
            </div>
        );
    }

    if (!cityStats) {
        return <div className="p-4 text-sm text-ink-400">Failed to load city statistics.</div>;
    }

    return (
        <div className="p-4 space-y-4 lg:overflow-y-auto lg:max-h-[calc(100vh-56px)]">
            <div className="flex items-center gap-2">
                <SkylineIcon width={18} height={18} className="text-signal-400" />
                <h2 className="font-display text-sm font-semibold text-ink-100">Delhi UHI Overview</h2>
            </div>

            <WardSearch geojsonData={geojsonData} onSelectBlock={onSelectBlock} />

            {/* Hero stat */}
            <div className="rounded-2xl border border-ink-600 bg-gradient-to-br from-ink-800 to-ink-900 px-4 py-4">
                <div className="text-[10px] text-ink-400 uppercase tracking-wider mb-1">City Mean Surface Temp</div>
                <div className="flex items-baseline gap-1">
                    <AnimatedNumber
                        value={cityStats.city_mean_lst}
                        decimals={1}
                        className="font-display text-4xl font-semibold text-ink-100"
                    />
                    <span className="text-lg text-ink-400">°C</span>
                </div>
            </div>

            {/* Secondary stats */}
            <div className="grid grid-cols-3 gap-2">
                <MetricTile label="Max" value={<AnimatedNumber value={cityStats.max_lst} decimals={1} suffix="°" />} color="text-red-400" />
                <MetricTile label="Min" value={<AnimatedNumber value={cityStats.min_lst} decimals={1} suffix="°" />} color="text-sky-400" />
                <MetricTile label="UHI Δ" value={<AnimatedNumber value={cityStats.uhi_intensity} decimals={1} suffix="°" />} color="text-orange-400" />
            </div>

            <DonutChart categories={cityStats.categories} />

            <div>
                <h4 className="text-[10px] font-semibold text-red-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <FlameIcon width={12} height={12} /> Top 5 Hottest
                </h4>
                <div className="space-y-1">
                    {cityStats.top5_hottest?.map((block, idx) => (
                        <BlockRow key={block.block_id} block={block} idx={idx} colorClass="text-red-400" onSelectBlock={onSelectBlock} delay={idx * 0.04} />
                    ))}
                </div>
            </div>

            <div>
                <h4 className="text-[10px] font-semibold text-sky-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <DropletIcon width={12} height={12} /> Top 5 Coolest
                </h4>
                <div className="space-y-1">
                    {cityStats.top5_coolest?.map((block, idx) => (
                        <BlockRow key={block.block_id} block={block} idx={idx} colorClass="text-sky-400" onSelectBlock={onSelectBlock} delay={idx * 0.04} />
                    ))}
                </div>
            </div>

            <div className="border-t border-ink-700 pt-3">
                <button
                    onClick={() => setShowModel(!showModel)}
                    className="w-full flex items-center justify-between text-xs text-ink-300 hover:text-ink-100 transition-colors py-1"
                >
                    <span className="font-semibold uppercase tracking-wider flex items-center gap-1.5">
                        <ChevronIcon open={showModel} width={12} height={12} /> Model Info
                    </span>
                    <span className="text-[10px] text-ink-500 font-mono">R² {modelInfo?.repeated_cv_r2_mean?.toFixed(2)}</span>
                </button>
                <AnimatePresence>
                    {showModel && (
                        <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.2 }}
                            className="overflow-hidden"
                        >
                            <div className="pt-3"><ModelInfoPanel modelInfo={modelInfo} /></div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
}
