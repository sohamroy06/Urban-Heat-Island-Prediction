import React, { useState, useEffect } from 'react';
import {
    PieChart, Pie, Cell, ResponsiveContainer, Tooltip,
} from 'recharts';
import { fetchCityStats, fetchModelInfo } from '../api/shadowmap';

const RISK_COLORS = {
    critical: '#ef4444',
    hot: '#f97316',
    moderate: '#eab308',
    cool: '#06b6d4',
};

function DonutChart({ categories }) {
    if (!categories) return null;

    const data = [
        { name: 'Critical', value: categories.critical.count, color: RISK_COLORS.critical },
        { name: 'Hot', value: categories.hot.count, color: RISK_COLORS.hot },
        { name: 'Moderate', value: categories.moderate.count, color: RISK_COLORS.moderate },
        { name: 'Cool', value: categories.cool.count, color: RISK_COLORS.cool },
    ].filter(d => d.value > 0);

    return (
        <div className="relative">
            <ResponsiveContainer width="100%" height={160}>
                <PieChart>
                    <Pie
                        data={data}
                        cx="50%"
                        cy="50%"
                        innerRadius={45}
                        outerRadius={65}
                        paddingAngle={3}
                        dataKey="value"
                        startAngle={90}
                        endAngle={-270}
                    >
                        {data.map((entry, idx) => (
                            <Cell key={idx} fill={entry.color} stroke="none" />
                        ))}
                    </Pie>
                    <Tooltip
                        contentStyle={{
                            background: '#1a1d27',
                            border: '1px solid #363c50',
                            borderRadius: 6,
                            fontSize: 11,
                            color: '#e2e8f0',
                        }}
                        formatter={(val, name) => [`${val} blocks`, name]}
                    />
                </PieChart>
            </ResponsiveContainer>
            <div className="flex flex-wrap justify-center gap-x-3 gap-y-1 mt-1">
                {data.map((d) => (
                    <div key={d.name} className="flex items-center gap-1.5">
                        <span
                            className="inline-block w-2 h-2 rounded-full"
                            style={{ backgroundColor: d.color }}
                        ></span>
                        <span className="text-[10px] text-gray-400">
                            {d.name} ({d.value})
                        </span>
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
        <div className="space-y-3 animate-fade-in">
            <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                Model Performance
            </h4>
            <div className="grid grid-cols-2 gap-2">
                <div className="bg-dark-600 rounded-lg px-3 py-2 border border-dark-400 text-center">
                    <div className="text-[10px] text-gray-500">R² Score</div>
                    <div className="text-sm font-bold text-emerald-400">{modelInfo.r2?.toFixed(3)}</div>
                </div>
                <div className="bg-dark-600 rounded-lg px-3 py-2 border border-dark-400 text-center">
                    <div className="text-[10px] text-gray-500">RMSE</div>
                    <div className="text-sm font-bold text-amber-400">{modelInfo.rmse?.toFixed(2)}°C</div>
                </div>
                <div className="bg-dark-600 rounded-lg px-3 py-2 border border-dark-400 text-center">
                    <div className="text-[10px] text-gray-500">MAE</div>
                    <div className="text-sm font-bold text-cyan-400">{modelInfo.mae?.toFixed(2)}°C</div>
                </div>
                <div className="bg-dark-600 rounded-lg px-3 py-2 border border-dark-400 text-center">
                    <div className="text-[10px] text-gray-500">Spatial CV R²</div>
                    <div className="text-sm font-bold text-purple-400">{modelInfo.spatial_cv_r2?.toFixed(3)}</div>
                </div>
            </div>

            <div>
                <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                    Feature Importance
                </h4>
                <div className="space-y-1.5">
                    {sorted.map(([feature, pct]) => (
                        <div key={feature} className="flex items-center gap-2">
                            <div className="flex-1 min-w-0">
                                <div className="flex justify-between mb-0.5">
                                    <span className="text-[10px] text-gray-400 truncate">
                                        {feature.replace(/_/g, ' ')}
                                    </span>
                                    <span className="text-[10px] text-gray-500 font-mono">
                                        {pct.toFixed(1)}%
                                    </span>
                                </div>
                                <div className="h-1.5 bg-dark-400 rounded-full overflow-hidden">
                                    <div
                                        className="h-full rounded-full bg-gradient-to-r from-amber-500 to-orange-500 transition-all duration-500"
                                        style={{ width: `${Math.min(pct, 100)}%` }}
                                    ></div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="text-[10px] text-gray-600 pt-1 border-t border-dark-400">
                Model: {modelInfo.model_type} | Train: {modelInfo.n_train} | Test: {modelInfo.n_test}
            </div>
        </div>
    );
}

export default function CityStats({ onSelectBlock }) {
    const [cityStats, setCityStats] = useState(null);
    const [modelInfo, setModelInfo] = useState(null);
    const [showModel, setShowModel] = useState(false);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        async function loadData() {
            try {
                const [stats, info] = await Promise.all([
                    fetchCityStats(),
                    fetchModelInfo(),
                ]);
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
                <div className="w-5 h-5 border-2 border-amber-500 border-t-transparent rounded-full animate-spin"></div>
                <span className="text-sm text-gray-400">Loading city stats...</span>
            </div>
        );
    }

    if (!cityStats) {
        return (
            <div className="p-4 text-sm text-gray-500">Failed to load city statistics.</div>
        );
    }

    return (
        <div className="p-4 space-y-4 overflow-y-auto max-h-[calc(100vh-56px)]">
            {/* Header */}
            <div>
                <h2 className="text-sm font-bold text-white flex items-center gap-2">
                    <span className="text-lg">🌆</span> Delhi UHI Overview
                </h2>
            </div>

            {/* Summary Stats */}
            <div className="grid grid-cols-2 gap-2">
                <div className="bg-dark-600 rounded-lg px-3 py-2 border border-dark-400">
                    <div className="text-[10px] text-gray-500">Mean LST</div>
                    <div className="text-base font-bold text-white">{cityStats.city_mean_lst}°C</div>
                </div>
                <div className="bg-dark-600 rounded-lg px-3 py-2 border border-dark-400">
                    <div className="text-[10px] text-gray-500">Max LST</div>
                    <div className="text-base font-bold text-red-400">{cityStats.max_lst}°C</div>
                </div>
                <div className="bg-dark-600 rounded-lg px-3 py-2 border border-dark-400">
                    <div className="text-[10px] text-gray-500">Min LST</div>
                    <div className="text-base font-bold text-cyan-400">{cityStats.min_lst}°C</div>
                </div>
                <div className="bg-dark-600 rounded-lg px-3 py-2 border border-dark-400 glow-hot">
                    <div className="text-[10px] text-gray-500">UHI Intensity</div>
                    <div className="text-base font-bold text-orange-400">{cityStats.uhi_intensity}°C</div>
                </div>
            </div>

            {/* Category Donut */}
            <DonutChart categories={cityStats.categories} />

            {/* Top 5 Hottest */}
            <div>
                <h4 className="text-xs font-semibold text-red-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <span>🔥</span> Top 5 Hottest Blocks
                </h4>
                <div className="space-y-1">
                    {cityStats.top5_hottest?.map((block, idx) => (
                        <button
                            key={block.block_id}
                            onClick={() => onSelectBlock && onSelectBlock(block.block_id)}
                            className="w-full flex items-center justify-between bg-dark-600 hover:bg-dark-500 rounded-lg px-3 py-2 border border-dark-400 transition-colors cursor-pointer text-left"
                        >
                            <div className="flex items-center gap-2 min-w-0">
                                <span className="text-xs font-bold text-red-400 w-4">#{idx + 1}</span>
                                <div className="min-w-0">
                                    <p className="text-xs text-gray-200 truncate">{block.block_name}</p>
                                    <p className="text-[9px] text-gray-500">{block.ward}</p>
                                </div>
                            </div>
                            <div className="text-right flex-shrink-0 ml-2">
                                <p className="text-xs font-bold text-red-400">{block.predicted_lst}°C</p>
                                <p className="text-[8px] text-gray-600">{block.ci_lower}–{block.ci_upper}</p>
                            </div>
                        </button>
                    ))}
                </div>
            </div>

            {/* Top 5 Coolest */}
            <div>
                <h4 className="text-xs font-semibold text-cyan-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <span>❄️</span> Top 5 Coolest Blocks
                </h4>
                <div className="space-y-1">
                    {cityStats.top5_coolest?.map((block, idx) => (
                        <button
                            key={block.block_id}
                            onClick={() => onSelectBlock && onSelectBlock(block.block_id)}
                            className="w-full flex items-center justify-between bg-dark-600 hover:bg-dark-500 rounded-lg px-3 py-2 border border-dark-400 transition-colors cursor-pointer text-left"
                        >
                            <div className="flex items-center gap-2 min-w-0">
                                <span className="text-xs font-bold text-cyan-400 w-4">#{idx + 1}</span>
                                <div className="min-w-0">
                                    <p className="text-xs text-gray-200 truncate">{block.block_name}</p>
                                    <p className="text-[9px] text-gray-500">{block.ward}</p>
                                </div>
                            </div>
                            <div className="text-right flex-shrink-0 ml-2">
                                <p className="text-xs font-bold text-cyan-400">{block.predicted_lst}°C</p>
                                <p className="text-[8px] text-gray-600">{block.ci_lower}–{block.ci_upper}</p>
                            </div>
                        </button>
                    ))}
                </div>
            </div>

            {/* Model Info Toggle */}
            <div className="border-t border-dark-400 pt-3">
                <button
                    onClick={() => setShowModel(!showModel)}
                    className="w-full flex items-center justify-between text-xs text-gray-400 hover:text-gray-200 transition-colors py-1"
                >
                    <span className="font-semibold uppercase tracking-wider">
                        {showModel ? '▼' : '▶'} Model Info
                    </span>
                    <span className="text-[10px] text-gray-600">
                        R²: {modelInfo?.r2?.toFixed(3)}
                    </span>
                </button>
                {showModel && <div className="mt-3"><ModelInfoPanel modelInfo={modelInfo} /></div>}
            </div>
        </div>
    );
}
