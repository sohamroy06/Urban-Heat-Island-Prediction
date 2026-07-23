import React, { useState, useEffect } from 'react';
import { fetchCityStats, fetchModelInfo } from '../api/shadowmap';

function DonutChart({ categories }) {
    if (!categories) return null;

    const counts = [
        { name: 'Critical', value: categories.critical.count, color: 'var(--tw-colors-error, #ffb4ab)', bg: 'bg-error' },
        { name: 'Hot', value: categories.hot.count, color: 'var(--tw-colors-primary-container, #f59e0b)', bg: 'bg-primary-container' },
        { name: 'Moderate', value: categories.moderate.count, color: 'var(--tw-colors-primary, #ffc174)', bg: 'bg-primary' },
        { name: 'Cool', value: categories.cool.count, color: 'var(--tw-colors-secondary, #4edea3)', bg: 'bg-secondary' },
    ];

    const total = counts.reduce((acc, curr) => acc + curr.value, 0);
    
    let currentPercent = 0;
    const gradientStops = counts.map(item => {
        if (total === 0) return '';
        const pct = (item.value / total) * 100;
        const stop = `${item.color} ${currentPercent}% ${currentPercent + pct}%`;
        currentPercent += pct;
        return stop;
    }).filter(Boolean).join(', ');

    return (
        <div className="w-full h-48 flex items-center justify-center relative my-4">
            <div 
                className="w-32 h-32 rounded-full relative" 
                style={{ background: `conic-gradient(${gradientStops || '#333539 0% 100%'})` }}
            >
                <div className="absolute inset-[18px] bg-surface-container-low rounded-full"></div>
            </div>
            
            <div className="absolute bottom-[-20px] w-full flex justify-center gap-4">
                {counts.filter(d => d.value > 0).map((d) => (
                    <div key={d.name} className="flex items-center gap-1.5">
                        <div className={`w-2 h-2 rounded-full ${d.bg}`}></div>
                        <span className="font-label-caps text-label-caps text-on-surface-variant text-[9px]">{d.name}</span>
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
        <div className="space-y-4 animate-fade-in mt-4">
            <h4 className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-widest">
                Model Performance
            </h4>
            <div className="grid grid-cols-2 gap-2">
                <div className="bg-surface-container rounded-lg p-3 border border-white/5 text-center">
                    <div className="font-label-caps text-label-caps text-on-surface-variant mb-1">R² Score</div>
                    <div className="font-numeric-data text-numeric-data text-secondary text-[16px]">{modelInfo.r2?.toFixed(3)}</div>
                </div>
                <div className="bg-surface-container rounded-lg p-3 border border-white/5 text-center">
                    <div className="font-label-caps text-label-caps text-on-surface-variant mb-1">RMSE</div>
                    <div className="font-numeric-data text-numeric-data text-primary text-[16px]">{modelInfo.rmse?.toFixed(2)}°C</div>
                </div>
                <div className="bg-surface-container rounded-lg p-3 border border-white/5 text-center">
                    <div className="font-label-caps text-label-caps text-on-surface-variant mb-1">MAE</div>
                    <div className="font-numeric-data text-numeric-data text-tertiary text-[16px]">{modelInfo.mae?.toFixed(2)}°C</div>
                </div>
                <div className="bg-surface-container rounded-lg p-3 border border-white/5 text-center">
                    <div className="font-label-caps text-label-caps text-on-surface-variant mb-1">Spatial CV R²</div>
                    <div className="font-numeric-data text-numeric-data text-primary-container text-[16px]">{modelInfo.spatial_cv_r2?.toFixed(3)}</div>
                </div>
            </div>

            <div>
                <h4 className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-widest mb-3">
                    Feature Importance
                </h4>
                <div className="space-y-2">
                    {sorted.map(([feature, pct]) => (
                        <div key={feature} className="flex flex-col gap-1">
                            <div className="flex justify-between items-center">
                                <span className="font-body-md text-[11px] text-on-surface-variant truncate">
                                    {feature.replace(/_/g, ' ')}
                                </span>
                                <span className="font-numeric-data text-[11px] text-on-surface-variant">
                                    {pct.toFixed(1)}%
                                </span>
                            </div>
                            <div className="h-1 bg-surface-container-highest rounded-full overflow-hidden">
                                <div
                                    className="h-full rounded-full bg-primary transition-all duration-500"
                                    style={{ width: `${Math.min(pct, 100)}%` }}
                                ></div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="font-label-caps text-label-caps text-on-surface-variant pt-2 border-t border-white/5 opacity-60">
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
            <div className="p-card-padding flex items-center gap-3">
                <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
                <span className="font-body-md text-on-surface-variant">Loading city stats...</span>
            </div>
        );
    }

    if (!cityStats) {
        return (
            <div className="p-card-padding font-body-md text-on-surface-variant">Failed to load city statistics.</div>
        );
    }

    return (
        <div className="p-card-padding flex flex-col gap-8 h-full">
            {/* Header */}
            <div className="flex items-center gap-3 shrink-0">
                <span className="material-symbols-outlined text-tertiary-container" style={{ fontVariationSettings: "'FILL' 1" }}>domain</span>
                <h2 className="font-headline-md text-headline-md text-on-surface">Delhi UHI Overview</h2>
            </div>

            <div className="flex-1 overflow-y-auto pr-2 pb-6">
                {/* Summary Stats */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-surface-container rounded-lg p-5 border border-white/5 flex flex-col gap-2 hover:bg-surface-container-high transition-colors">
                        <span className="font-label-caps text-label-caps text-on-surface-variant">Mean LST</span>
                        <span className="font-numeric-data text-numeric-data text-on-surface text-[24px]">{cityStats.city_mean_lst}°C</span>
                    </div>
                    <div className="bg-surface-container rounded-lg p-5 border border-white/5 flex flex-col gap-2 hover:bg-surface-container-high transition-colors">
                        <span className="font-label-caps text-label-caps text-on-surface-variant">Max LST</span>
                        <span className="font-numeric-data text-numeric-data text-error text-[24px]">{cityStats.max_lst}°C</span>
                    </div>
                    <div className="bg-surface-container rounded-lg p-5 border border-white/5 flex flex-col gap-2 hover:bg-surface-container-high transition-colors">
                        <span className="font-label-caps text-label-caps text-on-surface-variant">Min LST</span>
                        <span className="font-numeric-data text-numeric-data text-secondary text-[24px]">{cityStats.min_lst}°C</span>
                    </div>
                    <div className="bg-surface-container rounded-lg p-5 border border-white/5 flex flex-col gap-2 hover:bg-surface-container-high transition-colors">
                        <span className="font-label-caps text-label-caps text-on-surface-variant">UHI Intensity</span>
                        <span className="font-numeric-data text-numeric-data text-primary text-[24px]">{cityStats.uhi_intensity}°C</span>
                    </div>
                </div>

                {/* Category Donut */}
                <div className="my-8">
                    <DonutChart categories={cityStats.categories} />
                </div>

                {/* Top 5 Hottest */}
                <div className="flex flex-col gap-4 mt-6">
                    <div className="flex items-center gap-2">
                        <span className="material-symbols-outlined text-error text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>local_fire_department</span>
                        <h3 className="font-label-caps text-label-caps text-error uppercase tracking-widest">Top 5 Hottest Blocks</h3>
                    </div>
                    <div className="flex flex-col">
                        {cityStats.top5_hottest?.map((block, idx) => (
                            <div 
                                key={block.block_id} 
                                onClick={() => onSelectBlock && onSelectBlock(block.block_id)}
                                className="flex items-center justify-between py-3 border-b border-white/5 group hover:bg-white/[0.02] transition-colors rounded-md px-2 cursor-pointer"
                            >
                                <div className="flex items-center gap-3">
                                    <span className={`font-numeric-data text-numeric-data text-error text-[14px] ${idx > 0 ? `opacity-${100 - idx * 10}` : ''}`}>#{idx + 1}</span>
                                    <div className="flex flex-col">
                                        <span className="font-body-md text-body-md text-on-surface text-[13px]">{block.block_name}</span>
                                        <span className="font-body-md text-body-md text-on-surface-variant text-[11px]">{block.ward}</span>
                                    </div>
                                </div>
                                <div className="flex flex-col items-end">
                                    <span className={`font-numeric-data text-numeric-data text-error text-[14px] ${idx > 0 ? `opacity-${100 - idx * 10}` : ''}`}>{block.predicted_lst}°C</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Top 5 Coolest */}
                <div className="flex flex-col gap-4 mt-8">
                    <div className="flex items-center gap-2">
                        <span className="material-symbols-outlined text-secondary text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>ac_unit</span>
                        <h3 className="font-label-caps text-label-caps text-secondary uppercase tracking-widest">Top 5 Coolest Blocks</h3>
                    </div>
                    <div className="flex flex-col">
                        {cityStats.top5_coolest?.map((block, idx) => (
                            <div 
                                key={block.block_id} 
                                onClick={() => onSelectBlock && onSelectBlock(block.block_id)}
                                className="flex items-center justify-between py-3 border-b border-white/5 group hover:bg-white/[0.02] transition-colors rounded-md px-2 cursor-pointer"
                            >
                                <div className="flex items-center gap-3">
                                    <span className={`font-numeric-data text-numeric-data text-secondary text-[14px] ${idx > 0 ? `opacity-${100 - idx * 10}` : ''}`}>#{idx + 1}</span>
                                    <div className="flex flex-col">
                                        <span className="font-body-md text-body-md text-on-surface text-[13px]">{block.block_name}</span>
                                        <span className="font-body-md text-body-md text-on-surface-variant text-[11px]">{block.ward}</span>
                                    </div>
                                </div>
                                <div className="flex flex-col items-end">
                                    <span className={`font-numeric-data text-numeric-data text-secondary text-[14px] ${idx > 0 ? `opacity-${100 - idx * 10}` : ''}`}>{block.predicted_lst}°C</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Model Info Toggle */}
                <div className="mt-8 pt-4 border-t border-white/5">
                    <button
                        onClick={() => setShowModel(!showModel)}
                        className="w-full flex items-center justify-between font-label-caps text-label-caps text-on-surface-variant hover:text-on-surface transition-colors py-2"
                    >
                        <span className="uppercase tracking-widest flex items-center gap-2">
                            <span className="material-symbols-outlined text-[14px]">{showModel ? 'expand_less' : 'expand_more'}</span>
                            Model Info
                        </span>
                        <span>R²: {modelInfo?.r2?.toFixed(3)}</span>
                    </button>
                    {showModel && <ModelInfoPanel modelInfo={modelInfo} />}
                </div>
            </div>
        </div>
    );
}
