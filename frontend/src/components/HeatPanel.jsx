import React from 'react';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts';

function getRiskClass(risk) {
    switch (risk) {
        case 'Critical': return 'risk-critical';
        case 'Hot': return 'risk-hot';
        case 'Moderate': return 'risk-moderate';
        case 'Cool': return 'risk-cool';
        default: return 'risk-moderate';
    }
}

function ContributionChart({ contributions }) {
    if (!contributions || contributions.length === 0) return null;

    const data = contributions.map((c) => ({
        name: c.feature
            .replace('building_density', 'Building Density')
            .replace('green_cover', 'Green Cover')
            .replace('road_density', 'Road Density')
            .replace('avg_building_height', 'Bldg Height')
            .replace('distance_to_water', 'Dist. to Water')
            .replace('impervious_surface_fraction', 'Impervious Sfc'),
        value: c.contribution,
        direction: c.direction,
    }));

    return (
        <div className="mt-6">
            <h4 className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-widest mb-4">
                Feature Contributions
            </h4>
            <div className="bg-surface-container rounded-lg p-5 border border-white/5">
                <ResponsiveContainer width="100%" height={data.length * 32 + 20}>
                    <BarChart
                        data={data}
                        layout="vertical"
                        margin={{ top: 0, right: 10, left: 0, bottom: 0 }}
                    >
                        <XAxis type="number" hide />
                        <YAxis
                            type="category"
                            dataKey="name"
                            width={95}
                            tick={{ fontSize: 10, fill: 'var(--tw-colors-on-surface-variant)' }}
                            axisLine={false}
                            tickLine={false}
                        />
                        <Tooltip
                            contentStyle={{
                                background: 'var(--tw-colors-surface-container-high)',
                                border: '1px solid rgba(255,255,255,0.1)',
                                borderRadius: 6,
                                fontSize: 11,
                                color: 'var(--tw-colors-on-surface)',
                            }}
                            formatter={(val) => [`${val > 0 ? '+' : ''}${val.toFixed(1)}°C`, 'Impact']}
                        />
                        <ReferenceLine x={0} stroke="rgba(255,255,255,0.1)" />
                        <Bar dataKey="value" radius={[0, 3, 3, 0]} barSize={14}>
                            {data.map((entry, idx) => (
                                <Cell
                                    key={idx}
                                    fill={entry.direction === 'heating' ? 'var(--tw-colors-primary)' : 'var(--tw-colors-secondary)'}
                                    opacity={0.85}
                                />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}

export default function HeatPanel({ blockData, loading }) {
    if (loading) {
        return (
            <div className="p-card-padding animate-fade-in">
                <div className="flex items-center gap-3">
                    <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
                    <span className="font-body-md text-on-surface-variant">Loading block data...</span>
                </div>
            </div>
        );
    }

    if (!blockData) {
        return (
            <div className="p-card-padding text-center">
                <div className="text-4xl mb-3 opacity-50"><span className="material-symbols-outlined text-[48px] text-on-surface-variant">map</span></div>
                <p className="font-body-md text-on-surface-variant">
                    Click a block on the map to view its heat profile
                </p>
            </div>
        );
    }

    const {
        block_name, ward, predicted_lst, ci_lower, ci_upper, ci_width,
        feature_contributions, rank, total_blocks, risk_level, historical_context,
        features,
    } = blockData;

    const ciHalf = (ci_width / 2).toFixed(1);

    return (
        <div className="p-card-padding space-y-6 animate-fade-in overflow-y-auto max-h-[calc(100vh-160px)]">
            {/* Block Header */}
            <div>
                <div className="flex items-center justify-between mb-1">
                    <h3 className="font-headline-sm text-headline-sm text-on-surface truncate flex-1 mr-2">
                        {block_name}
                    </h3>
                    <span className={`risk-badge ${getRiskClass(risk_level)}`}>
                        {risk_level}
                    </span>
                </div>
                <p className="font-body-md text-[13px] text-on-surface-variant">{ward}</p>
            </div>

            {/* Temperature Display */}
            <div className="bg-surface-container rounded-xl p-6 text-center border border-white/10">
                <p className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-widest mb-2">
                    Surface Temperature
                </p>
                <div className="flex items-baseline justify-center gap-1">
                    <span className="font-display-lg text-display-lg text-on-surface">
                        {predicted_lst}
                    </span>
                    <span className="text-xl text-on-surface-variant font-medium">°C</span>
                    <span className="font-body-md text-[14px] text-on-surface-variant ml-2">
                        ± {ciHalf}°C
                    </span>
                </div>
                <div className="font-label-caps text-[10px] text-on-surface-variant mt-2 tracking-wider">
                    80% CI: {ci_lower}°C – {ci_upper}°C
                </div>
            </div>

            {/* Rank */}
            {rank && (
                <div className="flex items-center justify-between bg-surface-container rounded-lg px-5 py-4 border border-white/5">
                    <span className="font-label-caps text-label-caps text-on-surface-variant uppercase">Hotness Rank</span>
                    <span className="font-numeric-data text-[16px] font-semibold text-primary">
                        {rank}{rank === 1 ? 'st' : rank === 2 ? 'nd' : rank === 3 ? 'rd' : 'th'} of {total_blocks}
                    </span>
                </div>
            )}

            {/* Context */}
            {historical_context && (
                <p className="font-body-md text-[13px] text-on-surface-variant italic leading-relaxed px-1">
                    {historical_context}
                </p>
            )}

            {/* Feature Values */}
            <div>
                <h4 className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-widest mb-4">
                    Block Features
                </h4>
                <div className="grid grid-cols-2 gap-3">
                    {features && Object.entries(features).map(([key, value]) => (
                        <div key={key} className="bg-surface-container rounded-lg px-4 py-3 border border-white/5">
                            <div className="font-label-caps text-[10px] text-on-surface-variant truncate mb-1">
                                {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                            </div>
                            <div className="font-numeric-data text-[15px] font-medium text-on-surface">
                                {typeof value === 'number' ? value.toFixed(3) : value}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Contribution Chart */}
            <ContributionChart contributions={feature_contributions} />
        </div>
    );
}
