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
        <div className="mt-4">
            <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                Feature Contributions
            </h4>
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
                        tick={{ fontSize: 10, fill: '#94a3b8' }}
                        axisLine={false}
                        tickLine={false}
                    />
                    <Tooltip
                        contentStyle={{
                            background: '#1a1d27',
                            border: '1px solid #363c50',
                            borderRadius: 6,
                            fontSize: 11,
                            color: '#e2e8f0',
                        }}
                        formatter={(val) => [`${val > 0 ? '+' : ''}${val.toFixed(1)}°C`, 'Impact']}
                    />
                    <ReferenceLine x={0} stroke="#363c50" />
                    <Bar dataKey="value" radius={[0, 3, 3, 0]} barSize={14}>
                        {data.map((entry, idx) => (
                            <Cell
                                key={idx}
                                fill={entry.direction === 'heating' ? '#f97316' : '#06b6d4'}
                                opacity={0.85}
                            />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
}

export default function HeatPanel({ blockData, loading }) {
    if (loading) {
        return (
            <div className="p-4 animate-fade-in">
                <div className="flex items-center gap-3">
                    <div className="w-5 h-5 border-2 border-amber-500 border-t-transparent rounded-full animate-spin"></div>
                    <span className="text-sm text-gray-400">Loading block data...</span>
                </div>
            </div>
        );
    }

    if (!blockData) {
        return (
            <div className="p-4 text-center">
                <div className="text-4xl mb-3 opacity-50">🗺️</div>
                <p className="text-sm text-gray-400">
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
        <div className="p-4 space-y-4 animate-fade-in overflow-y-auto max-h-[calc(100vh-160px)]">
            {/* Block Header */}
            <div>
                <div className="flex items-center justify-between mb-1">
                    <h3 className="text-sm font-semibold text-white truncate flex-1 mr-2">
                        {block_name}
                    </h3>
                    <span className={`risk-badge ${getRiskClass(risk_level)}`}>
                        {risk_level}
                    </span>
                </div>
                <p className="text-xs text-gray-500">{ward}</p>
            </div>

            {/* Temperature Display */}
            <div className="bg-dark-600 rounded-xl p-4 text-center border border-dark-400">
                <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">
                    Surface Temperature
                </p>
                <div className="flex items-baseline justify-center gap-1">
                    <span className="text-4xl font-bold text-white">
                        {predicted_lst}
                    </span>
                    <span className="text-lg text-gray-400">°C</span>
                    <span className="text-sm text-gray-500 ml-1">
                        ± {ciHalf}°C
                    </span>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                    80% CI: {ci_lower}°C – {ci_upper}°C
                </div>
            </div>

            {/* Rank */}
            {rank && (
                <div className="flex items-center justify-between bg-dark-600 rounded-lg px-3 py-2 border border-dark-400">
                    <span className="text-xs text-gray-400">Hotness Rank</span>
                    <span className="text-sm font-semibold text-amber-400">
                        {rank}{rank === 1 ? 'st' : rank === 2 ? 'nd' : rank === 3 ? 'rd' : 'th'} of {total_blocks}
                    </span>
                </div>
            )}

            {/* Context */}
            {historical_context && (
                <p className="text-xs text-gray-400 italic leading-relaxed px-1">
                    {historical_context}
                </p>
            )}

            {/* Feature Values */}
            <div>
                <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                    Block Features
                </h4>
                <div className="grid grid-cols-2 gap-1.5">
                    {features && Object.entries(features).map(([key, value]) => (
                        <div key={key} className="bg-dark-600 rounded px-2 py-1.5 border border-dark-400">
                            <div className="text-[10px] text-gray-500 truncate">
                                {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                            </div>
                            <div className="text-xs font-mono font-medium text-gray-200">
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
