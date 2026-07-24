import React from 'react';
import { motion } from 'framer-motion';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts';
import AnimatedNumber from './AnimatedNumber';
import { MapPinIcon } from './Icons';

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
            <h4 className="text-[10px] font-semibold text-ink-400 uppercase tracking-wider mb-2">
                Feature Contributions
            </h4>
            <ResponsiveContainer width="100%" height={data.length * 30 + 20}>
                <BarChart data={data} layout="vertical" margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
                    <XAxis type="number" hide />
                    <YAxis
                        type="category"
                        dataKey="name"
                        width={95}
                        tick={{ fontSize: 10, fill: '#a5a29c' }}
                        axisLine={false}
                        tickLine={false}
                    />
                    <Tooltip
                        contentStyle={{ background: '#1c1917', border: '1px solid #332d2a', borderRadius: 8, fontSize: 11, color: '#e8e0d8' }}
                        formatter={(val) => [`${val > 0 ? '+' : ''}${val.toFixed(1)}°C`, 'Impact']}
                    />
                    <ReferenceLine x={0} stroke="#443c37" />
                    <Bar dataKey="value" radius={[0, 3, 3, 0]} barSize={12} animationDuration={500}>
                        {data.map((entry, idx) => (
                            <Cell key={idx} fill={entry.direction === 'heating' ? '#f97316' : '#38bdf8'} opacity={0.9} />
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
            <div className="p-4">
                <div className="flex items-center gap-3">
                    <div className="w-4 h-4 border-2 border-signal-400 border-t-transparent rounded-full animate-spin" />
                    <span className="text-sm text-ink-300">Loading block data…</span>
                </div>
            </div>
        );
    }

    if (!blockData) {
        return (
            <div className="p-6 text-center flex flex-col items-center gap-3">
                <MapPinIcon width={30} height={30} className="text-ink-500" />
                <p className="text-sm text-ink-400 max-w-[200px]">
                    Click a block on the map — or search a ward — to view its heat profile
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
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2 }}
            className="p-4 space-y-4 lg:overflow-y-auto lg:max-h-[calc(100vh-160px)]"
        >
            <div>
                <div className="flex items-center justify-between mb-1 gap-2">
                    <h3 className="font-display text-base font-semibold text-ink-100 truncate flex-1">
                        {block_name}
                    </h3>
                    <span className={`risk-badge ${getRiskClass(risk_level)}`}>{risk_level}</span>
                </div>
                <p className="text-xs text-ink-400">{ward}</p>
            </div>

            <div className="rounded-2xl border border-ink-600 bg-gradient-to-br from-ink-800 to-ink-900 p-4 text-center">
                <p className="text-[10px] text-ink-400 uppercase tracking-wider mb-1">Surface Temperature</p>
                <div className="flex items-baseline justify-center gap-1">
                    <AnimatedNumber value={predicted_lst} decimals={1} className="font-display text-4xl font-semibold text-ink-100" />
                    <span className="text-lg text-ink-400">°C</span>
                    <span className="text-sm text-ink-500 ml-1">± {ciHalf}°C</span>
                </div>
                <div className="text-xs text-ink-400 mt-1 font-mono">
                    80% CI: {ci_lower}°C – {ci_upper}°C
                </div>
            </div>

            {rank && (
                <div className="flex items-center justify-between bg-ink-800 rounded-lg px-3 py-2 border border-ink-600">
                    <span className="text-xs text-ink-300">Hotness Rank</span>
                    <span className="text-sm font-semibold text-signal-300 font-mono">
                        {rank} / {total_blocks}
                    </span>
                </div>
            )}

            {historical_context && (
                <p className="text-xs text-ink-300 italic leading-relaxed px-0.5">{historical_context}</p>
            )}

            <div>
                <h4 className="text-[10px] font-semibold text-ink-400 uppercase tracking-wider mb-2">Block Features</h4>
                <div className="grid grid-cols-2 gap-1.5">
                    {features && Object.entries(features).map(([key, value]) => (
                        <div key={key} className="bg-ink-800 rounded-lg px-2 py-1.5 border border-ink-600">
                            <div className="text-[9px] text-ink-500 truncate">
                                {key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                            </div>
                            <div className="text-xs font-mono font-medium text-ink-100">
                                {typeof value === 'number' ? value.toFixed(3) : value}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <ContributionChart contributions={feature_contributions} />
        </motion.div>
    );
}
