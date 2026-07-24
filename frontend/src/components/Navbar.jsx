import React from 'react';
import { motion } from 'framer-motion';
import { ThermoIcon } from './Icons';

export default function Navbar() {
    return (
        <nav className="h-14 flex-shrink-0 bg-ink-900 border-b border-ink-600 flex items-center justify-between px-4 sm:px-6 z-50 relative">
            <div className="flex items-center gap-3">
                <motion.div
                    initial={{ rotate: -8, scale: 0.85 }}
                    animate={{ rotate: 0, scale: 1 }}
                    transition={{ type: 'spring', stiffness: 200, damping: 14 }}
                    className="w-8 h-8 rounded-lg bg-gradient-to-br from-signal-500 to-signal-600 flex items-center justify-center text-ink-950"
                >
                    <ThermoIcon width={17} height={17} strokeWidth={2} />
                </motion.div>
                <div>
                    <h1 className="font-display text-lg font-semibold text-ink-100 tracking-tight leading-none">
                        ShadowMap
                    </h1>
                    <p className="text-[10px] text-ink-300 tracking-[0.15em] uppercase leading-none mt-1">
                        Urban Heat Island Simulator
                    </p>
                </div>
            </div>

            <div className="flex items-center gap-3 sm:gap-4">
                <div className="hidden md:flex items-center gap-2 text-xs text-ink-300">
                    <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60" />
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
                    </span>
                    <span>Delhi, India</span>
                </div>
                <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-ink-800 border border-ink-600">
                    <span className="text-[11px] text-ink-300">Model</span>
                    <span className="text-[11px] font-semibold text-signal-300">XGBoost + Quantile</span>
                </div>
            </div>
        </nav>
    );
}
