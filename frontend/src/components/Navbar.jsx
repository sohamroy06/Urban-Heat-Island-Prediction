import React from 'react';

export default function Navbar() {
    return (
        <nav className="min-h-14 bg-dark-700/95 backdrop-blur border-b border-dark-400 flex items-center justify-between gap-3 px-3 py-2 sm:px-5 lg:px-6 z-50 relative shadow-lg">
            <div className="flex items-center gap-3 min-w-0">
                <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-amber-400 to-orange-600 flex items-center justify-center text-lg shadow-lg shadow-orange-950/30 flex-shrink-0">
                    🌡️
                </div>
                <div className="min-w-0">
                    <h1 className="text-base sm:text-lg font-bold text-white tracking-tight leading-none truncate">
                        ShadowMap
                    </h1>
                    <p className="text-[9px] sm:text-[10px] text-gray-400 tracking-widest uppercase leading-none mt-1 truncate">
                        Urban Heat Island Simulator
                    </p>
                </div>
            </div>

            <div className="flex items-center gap-2 sm:gap-4 flex-shrink-0">
                <div className="hidden sm:flex items-center gap-2 text-xs text-gray-400">
                    <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                    <span>Delhi, India</span>
                </div>
                <div className="flex items-center gap-1 px-2.5 sm:px-3 py-1.5 rounded-lg bg-dark-600 border border-dark-400">
                    <span className="hidden sm:inline text-xs text-gray-300">Powered by</span>
                    <span className="text-[11px] sm:text-xs font-semibold text-amber-400 whitespace-nowrap">XGBoost</span>
                </div>
            </div>
        </nav>
    );
}
