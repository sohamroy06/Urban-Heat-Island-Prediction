import React from 'react';

export default function Navbar() {
    return (
        <nav className="h-14 bg-dark-700 border-b border-dark-400 flex items-center justify-between px-6 z-50 relative">
            <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center text-lg">
                    🌡️
                </div>
                <div>
                    <h1 className="text-lg font-bold text-white tracking-tight leading-none">
                        ShadowMap
                    </h1>
                    <p className="text-[10px] text-gray-400 tracking-widest uppercase leading-none mt-0.5">
                        Urban Heat Island Simulator
                    </p>
                </div>
            </div>

            <div className="flex items-center gap-4">
                <div className="hidden md:flex items-center gap-2 text-xs text-gray-400">
                    <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                    <span>Delhi, India</span>
                </div>
                <div className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-dark-600 border border-dark-400">
                    <span className="text-xs text-gray-300">Powered by</span>
                    <span className="text-xs font-semibold text-amber-400">GBR Model</span>
                </div>
            </div>
        </nav>
    );
}
