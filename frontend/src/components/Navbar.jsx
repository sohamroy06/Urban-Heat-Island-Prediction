import React from 'react';

export default function Navbar({ activePage = 'simulator', onNavigate }) {
    const navLinks = [
        { id: 'dashboard', label: 'Dashboard' },
        { id: 'analytics', label: 'Analytics' },
        { id: 'simulator', label: 'Simulator' },
        { id: 'reports', label: 'Reports' },
    ];

    return (
        <header className="bg-surface dark:bg-surface border-b border-white/10 flex justify-between items-center h-20 px-8 w-full z-50 shrink-0">
            {/* Brand & Logo */}
            <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary to-primary-container flex items-center justify-center shadow-[0_0_20px_rgba(245,158,11,0.2)]">
                    <span className="material-symbols-outlined text-surface-lowest text-2xl" style={{ fontVariationSettings: "'FILL' 1" }}>thermostat</span>
                </div>
                <div className="flex flex-col">
                    <span className="font-display-lg text-display-lg font-bold text-on-surface tracking-wider text-[22px] leading-tight">ShadowMap</span>
                    <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-widest text-[9px]">Urban Heat Island Simulator</span>
                </div>
            </div>

            {/* Navigation Links */}
            <nav className="hidden md:flex items-center gap-8 h-full">
                {navLinks.map(link => (
                    <a
                        key={link.id}
                        href="#"
                        onClick={(e) => { e.preventDefault(); onNavigate && onNavigate(link.id); }}
                        className={`font-label-caps text-label-caps uppercase h-full flex items-center px-2 transition-all duration-200 ${
                            activePage === link.id
                                ? 'text-primary border-b-2 border-primary hover:bg-white/5 relative top-[1px]'
                                : 'text-on-surface-variant hover:text-on-surface hover:bg-white/5'
                        }`}
                    >
                        {link.label}
                    </a>
                ))}
            </nav>

            {/* Trailing Actions */}
            <div className="flex items-center gap-6">
                <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-secondary shadow-[0_0_10px_rgba(78,222,163,0.5)]"></div>
                    <span className="font-label-caps text-label-caps text-on-surface-variant">Delhi, India</span>
                </div>
                <div className="h-4 w-px bg-white/10"></div>
                <div className="flex items-center gap-2 border border-white/10 rounded-full px-3 py-1.5 bg-surface-container-lowest">
                    <span className="font-label-caps text-label-caps text-on-surface-variant">Powered by</span>
                    <span className="font-label-caps text-label-caps text-primary">XGBoost</span>
                </div>
                <div className="flex items-center gap-3 border-l border-white/10 pl-6">
                    <button className="text-on-surface-variant hover:text-on-surface transition-colors">
                        <span className="material-symbols-outlined">notifications</span>
                    </button>
                    <button className="text-on-surface-variant hover:text-on-surface transition-colors">
                        <span className="material-symbols-outlined">settings</span>
                    </button>
                </div>
            </div>
        </header>
    );
}
