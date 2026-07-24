import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Navbar from './components/Navbar';
import MapView from './components/MapView';
import CityStats from './components/CityStats';
import HeatPanel from './components/HeatPanel';
import WhatIfPanel from './components/WhatIfPanel';
import { ThermoIcon, FlaskIcon, SkylineIcon, AlertIcon } from './components/Icons';
import { fetchBlocks, fetchBlock } from './api/shadowmap';

export default function App() {
    const [geojsonData, setGeojsonData] = useState(null);
    const [selectedBlockId, setSelectedBlockId] = useState(null);
    const [blockData, setBlockData] = useState(null);
    const [blockLoading, setBlockLoading] = useState(false);
    const [activeTab, setActiveTab] = useState('info');
    const [flyToCenter, setFlyToCenter] = useState(null);
    const [error, setError] = useState(null);
    const [mapRefreshToken, setMapRefreshToken] = useState(0);
    const [mobilePanel, setMobilePanel] = useState('stats');

    const selectRequestRef = useRef(0);
    const baselineRef = useRef(new Map());

    useEffect(() => {
        async function loadBlocks() {
            try {
                const data = await fetchBlocks();
                setGeojsonData(data);
                const baseline = new Map();
                for (const feature of data.features || []) {
                    const p = feature.properties;
                    baseline.set(p.block_id, {
                        predicted_lst: p.predicted_lst,
                        ci_lower: p.ci_lower,
                        ci_upper: p.ci_upper,
                        ci_width: p.ci_width,
                    });
                }
                baselineRef.current = baseline;
            } catch (err) {
                console.error('Failed to load blocks:', err);
                setError('Failed to connect to the ShadowMap API. Make sure the backend is running on port 8000.');
            }
        }
        loadBlocks();
    }, []);

    const handleSelectBlock = useCallback(async (blockId) => {
        if (blockId === selectedBlockId) return;

        const requestId = ++selectRequestRef.current;
        setSelectedBlockId(blockId);
        setBlockLoading(true);
        setBlockData(null);

        try {
            const data = await fetchBlock(blockId);
            if (requestId !== selectRequestRef.current) return; // a newer selection landed first
            setBlockData(data);

            if (data.lat && data.lon) {
                setFlyToCenter([data.lat, data.lon]);
            }
        } catch (err) {
            console.error('Failed to fetch block:', err);
        } finally {
            if (requestId === selectRequestRef.current) setBlockLoading(false);
        }
    }, [selectedBlockId]);

    const handleCityStatsSelect = useCallback((blockId) => {
        handleSelectBlock(blockId);
        setActiveTab('info');
        setMobilePanel('details');
    }, [handleSelectBlock]);

    // City-wide what-if mutates the *same* geojsonData object's feature
    // properties in place (react-leaflet doesn't re-diff `data`) and bumps
    // mapRefreshToken so MapView restyles the already-mounted layers.
    const applyCityWideDeltas = useCallback((blockDeltas) => {
        if (!geojsonData) return;
        const byId = new Map(blockDeltas.map((b) => [b.block_id, b]));
        for (const feature of geojsonData.features) {
            const update = byId.get(feature.properties.block_id);
            if (update) {
                feature.properties.predicted_lst = update.predicted_lst;
            }
        }
        setMapRefreshToken((t) => t + 1);
    }, [geojsonData]);

    const resetCityWide = useCallback(() => {
        if (!geojsonData) return;
        for (const feature of geojsonData.features) {
            const base = baselineRef.current.get(feature.properties.block_id);
            if (base) Object.assign(feature.properties, base);
        }
        setMapRefreshToken((t) => t + 1);
    }, [geojsonData]);

    if (error) {
        return (
            <div className="h-screen bg-ink-950 flex items-center justify-center p-4">
                <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="max-w-md w-full text-center p-8 rounded-2xl border border-ink-600 bg-ink-800"
                >
                    <AlertIcon width={40} height={40} className="mx-auto mb-4 text-red-400" />
                    <h2 className="font-display text-xl font-semibold text-ink-100 mb-2">Connection Error</h2>
                    <p className="text-ink-300 text-sm mb-4">{error}</p>
                    <div className="bg-ink-900 rounded-lg p-4 text-left">
                        <p className="text-xs text-ink-400 font-mono mb-1">Start the backend:</p>
                        <code className="text-xs text-signal-300 font-mono">
                            cd backend && uvicorn main:app --reload --port 8000
                        </code>
                    </div>
                    <button
                        onClick={() => window.location.reload()}
                        className="mt-4 px-4 py-2 bg-signal-500 hover:bg-signal-600 text-white rounded-lg text-sm font-semibold transition-colors"
                    >
                        Retry Connection
                    </button>
                </motion.div>
            </div>
        );
    }

    return (
        <div className="min-h-screen lg:h-screen flex flex-col bg-ink-950 text-ink-100 lg:overflow-hidden">
            <Navbar />

            <div className="flex-1 min-h-0 lg:flex lg:overflow-hidden">
                {/* Left Panel — City Stats */}
                <aside className="hidden lg:block w-[300px] xl:w-[320px] bg-ink-900 border-r border-ink-600 flex-shrink-0 overflow-hidden">
                    <CityStats geojsonData={geojsonData} onSelectBlock={handleCityStatsSelect} />
                </aside>

                {/* Center Panel — Map */}
                <main className="relative h-[52vh] min-h-[340px] lg:h-auto lg:flex-1 lg:min-h-0 overflow-hidden border-b border-ink-600 lg:border-b-0">
                    <MapView
                        geojsonData={geojsonData}
                        selectedBlockId={selectedBlockId}
                        onSelectBlock={handleSelectBlock}
                        flyToCenter={flyToCenter}
                        refreshToken={mapRefreshToken}
                    />
                </main>

                {/* Right Panel — Block Info / What-If */}
                <aside className="hidden lg:flex w-[330px] xl:w-[360px] bg-ink-900 border-l border-ink-600 flex-shrink-0 flex-col overflow-hidden">
                    <TabBar activeTab={activeTab} setActiveTab={setActiveTab} />
                    <div className="flex-1 overflow-hidden">
                        <AnimatePresence>
                            {activeTab === 'info' ? (
                                <motion.div
                                    key="info"
                                    initial={{ opacity: 0, x: 8 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -8 }}
                                    transition={{ duration: 0.16 }}
                                    className="h-full"
                                >
                                    <HeatPanel blockData={blockData} loading={blockLoading} />
                                </motion.div>
                            ) : (
                                <motion.div
                                    key="whatif"
                                    initial={{ opacity: 0, x: 8 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -8 }}
                                    transition={{ duration: 0.16 }}
                                    className="h-full"
                                >
                                    <WhatIfPanel
                                        blockData={blockData}
                                        onCityWideResult={applyCityWideDeltas}
                                        onCityWideReset={resetCityWide}
                                    />
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </aside>

                {/* Mobile / Tablet Panels */}
                <section className="lg:hidden bg-ink-950 px-3 py-3 sm:px-4">
                    <div className="sticky top-0 z-20 -mx-3 sm:-mx-4 px-3 sm:px-4 pb-3 bg-ink-950/95 backdrop-blur border-b border-ink-600">
                        <div className="grid grid-cols-3 gap-2 rounded-xl bg-ink-800 p-1 border border-ink-600">
                            <MobileTabButton icon={SkylineIcon} label="City" active={mobilePanel === 'stats'} onClick={() => setMobilePanel('stats')} />
                            <MobileTabButton icon={ThermoIcon} label="Block" active={mobilePanel === 'details'} onClick={() => { setMobilePanel('details'); setActiveTab('info'); }} />
                            <MobileTabButton icon={FlaskIcon} label="Sim" active={mobilePanel === 'simulate'} onClick={() => { setMobilePanel('simulate'); setActiveTab('whatif'); }} />
                        </div>
                    </div>

                    <div className="pt-3">
                        {mobilePanel === 'stats' && <CityStats geojsonData={geojsonData} onSelectBlock={handleCityStatsSelect} />}
                        {mobilePanel === 'details' && <HeatPanel blockData={blockData} loading={blockLoading} />}
                        {mobilePanel === 'simulate' && (
                            <WhatIfPanel
                                blockData={blockData}
                                onCityWideResult={applyCityWideDeltas}
                                onCityWideReset={resetCityWide}
                            />
                        )}
                    </div>
                </section>
            </div>
        </div>
    );
}

function TabBar({ activeTab, setActiveTab }) {
    return (
        <div className="flex border-b border-ink-600 flex-shrink-0">
            <TabButton active={activeTab === 'info'} onClick={() => setActiveTab('info')} icon={ThermoIcon} label="Block Info" />
            <TabButton active={activeTab === 'whatif'} onClick={() => setActiveTab('whatif')} icon={FlaskIcon} label="What-If" />
        </div>
    );
}

function TabButton({ active, onClick, icon: Icon, label }) {
    return (
        <button
            onClick={onClick}
            className={`relative flex-1 py-3 text-xs font-semibold uppercase tracking-wider transition-colors flex items-center justify-center gap-1.5 ${active ? 'text-signal-300' : 'text-ink-400 hover:text-ink-200'
                }`}
        >
            <Icon width={14} height={14} />
            {label}
            {active && (
                <motion.div layoutId="tab-underline" className="absolute bottom-0 left-0 right-0 h-[2px] bg-signal-400" />
            )}
        </button>
    );
}

function MobileTabButton({ icon: Icon, label, active, onClick }) {
    return (
        <button
            onClick={onClick}
            className={`relative flex flex-col items-center gap-1 py-2 rounded-lg text-[11px] font-semibold transition-colors ${active ? 'text-signal-300' : 'text-ink-400'
                }`}
        >
            {active && (
                <motion.div layoutId="mobile-tab-active" className="absolute inset-0 bg-ink-700 rounded-lg -z-10" />
            )}
            <Icon width={16} height={16} />
            {label}
        </button>
    );
}
