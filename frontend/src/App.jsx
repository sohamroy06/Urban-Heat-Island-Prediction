import React, { useState, useEffect, useCallback } from 'react';
import Navbar from './components/Navbar';
import MapView from './components/MapView';
import CityStats from './components/CityStats';
import HeatPanel from './components/HeatPanel';
import WhatIfPanel from './components/WhatIfPanel';
import { fetchBlocks, fetchBlock } from './api/shadowmap';

export default function App() {
    const [geojsonData, setGeojsonData] = useState(null);
    const [selectedBlockId, setSelectedBlockId] = useState(null);
    const [blockData, setBlockData] = useState(null);
    const [blockLoading, setBlockLoading] = useState(false);
    const [activeTab, setActiveTab] = useState('info');
    const [mobilePanel, setMobilePanel] = useState('stats');
    const [flyToCenter, setFlyToCenter] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        async function loadBlocks() {
            try {
                const data = await fetchBlocks();
                setGeojsonData(data);
            } catch (err) {
                console.error('Failed to load blocks:', err);
                setError('Failed to connect to the ShadowMap API. Make sure the backend is running on port 8000.');
            }
        }
        loadBlocks();
    }, []);

    const handleSelectBlock = useCallback(async (blockId) => {
        if (blockId === selectedBlockId) return;
        setSelectedBlockId(blockId);
        setBlockLoading(true);
        setBlockData(null);

        try {
            const data = await fetchBlock(blockId);
            setBlockData(data);

            if (data.lat && data.lon) {
                setFlyToCenter([data.lat, data.lon]);
            }
        } catch (err) {
            console.error('Failed to fetch block:', err);
        } finally {
            setBlockLoading(false);
        }
    }, [selectedBlockId]);

    const handleCityStatsSelect = useCallback((blockId) => {
        handleSelectBlock(blockId);
        setActiveTab('info');
        setMobilePanel('details');
    }, [handleSelectBlock]);

    if (error) {
        return (
            <div className="min-h-screen bg-dark-900 flex items-center justify-center p-4">
                <div className="w-full max-w-md text-center p-6 sm:p-8 rounded-2xl border border-dark-400 bg-dark-700 shadow-2xl">
                    <div className="text-6xl mb-4">⚠️</div>
                    <h2 className="text-xl font-bold text-white mb-2">Connection Error</h2>
                    <p className="text-gray-400 text-sm mb-4">{error}</p>
                    <div className="bg-dark-700 rounded-lg p-4 text-left">
                        <p className="text-xs text-gray-500 font-mono mb-1">Start the backend:</p>
                        <code className="text-xs text-amber-400 font-mono">
                            cd backend && uvicorn main:app --reload --port 8000
                        </code>
                    </div>
                    <button
                        onClick={() => window.location.reload()}
                        className="mt-4 px-4 py-2 bg-amber-500 hover:bg-amber-600 text-dark-900 rounded-lg text-sm font-semibold transition-colors"
                    >
                        Retry Connection
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen lg:h-screen flex flex-col bg-dark-900 text-gray-100 lg:overflow-hidden">
            <Navbar />

            <div className="flex-1 min-h-0 lg:flex lg:overflow-hidden">
                {/* Left Panel — City Stats (30%) */}
                <aside className="hidden lg:block w-[320px] xl:w-[340px] bg-dark-700/95 border-r border-dark-400 flex-shrink-0 overflow-hidden">
                    <CityStats onSelectBlock={handleCityStatsSelect} />
                </aside>

                {/* Center Panel — Map (flexible) */}
                <main className="relative h-[54vh] min-h-[360px] lg:h-auto lg:flex-1 lg:min-h-0 overflow-hidden border-b border-dark-400 lg:border-b-0">
                    <MapView
                        geojsonData={geojsonData}
                        selectedBlockId={selectedBlockId}
                        onSelectBlock={handleSelectBlock}
                        flyToCenter={flyToCenter}
                    />
                </main>

                {/* Right Panel — Block Info / What-If (20%) */}
                <aside className="hidden lg:flex w-[360px] xl:w-[390px] bg-dark-700/95 border-l border-dark-400 flex-shrink-0 flex-col overflow-hidden">
                    {/* Tab Buttons */}
                    <div className="flex border-b border-dark-400">
                        <button
                            onClick={() => setActiveTab('info')}
                            className={`flex-1 py-2.5 text-xs font-semibold uppercase tracking-wider transition-colors ${activeTab === 'info'
                                    ? 'text-amber-400 border-b-2 border-amber-400 bg-dark-600'
                                    : 'text-gray-500 hover:text-gray-300'
                                }`}
                        >
                            🌡️ Block Info
                        </button>
                        <button
                            onClick={() => setActiveTab('whatif')}
                            className={`flex-1 py-2.5 text-xs font-semibold uppercase tracking-wider transition-colors ${activeTab === 'whatif'
                                    ? 'text-teal-400 border-b-2 border-teal-400 bg-dark-600'
                                    : 'text-gray-500 hover:text-gray-300'
                                }`}
                        >
                            🔬 What-If
                        </button>
                    </div>

                    {/* Tab Content */}
                    <div className="flex-1 overflow-hidden">
                        {activeTab === 'info' ? (
                            <HeatPanel blockData={blockData} loading={blockLoading} />
                        ) : (
                            <WhatIfPanel blockData={blockData} />
                        )}
                    </div>
                </aside>

                {/* Mobile / Tablet Panels */}
                <section className="lg:hidden bg-dark-800 px-3 py-3 sm:px-4">
                    <div className="sticky top-0 z-20 -mx-3 sm:-mx-4 px-3 sm:px-4 pb-3 bg-dark-800/95 backdrop-blur border-b border-dark-400">
                        <div className="grid grid-cols-3 gap-2 rounded-xl bg-dark-700 p-1 border border-dark-400 shadow-lg">
                            <button
                                onClick={() => setMobilePanel('stats')}
                                className={`mobile-tab ${mobilePanel === 'stats' ? 'mobile-tab-active' : ''}`}
                            >
                                City
                            </button>
                            <button
                                onClick={() => {
                                    setMobilePanel('details');
                                    setActiveTab('info');
                                }}
                                className={`mobile-tab ${mobilePanel === 'details' ? 'mobile-tab-active' : ''}`}
                            >
                                Block
                            </button>
                            <button
                                onClick={() => {
                                    setMobilePanel('simulate');
                                    setActiveTab('whatif');
                                }}
                                className={`mobile-tab ${mobilePanel === 'simulate' ? 'mobile-tab-active' : ''}`}
                            >
                                Sim
                            </button>
                        </div>
                    </div>

                    <div className="pt-3">
                        {mobilePanel === 'stats' && (
                            <div className="panel-shell">
                                <CityStats onSelectBlock={handleCityStatsSelect} />
                            </div>
                        )}
                        {mobilePanel === 'details' && (
                            <div className="panel-shell">
                                <HeatPanel blockData={blockData} loading={blockLoading} />
                            </div>
                        )}
                        {mobilePanel === 'simulate' && (
                            <div className="panel-shell">
                                <WhatIfPanel blockData={blockData} />
                            </div>
                        )}
                    </div>
                </section>
            </div>
        </div>
    );
}
