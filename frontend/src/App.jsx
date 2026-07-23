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
    const [activePage, setActivePage] = useState('simulator');
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
    }, [handleSelectBlock]);

    if (error) {
        return (
            <div className="h-screen bg-surface flex items-center justify-center">
                <div className="max-w-md text-center p-8">
                    <div className="text-6xl mb-4">⚠️</div>
                    <h2 className="text-xl font-bold text-on-surface mb-2">Connection Error</h2>
                    <p className="text-on-surface-variant text-sm mb-4">{error}</p>
                    <div className="bg-surface-container rounded-lg p-4 text-left border border-white/5">
                        <p className="text-xs text-on-surface-variant font-mono mb-1">Start the backend:</p>
                        <code className="text-xs text-primary font-mono">
                            cd backend && uvicorn main:app --reload --port 8000
                        </code>
                    </div>
                    <button
                        onClick={() => window.location.reload()}
                        className="mt-4 px-4 py-2 bg-primary hover:bg-primary/90 text-on-primary rounded-lg text-sm font-semibold transition-colors"
                    >
                        Retry Connection
                    </button>
                </div>
            </div>
        );
    }

    if (activePage !== 'simulator') {
        return (
            <div className="h-screen w-screen flex flex-col bg-surface text-on-surface font-body-md text-body-md overflow-hidden antialiased selection:bg-primary-container selection:text-on-primary-container">
                <Navbar activePage={activePage} onNavigate={setActivePage} />
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                        <h2 className="font-headline-md text-headline-md text-on-surface mb-2 capitalize">{activePage}</h2>
                        <p className="text-on-surface-variant font-body-md text-body-md">This page is under construction.</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="h-screen w-screen flex flex-col bg-surface text-on-surface font-body-md text-body-md overflow-hidden antialiased selection:bg-primary-container selection:text-on-primary-container">
            <Navbar activePage={activePage} onNavigate={setActivePage} />

            <main className="flex-1 flex overflow-hidden w-full relative">
                {/* Left Panel — City Stats */}
                <aside className="hidden md:flex w-[24rem] lg:w-[26rem] bg-surface-container-low border-r border-white/5 flex-col z-10 shadow-[20px_0_40px_rgba(0,0,0,0.2)] shrink-0 overflow-y-auto">
                    <CityStats onSelectBlock={handleCityStatsSelect} />
                </aside>

                {/* Center Panel — Map */}
                <section className="flex-1 relative bg-surface-container-lowest overflow-hidden">
                    <MapView
                        geojsonData={geojsonData}
                        selectedBlockId={selectedBlockId}
                        onSelectBlock={handleSelectBlock}
                        flyToCenter={flyToCenter}
                    />
                </section>

                {/* Right Panel — Block Info / What-If */}
                <aside className="hidden md:flex w-[26rem] lg:w-[28rem] bg-surface-container-low border-l border-white/5 flex-col z-10 shadow-[-20px_0_40px_rgba(0,0,0,0.2)] shrink-0 overflow-hidden">
                    {/* Tab Buttons */}
                    <div className="flex border-b border-white/5 w-full bg-surface-container-lowest shrink-0">
                        <button
                            onClick={() => setActiveTab('info')}
                            className={`flex-1 py-4 flex justify-center items-center gap-2 font-label-caps text-label-caps uppercase transition-colors ${activeTab === 'info'
                                    ? 'text-primary border-b-2 border-primary bg-primary/5'
                                    : 'text-on-surface-variant hover:text-on-surface hover:bg-white/5'
                                }`}
                        >
                            <span className="material-symbols-outlined text-[16px]">info</span> Block Info
                        </button>
                        <button
                            onClick={() => setActiveTab('whatif')}
                            className={`flex-1 py-4 flex justify-center items-center gap-2 font-label-caps text-label-caps uppercase transition-colors ${activeTab === 'whatif'
                                    ? 'text-secondary border-b-2 border-secondary bg-secondary/5'
                                    : 'text-on-surface-variant hover:text-on-surface hover:bg-white/5'
                                }`}
                        >
                            <span className="material-symbols-outlined text-[16px]">science</span> What-If
                        </button>
                    </div>

                    {/* Tab Content */}
                    <div className="flex-1 overflow-y-auto">
                        {activeTab === 'info' ? (
                            <HeatPanel blockData={blockData} loading={blockLoading} />
                        ) : (
                            <WhatIfPanel blockData={blockData} />
                        )}
                    </div>
                </aside>
            </main>
        </div>
    );
}
