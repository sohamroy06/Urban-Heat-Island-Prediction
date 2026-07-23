import React, { useEffect, useRef, useMemo } from 'react';
import { MapContainer, TileLayer, GeoJSON, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

function getColor(lst) {
    if (lst >= 44) return '#991b1b';
    if (lst >= 40) return '#dc2626';
    if (lst >= 38) return '#ea580c';
    if (lst >= 36) return '#f59e0b';
    if (lst >= 34) return '#eab308';
    if (lst >= 32) return '#84cc16';
    if (lst >= 30) return '#06b6d4';
    return '#0ea5e9';
}

function Legend() {
    const map = useMap();

    useEffect(() => {
        const L = window.L || require('leaflet');
        const legend = L.control({ position: 'bottomleft' });

        legend.onAdd = function () {
            const div = L.DomUtil.create('div', 'legend');
            div.style.cssText = `
                background: rgba(30, 32, 36, 0.9);
                padding: 20px;
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                color: var(--tw-colors-on-surface, #e2e2e8);
                font-family: Inter, sans-serif;
                backdrop-filter: blur(20px);
                box-shadow: 0 20px 40px rgba(0,0,0,0.4);
                width: 192px;
                margin-left: 24px;
                margin-bottom: 24px;
            `;

            const grades = [28, 30, 32, 34, 36, 38, 40, 44];
            const labels = ['< 30°C', '30°C', '32°C', '34°C', '36°C', '38°C', '40°C', '44°C+'];

            let html = '<span style="display:block;font-size:11px;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:16px;">Surface Temp (°C)</span>';
            html += '<div style="display:flex;flex-direction:column;gap:10px;">';
            for (let i = 0; i < grades.length; i++) {
                html +=
                    `<div style="display:flex;align-items:center;gap:12px;">` +
                    `<div style="width:16px;height:12px;border-radius:2px;background:${getColor(grades[i])};"></div>` +
                    `<span style="font-size:12px;color:var(--tw-colors-on-surface-variant, #d8c3ad);">${labels[i]}</span>` +
                    `</div>`;
            }
            html += '</div>';
            div.innerHTML = html;
            return div;
        };

        legend.addTo(map);
        return () => legend.remove();
    }, [map]);

    return null;
}

function MapUpdater({ center }) {
    const map = useMap();
    useEffect(() => {
        if (center) {
            map.flyTo(center, 13, { duration: 0.8 });
        }
    }, [center, map]);
    return null;
}

export default function MapView({ geojsonData, selectedBlockId, onSelectBlock, flyToCenter }) {
    const geoJsonRef = useRef(null);

    const delhiCenter = [28.63, 77.22];
    const defaultZoom = 11;

    const style = useMemo(() => {
        return (feature) => {
            const lst = feature.properties.predicted_lst || feature.properties.lst || 35;
            const isSelected = feature.properties.block_id === selectedBlockId;
            return {
                fillColor: getColor(lst),
                weight: isSelected ? 2.5 : 0.8,
                opacity: 1,
                color: isSelected ? '#ffc174' : 'rgba(255,255,255,0.1)',
                fillOpacity: isSelected ? 0.9 : 0.7,
            };
        };
    }, [selectedBlockId]);

    const onEachFeature = useMemo(() => {
        return (feature, layer) => {
            const props = feature.properties;
            const lst = props.predicted_lst || props.lst || 'N/A';
            const ci_lower = props.ci_lower || '';
            const ci_upper = props.ci_upper || '';

            const tooltipContent = `
        <div style="font-family:Inter,sans-serif;padding:4px;">
          <div style="font-weight:600;font-size:12px;color:var(--tw-colors-on-surface,#e2e2e8);margin-bottom:3px;">
            ${props.block_name || props.block_id}
          </div>
          <div style="font-size:18px;font-weight:600;color:${getColor(lst)};">
            ${lst}°C
          </div>
          ${ci_lower ? `<div style="font-size:10px;color:var(--tw-colors-on-surface-variant,#d8c3ad);">CI: ${ci_lower}°C – ${ci_upper}°C</div>` : ''}
        </div>
      `;

            layer.bindTooltip(tooltipContent, {
                sticky: true,
                className: 'custom-tooltip',
                direction: 'top',
            });

            layer.on('click', () => {
                if (onSelectBlock) {
                    onSelectBlock(props.block_id);
                }
            });
        };
    }, [onSelectBlock]);

    const geoJsonKey = useMemo(() => {
        return selectedBlockId ? `geo-${selectedBlockId}` : 'geo-default';
    }, [selectedBlockId]);

    return (
        <div className="h-full w-full relative">
            <MapContainer
                center={delhiCenter}
                zoom={defaultZoom}
                className="h-full w-full"
                zoomControl={true}
                attributionControl={false}
            >
                <TileLayer
                    url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>'
                    maxZoom={19}
                />

                {geojsonData && (
                    <GeoJSON
                        key={geoJsonKey}
                        ref={geoJsonRef}
                        data={geojsonData}
                        style={style}
                        onEachFeature={onEachFeature}
                    />
                )}

                <Legend />
                {flyToCenter && <MapUpdater center={flyToCenter} />}
            </MapContainer>

            {/* Selected Block Info Floating Panel */}
            {selectedBlockId && (
                <div className="absolute top-6 left-20 bg-surface-container/90 border border-white/10 rounded-lg p-4 backdrop-blur-xl shadow-[0_20px_40px_rgba(0,0,0,0.4)] min-w-[200px] z-[1000] pointer-events-none transition-all">
                    <span className="font-label-caps text-label-caps text-primary uppercase tracking-widest block mb-1">Delhi Heat Index</span>
                    <span className="font-headline-sm text-headline-sm text-on-surface font-semibold block">Selected {selectedBlockId}</span>
                </div>
            )}

            {!geojsonData && (
                <div className="absolute inset-0 flex items-center justify-center bg-surface-container-lowest/80 z-[1000]">
                    <div className="text-center animate-fade-in">
                        <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                        <p className="font-body-md text-on-surface-variant">Loading map data...</p>
                    </div>
                </div>
            )}
        </div>
    );
}
