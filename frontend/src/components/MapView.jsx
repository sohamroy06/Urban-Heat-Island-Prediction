import React, { useEffect, useRef, useMemo } from 'react';
import { MapContainer, TileLayer, GeoJSON, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

export function getColor(lst) {
    if (lst >= 44) return '#7f1d1d';
    if (lst >= 42) return '#991b1b';
    if (lst >= 40) return '#dc2626';
    if (lst >= 38) return '#f97316';
    if (lst >= 36) return '#f59e0b';
    if (lst >= 34) return '#eab308';
    if (lst >= 32) return '#a3e635';
    if (lst >= 30) return '#22d3ee';
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
        background: rgba(28, 25, 23, 0.95);
        padding: 12px 14px;
        border-radius: 10px;
        border: 1px solid #332d2a;
        color: #e8e0d8;
        font-family: 'Public Sans', sans-serif;
        font-size: 11px;
        line-height: 1.6;
        backdrop-filter: blur(8px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
      `;

            const grades = [28, 30, 32, 34, 36, 38, 40, 42, 44];
            const labels = ['<30°C', '30°C', '32°C', '34°C', '36°C', '38°C', '40°C', '42°C', '44°C+'];

            let html = '<div style="font-weight:600;margin-bottom:6px;color:#a5b4fc;">Surface Temp (°C)</div>';
            for (let i = 0; i < grades.length; i++) {
                html +=
                    `<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">` +
                    `<span style="display:inline-block;width:14px;height:10px;border-radius:2px;background:${getColor(grades[i])};"></span>` +
                    `<span>${labels[i]}</span>` +
                    `</div>`;
            }
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

export default function MapView({ geojsonData, selectedBlockId, onSelectBlock, flyToCenter, refreshToken }) {
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
                color: isSelected ? '#a5b4fc' : '#332d2a',
                fillOpacity: isSelected ? 0.9 : 0.7,
            };
        };
    }, [selectedBlockId]);

    // react-leaflet's <GeoJSON> mounts once and does not re-diff `data`
    // (see App.jsx / MapView history: a `key`-based remount-per-click used
    // to live here and was the source of dropped/late click events). It
    // *does* reactively re-apply the `style` prop to existing layers, which
    // is all selection highlighting needs. City-wide what-if mutates the
    // same geojsonData object's feature properties in place and bumps
    // refreshToken so the already-mounted layers get restyled without a
    // remount.
    useEffect(() => {
        if (geoJsonRef.current) {
            geoJsonRef.current.setStyle(style);
        }
    }, [refreshToken, style]);

    const onEachFeature = useMemo(() => {
        return (feature, layer) => {
            const props = feature.properties;

            const renderTooltip = () => {
                const lst = props.predicted_lst || props.lst || 'N/A';
                const ci_lower = props.ci_lower || '';
                const ci_upper = props.ci_upper || '';
                return `
          <div style="font-family:'Public Sans',sans-serif;">
            <div style="font-weight:600;font-size:12px;margin-bottom:3px;">
              ${props.block_name || props.block_id}
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:18px;font-weight:600;color:${getColor(lst)};">
              ${lst}°C
            </div>
            ${ci_lower ? `<div style="font-size:10px;color:#8c7d72;">CI: ${ci_lower}°C – ${ci_upper}°C</div>` : ''}
          </div>
        `;
            };

            layer.bindTooltip(renderTooltip(), {
                sticky: true,
                className: 'custom-tooltip',
                direction: 'top',
            });

            layer.on('tooltipopen', () => layer.setTooltipContent(renderTooltip()));

            layer.on('click', () => {
                if (onSelectBlock) {
                    onSelectBlock(props.block_id);
                }
            });
        };
    }, [onSelectBlock]);

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
                        ref={geoJsonRef}
                        data={geojsonData}
                        style={style}
                        onEachFeature={onEachFeature}
                    />
                )}

                <Legend />
                {flyToCenter && <MapUpdater center={flyToCenter} />}
            </MapContainer>

            {!geojsonData && (
                <div className="absolute inset-0 flex items-center justify-center bg-ink-950/85 z-[1000]">
                    <div className="text-center">
                        <div className="w-10 h-10 border-2 border-signal-400 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                        <p className="text-ink-300 text-sm">Loading map data…</p>
                    </div>
                </div>
            )}
        </div>
    );
}
