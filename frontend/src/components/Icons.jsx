import React from 'react';

const base = {
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.75,
    strokeLinecap: 'round',
    strokeLinejoin: 'round',
};

export function ThermoIcon(props) {
    return (
        <svg {...base} {...props}>
            <path d="M12 14.5V5a2 2 0 1 0-4 0v9.5a4 4 0 1 0 4 0Z" />
            <path d="M10 8h1.5" />
        </svg>
    );
}

export function BuildingIcon(props) {
    return (
        <svg {...base} {...props}>
            <rect x="5" y="3" width="9" height="18" rx="0.5" />
            <path d="M14 21h5V9l-5-3" />
            <path d="M8 7h1M8 11h1M8 15h1M11 7h1M11 11h1M11 15h1" />
        </svg>
    );
}

export function TreeIcon(props) {
    return (
        <svg {...base} {...props}>
            <path d="M12 3 7 10h2.5L6 15h4v6h4v-6h4l-3.5-5H17Z" />
        </svg>
    );
}

export function SunIcon(props) {
    return (
        <svg {...base} {...props}>
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2.5M12 19.5V22M4.2 4.2l1.8 1.8M18 18l1.8 1.8M2 12h2.5M19.5 12H22M4.2 19.8 6 18M18 6l1.8-1.8" />
        </svg>
    );
}

export function FlameIcon(props) {
    return (
        <svg {...base} {...props}>
            <path d="M12 2c1 3-2.5 4-2.5 7.5A2.5 2.5 0 0 0 12 12a2 2 0 0 0 2-2c1.5 1 2.5 3 2.5 5a4.5 4.5 0 1 1-9 0C7.5 10 12 8 12 2Z" />
        </svg>
    );
}

export function DropletIcon(props) {
    return (
        <svg {...base} {...props}>
            <path d="M12 3c3 4 6 7.5 6 11a6 6 0 0 1-12 0c0-3.5 3-7 6-11Z" />
        </svg>
    );
}

export function MapPinIcon(props) {
    return (
        <svg {...base} {...props}>
            <path d="M12 21s7-6.5 7-12a7 7 0 1 0-14 0c0 5.5 7 12 7 12Z" />
            <circle cx="12" cy="9" r="2.25" />
        </svg>
    );
}

export function FlaskIcon(props) {
    return (
        <svg {...base} {...props}>
            <path d="M9 2h6M10 2v6.5L4.8 18a2 2 0 0 0 1.7 3h11a2 2 0 0 0 1.7-3L14 8.5V2" />
            <path d="M7.5 15h9" />
        </svg>
    );
}

export function SkylineIcon(props) {
    return (
        <svg {...base} {...props}>
            <path d="M3 21V10l4-3 4 3v11M11 21V6l4-3 4 3v15" />
            <path d="M3 21h18" />
        </svg>
    );
}

export function SearchIcon(props) {
    return (
        <svg {...base} {...props}>
            <circle cx="11" cy="11" r="6.5" />
            <path d="m20 20-4-4" />
        </svg>
    );
}

export function ChevronIcon({ open, ...props }) {
    return (
        <svg
            {...base}
            style={{ transform: open ? 'rotate(90deg)' : 'none', transition: 'transform 200ms ease' }}
            {...props}
        >
            <path d="m9 6 6 6-6 6" />
        </svg>
    );
}

export function SparklesIcon(props) {
    return (
        <svg {...base} {...props}>
            <path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2 2M16 16l2 2M18 6l-2 2M8 16l-2 2" />
            <circle cx="12" cy="12" r="2.25" />
        </svg>
    );
}

export function AlertIcon(props) {
    return (
        <svg {...base} {...props}>
            <path d="M12 3 2 20h20L12 3Z" />
            <path d="M12 9v5M12 17h.01" />
        </svg>
    );
}
