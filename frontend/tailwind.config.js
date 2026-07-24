/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                // Warm near-black neutrals — deliberately not the ubiquitous
                // blue-slate "AI dashboard" background.
                ink: {
                    950: '#0b0a09',
                    900: '#131110',
                    800: '#1c1917',
                    700: '#262220',
                    600: '#332d2a',
                    500: '#443c37',
                    400: '#665a52',
                    300: '#8c7d72',
                    200: '#b5a89d',
                    100: '#e8e0d8',
                },
                // Signature interactive accent — a cool violet against the
                // warm neutrals, kept out of the thermal ramp on purpose so
                // "chrome" and "data" never get visually confused.
                signal: {
                    300: '#a5b4fc',
                    400: '#818cf8',
                    500: '#6366f1',
                    600: '#4f46e5',
                },
                // The actual LST color ramp (mirrors MapView.jsx getColor),
                // reserved exclusively for temperature data.
                thermal: {
                    coolest: '#0ea5e9',
                    cooler: '#22d3ee',
                    cool: '#a3e635',
                    mild: '#eab308',
                    warm: '#f59e0b',
                    hot: '#f97316',
                    hotter: '#dc2626',
                    hottest: '#991b1b',
                    extreme: '#7f1d1d',
                },
            },
            fontFamily: {
                display: ['"Fraunces"', 'ui-serif', 'Georgia', 'serif'],
                sans: ['"Public Sans"', 'system-ui', 'sans-serif'],
                mono: ['"IBM Plex Mono"', 'monospace'],
            },
        },
    },
    plugins: [],
};
