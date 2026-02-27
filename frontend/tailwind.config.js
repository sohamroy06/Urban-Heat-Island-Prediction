/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                dark: {
                    900: '#0f1117',
                    800: '#14161e',
                    700: '#1a1d27',
                    600: '#222633',
                    500: '#2a2f3f',
                    400: '#363c50',
                },
                heat: {
                    critical: '#ef4444',
                    hot: '#f97316',
                    moderate: '#eab308',
                    cool: '#06b6d4',
                },
                accent: {
                    amber: '#f59e0b',
                    orange: '#f97316',
                    teal: '#14b8a6',
                    cyan: '#06b6d4',
                },
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
                mono: ['JetBrains Mono', 'monospace'],
            },
        },
    },
    plugins: [],
};
