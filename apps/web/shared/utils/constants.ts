export const CONSTANTS = {
    COMPOSABLES: {
        USE_MODAL: {
            STORAGE_LENGTH: 3,
        },
    },
    HOME: {
        STARS_COUNT: 1000,
        MIN_LOADING_TIME: 1500,
        MOON_COUNT: 3,
    },
} as const;

export const MOON_COLOR_SCHEMES = [
    {
        base: ["#4a4a4a", "#6b6b6b", "#3a3a3a", "#5a5a5a"],
        land: ["#2a2a2a", "#3d3d3d", "#505050"],
    },
    {
        base: ["#8b4513", "#a0522d", "#cd853f", "#d2691e"],
        land: ["#654321", "#8b7355", "#a0826d"],
        poles: "rgba(255, 255, 255, 0.6)",
    },
    {
        base: ["#1e3a8a", "#2563eb", "#1e40af", "#3b82f6"],
        land: ["#0f172a", "#1e293b", "#334155"],
        poles: "rgba(200, 220, 255, 0.8)",
    },
    {
        base: ["#eab308", "#fbbf24", "#d97706", "#f59e0b"],
        land: ["#92400e", "#b45309", "#c2410c"],
    },
    {
        base: ["#166534", "#15803d", "#16a34a", "#22c55e"],
        land: ["#064e3b", "#065f46", "#047857"],
    },
];
