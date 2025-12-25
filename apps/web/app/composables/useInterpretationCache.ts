interface InterpretationCache {
    [key: string]: {
        interpretation: string;
        timestamp: number;
    };
}

const CACHE_KEY = 'exoplanet-interpretations';
const CACHE_EXPIRY = 24 * 60 * 60 * 1000; // 24 hours

export function useInterpretationCache() {
    const getCache = (): InterpretationCache => {
        if (import.meta.server) return {};
        const cached = localStorage.getItem(CACHE_KEY);
        if (!cached) return {};
        try {
            return JSON.parse(cached);
        } catch {
            return {};
        }
    };

    const generateKey = (inputData: Record<string, string>): string => {
        const sortedData = Object.keys(inputData)
            .sort()
            .map(key => `${key}:${inputData[key]}`)
            .join('|');
        return btoa(sortedData).slice(0, 32);
    };

    const getInterpretation = (inputData: Record<string, string>): string | null => {
        const cache = getCache();
        const key = generateKey(inputData);
        const entry = cache[key];

        if (!entry) return null;

        // Check if expired
        if (Date.now() - entry.timestamp > CACHE_EXPIRY) {
            delete cache[key];
            localStorage.setItem(CACHE_KEY, JSON.stringify(cache));
            return null;
        }

        return entry.interpretation;
    };

    const saveInterpretation = (inputData: Record<string, string>, interpretation: string): void => {
        if (import.meta.server) return;
        const cache = getCache();
        const key = generateKey(inputData);

        cache[key] = {
            interpretation,
            timestamp: Date.now(),
        };

        localStorage.setItem(CACHE_KEY, JSON.stringify(cache));
    };

    const clearCache = (): void => {
        if (import.meta.server) return;
        localStorage.removeItem(CACHE_KEY);
    };

    const clearInterpretation = (inputData: Record<string, string>): void => {
        if (import.meta.server) return;
        const cache = getCache();
        const key = generateKey(inputData);

        if (cache[key]) {
            delete cache[key];
            localStorage.setItem(CACHE_KEY, JSON.stringify(cache));
        }
    };

    return {
        getInterpretation,
        saveInterpretation,
        clearCache,
        clearInterpretation,
    };
}
