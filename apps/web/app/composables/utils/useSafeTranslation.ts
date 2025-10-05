export default function () {
    const { t, te } = useI18n();

    function safeT(key: string | undefined, fallback?: string): string {
        if (!key) {
            return fallback || key || "";
        }

        if (te(key)) {
            return t(key);
        }

        return fallback || key || "";
    }

    return {
        safeT,
    };
}
