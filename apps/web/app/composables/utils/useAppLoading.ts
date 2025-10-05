export default function () {
    const hasLoadedOnce = useState("app:has-loaded", () => false);
    const isLoading = useState("app:is-loading", () => !hasLoadedOnce.value);

    const hideLoading = async (minTime: number = CONSTANTS.HOME.MIN_LOADING_TIME) => {
        if (hasLoadedOnce.value) {
            isLoading.value = false;
            return;
        }

        const startTime = Date.now();
        const elapsed = Date.now() - startTime;
        const remaining = Math.max(0, minTime - elapsed);

        // eslint-disable-next-line style/arrow-parens
        await new Promise((resolve) => setTimeout(resolve, remaining));

        isLoading.value = false;
        hasLoadedOnce.value = true;
    };

    return {
        isLoading,
        hideLoading,
        hasLoadedOnce,
    };
}
