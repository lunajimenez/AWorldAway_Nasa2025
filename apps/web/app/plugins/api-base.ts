export default defineNuxtPlugin(() => {
    const {
        public: { apiBase },
    } = useRuntimeConfig();

    const fetchRoot = $fetch.create({ baseURL: apiBase });
    const fetchApi = $fetch.create({ baseURL: `${apiBase}/api` });

    return {
        provide: {
            fetchRoot,
            fetchApi,
        },
    };
});
