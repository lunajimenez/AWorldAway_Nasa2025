import process from "node:process";
import tailwindcss from "@tailwindcss/vite";

export default defineNuxtConfig({
    compatibilityDate: "2025-07-15",
    devtools: { enabled: true },
    css: ["~/assets/css/tailwind.css"],
    vite: {
        plugins: [tailwindcss()],
        server: {
            fs: {
                strict: true,
            },
        },
        build: {
            sourcemap: false,
            minify: "terser",
            rollupOptions: {
                output: {
                    manualChunks: {
                        three: ["three"],
                    },
                },
            },
        },
    },
    imports: {
        dirs: ["composables/**", "./shared/utils/**"],
        imports: [{ from: "vue-sonner", name: "toast" }],
    },
    nitro: {
        imports: {
            dirs: ["./shared/utils/**"],
        },
        preset: "node-server",
    },
    runtimeConfig: {
        groqApiKey: process.env.API_GROP,
        public: {
            apiBase: process.env.API_BASE_URL,
        },
    },
    modules: [
        "@nuxt/eslint",
        "@nuxt/image",
        "shadcn-nuxt",
        "@nuxtjs/color-mode",
        "@vueuse/nuxt",
        "@vee-validate/nuxt",
        "@nuxtjs/i18n",
        "@nuxt/fonts",
    ],
    shadcn: {
        prefix: "",
        componentDir: "~/components/ui",
    },
    colorMode: {
        classSuffix: "",
        preference: "dark",
        fallback: "dark",
        hid: "nuxt-color-mode-script",
        globalName: "__NUXT_COLOR_MODE__",
        storageKey: "nuxt-color-mode",
    },
    eslint: {
        config: {
            standalone: false,
        },
    },
    veeValidate: {
        autoImports: false,
    },
    i18n: {
        baseUrl: process.env.NUXT_PUBLIC_SITE_URL,
        skipSettingLocaleOnNavigate: false,
        detectBrowserLanguage: {
            useCookie: true,
            redirectOn: "root",
            fallbackLocale: "en-US",
        },
        defaultLocale: "en-US",
        strategy: "prefix",
        locales: [
            {
                code: "en-US",
                language: "English",
                file: "en-US.json",
            },
            {
                code: "es-ES",
                language: "Español",
                file: "es-ES.json",
            },
        ],
    },
    fonts: {
        families: [
            {
                name: "Pixelify Sans",
                provider: "local",
            },
        ],
        assets: {
            prefix: "/_fonts/",
        },
    },
});
