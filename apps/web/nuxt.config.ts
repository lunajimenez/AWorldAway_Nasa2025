import tailwindcss from "@tailwindcss/vite";
import process from "node:process";

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
    },
    modules: [
        "@nuxt/eslint",
        "@nuxt/image",
        "shadcn-nuxt",
        "@nuxtjs/color-mode",
        "@vueuse/nuxt",
        "@vee-validate/nuxt",
        "@nuxtjs/i18n",
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
        baseUrl: process.env.BASE_URL,
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
});
