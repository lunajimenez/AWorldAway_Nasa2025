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
    ],
    shadcn: {
        prefix: "",
        componentDir: "~/components/ui",
    },
    colorMode: {
        classSuffix: "",
    },
    eslint: {
        config: {
            standalone: false,
        },
    },
    veeValidate: {
        autoImports: false,
    },
});
