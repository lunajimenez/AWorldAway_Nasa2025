// @ts-check
import withNuxt from "./.nuxt/eslint.config.mjs";
import antfu from "@antfu/eslint-config";

export default withNuxt(
    antfu({
        type: "app",
        typescript: true,
        vue: true,
        stylistic: {
            quotes: "double",
            indent: 4,
            semi: true,
        },
    }),
);
