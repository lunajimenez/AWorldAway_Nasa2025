import antfu from "@antfu/eslint-config";
// @ts-check
import withNuxt from "./.nuxt/eslint.config.mjs";

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
    {
        files: ["**/*.vue"],
        rules: {
            "@stylistic/indent": ["error", 4],
            "unicorn/number-literal-case": "off",
        },
    },
);
