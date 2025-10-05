<script setup lang="ts">
    import type { Locale } from "#i18n";
    import type { ClassValue } from "class-variance-authority/types";
    import type { AcceptableValue } from "reka-ui";
    import { Languages } from "lucide-vue-next";
    import { cn } from "~/lib/utils";

    defineOptions({
        inheritAttrs: false,
    });

    const props = withDefaults(defineProps<Props>(), {
        behavior: "change-locale",
        modelValue: undefined,
    });

    const emit = defineEmits<{ "update:modelValue": [value: Locale] }>();

    interface Props {
        behavior?: "change-locale" | "select";
        modelValue?: Locale;
    }

    const { locales, locale } = useI18n();

    const router = useRouter();
    const switchLocalePath = useSwitchLocalePath();
    function handleChange(locale: AcceptableValue) {
        const targetLocale = locale as Locale;

        if (props.behavior === "change-locale") {
            const targetPath = switchLocalePath(targetLocale);
            router.push(targetPath);

            return;
        }

        emit("update:modelValue", targetLocale);
    }
</script>

<template>
    <Select :default-value="locale" :model-value="modelValue" @update:model-value="handleChange">
        <SelectTrigger :class="cn('h-10 w-32', $attrs.class as ClassValue)">
            <slot name="trigger">
                <Languages class="size-4" />
                <SelectValue />
            </slot>
        </SelectTrigger>
        <SelectContent>
            <template v-for="_locale in locales" :key="_locale.code">
                <SelectItem :value="_locale.code">
                    {{ _locale.language }}
                </SelectItem>
            </template>
        </SelectContent>
    </Select>
</template>
