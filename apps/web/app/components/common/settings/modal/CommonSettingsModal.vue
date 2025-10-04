<script setup lang="ts">
    import { Settings, Globe } from "lucide-vue-next";

    const { t } = useI18n({ useScope: "local" });
</script>

<template>
    <SheetContent>
        <SheetHeader class="space-y-3 pb-6 border-b">
            <div class="flex items-center gap-3">
                <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                    <Settings class="size-5 text-primary" />
                </div>
                <div class="flex flex-col items-start">
                    <SheetTitle class="text-2xl">{{ t("title") }}</SheetTitle>
                    <SheetDescription class="text-left">{{ t("description") }}</SheetDescription>
                </div>
            </div>
        </SheetHeader>
        <div class="px-4 space-y-6">
            <div class="flex flex-col space-y-2">
                <div class="flex items-center space-x-2">
                    <Globe class="size-5 text-muted-foreground" />
                    <Label class="text-base font-semibold">{{ t("setting.lang.title") }}</Label>
                </div>

                <CommonSettingsLocale class="w-1/2 flex justify-center" />
            </div>
        </div>
    </SheetContent>
</template>

<i18n lang="yaml">
en-US:
    title: "Settings"
    description: "Personalize your experience"
    setting:
        lang:
            title: "Language"

es-ES:
    title: "Configuracion"
    description: "Personaliza tu experiencia"
    setting:
        lang:
            title: "Idioma"
</i18n>
