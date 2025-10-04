<script setup lang="ts">
    const { t } = useI18n({ useScope: "local" });
    const mountReference = useTemplateRef("MountRef");
</script>

<template>
    <Head>
        <Title>{{ t("meta.title") }}</Title>
    </Head>
    <main class="relative w-full h-screen overflow-hidden">
        <div ref="MountRef" class="w-full h-full" />

        <div
            class="absolute top-0 left-0 z-10 h-full w-full grid grid-cols-12 grid-rows-12 gap-4 p-4 pointer-events-none"
        >
            <div class="col-span-8 row-span-2 flex flex-col justify-center pointer-events-auto">
                <h1 class="text-2xl font-bold mb-2">{{ t("ui.title") }}</h1>
                <p class="text-sm opacity-80">{{ t("ui.subtitle") }}</p>
            </div>
        </div>
    </main>
</template>

<i18n lang="yaml">
en-US:
    meta:
        title: AWorldAway_Nasa2025

    ui:
        title: "3D Planet Explorer"
        subtitle: "A planet rotating in its orbit using Three.js"

es-ES:
    meta:
        title: AWorldAway_Nasa2025

    ui:
        title: "Explorador de Planeta 3D"
        subtitle: "Un planeta girando en su órbita usando Three.js"
</i18n>
