<script setup lang="ts">
    import { Settings } from "lucide-vue-next";

    definePageMeta({
        title: "pages.home.title",
    });

    const { t } = useI18n({ useScope: "local" });
    const modal = useModal();

    const mountReference = useTemplateRef("MountRef");
</script>

<template>
    <main class="relative w-full h-screen overflow-hidden">
        <div ref="MountRef" class="w-full h-full" />

        <div
            class="absolute top-0 left-0 z-10 h-full w-full grid grid-cols-12 grid-rows-12 gap-4 p-4 pointer-events-none"
        >
            <div
                class="col-start-1 col-span-8 row-start-1 row-span-2 flex flex-col justify-center pointer-events-auto"
            >
                <h1 class="text-2xl font-bold mb-2">{{ t("ui.title") }}</h1>
                <p class="text-sm opacity-80">{{ t("ui.subtitle") }}</p>
            </div>

            <div
                class="col-start-9 col-span-4 row-start-1 row-span-2 pointer-events-auto flex items-start justify-end"
            >
                <Button
                    @click="
                        () => {
                            modal.loadComponent({
                                loader: () =>
                                    import('@/components/common/settings/CommonSettingsModal.vue'),
                                key: 'settings:modal',
                            });

                            modal.open.value = true;
                        }
                    "
                >
                    <Settings />
                </Button>
            </div>

            <div
                class="col-start-9 col-span-4 row-start-11 row-span-2 flex items-end justify-end pointer-events-auto"
            >
                <div class="p-4 text-xs space-y-1 text-right">
                    <p>{{ t("controls.shift") }} 🚀</p>
                    <p>{{ t("controls.space") }} 🔼</p>
                    <p>{{ t("controls.planet") }} 🌍</p>
                </div>
            </div>
        </div>
    </main>
</template>

<i18n lang="yaml">
en-US:
    ui:
        title: "3D Planet Explorer"
        subtitle: "A planet rotating in its orbit using Three.js"

    controls:
        mouse: "Click and drag to rotate view"
        zoom: "Scroll to zoom"
        wasd: "WASD to move camera"
        shift: "Shift for faster movement"
        space: "Space to go up"
        planet: "Planet rotates automatically"

es-ES:
    ui:
        title: "Explorador de Planeta 3D"
        subtitle: "Un planeta girando en su órbita usando Three.js"

    controls:
        mouse: "Click y arrastra para rotar la vista"
        zoom: "Scroll para hacer zoom"
        wasd: "WASD para mover la cámara"
        shift: "Shift para mover más rápido"
        space: "Espacio para subir"
        planet: "El planeta gira automáticamente"
</i18n>
