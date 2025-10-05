<script setup lang="ts">
    import { Pointer, RotateCcw, Settings } from "lucide-vue-next";

    definePageMeta({
        title: "pages.home.title",
    });

    const modal = useModal();

    const { isLoading, hideLoading } = useAppLoading();

    const mountReference = useTemplateRef("MountRef");
    const { init, cleanup, Camera } = useControls();

    onMounted(() => {
        nextTick(async () => {
            init(mountReference.value);

            await hideLoading();
        });
    });

    onUnmounted(() => {
        cleanup(mountReference.value);
    });
</script>

<template>
    <main class="relative w-full h-screen overflow-hidden">
        <CommonHomeLoadingBanner v-model="isLoading" />

        <div ref="MountRef" class="w-full h-full" />

        <div
            class="absolute top-0 left-0 z-10 h-full w-full grid grid-cols-12 grid-rows-12 gap-4 p-4 pointer-events-none"
        >
            <div
                class="col-start-1 col-span-8 row-start-1 row-span-2 flex flex-col justify-center pointer-events-auto"
            >
                <h1 class="text-2xl font-bold mb-2">{{ $t("pages.home.ui.title") }}</h1>
                <p class="text-sm opacity-80">{{ $t("pages.home.ui.subtitle") }}</p>
            </div>

            <div
                class="col-start-9 col-span-4 row-start-1 row-span-2 flex flex-col justify-center items-end pointer-events-auto"
            >
                <Button
                    class="font-bold"
                    @click="
                        () => {
                            modal.loadComponent({
                                loader: () =>
                                    import(
                                        '~/components/common/settings/modal/CommonSettingsModelModal.vue'
                                    ),
                                key: 'settings:model',
                            });

                            modal.open.value = true;
                        }
                    "
                >
                    {{ $t("pages.home.ui.try_out") }}
                </Button>
            </div>

            <div
                class="col-start-1 col-span-4 row-start-12 row-span-1 flex items-end space-x-2 pointer-events-auto"
            >
                <Button @click="Camera.reset()">
                    <Pointer />
                    <RotateCcw />
                </Button>
                <Button
                    @click="
                        () => {
                            modal.loadComponent({
                                loader: () =>
                                    import(
                                        '~/components/common/settings/modal/CommonSettingsModal.vue'
                                    ),
                                key: 'settings:modal',
                            });

                            modal.open.value = true;
                        }
                    "
                >
                    <Settings />
                </Button>

                <Button variant="link" as-child>
                    <NuxtLink to="/about"> {{ $t("pages.about.title") }} </NuxtLink>
                </Button>
            </div>

            <div
                class="col-start-9 col-span-4 row-start-11 row-span-2 flex items-end justify-end pointer-events-auto"
            >
                <div class="text-xs space-y-1 text-right">
                    <p>{{ $t("pages.home.controls.space") }} 🔼</p>
                    <p>{{ $t("pages.home.controls.planet") }} 🌍</p>
                </div>
            </div>
        </div>
    </main>
</template>
