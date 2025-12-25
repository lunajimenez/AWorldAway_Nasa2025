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
            class="absolute top-0 left-0 z-10 h-full w-full flex flex-col justify-between p-3 sm:p-4 pointer-events-none"
        >
            <!-- Header Section -->
            <div class="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
                <div class="pointer-events-auto">
                    <h1 class="text-lg sm:text-2xl font-bold mb-1 sm:mb-2">{{ $t("pages.home.ui.title") }}</h1>
                    <p class="text-xs sm:text-sm opacity-80">{{ $t("pages.home.ui.subtitle") }}</p>
                </div>

                <div class="pointer-events-auto">
                    <Button
                        class="font-bold text-sm sm:text-base"
                        size="sm"
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
            </div>

            <!-- Footer Section -->
            <div class="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-3">
                <div class="flex flex-wrap items-center gap-2 pointer-events-auto">
                    <Button size="sm" @click="Camera.reset()">
                        <Pointer class="w-4 h-4" />
                        <RotateCcw class="w-4 h-4" />
                    </Button>

                    <CommonSettingsLocale />

                    <Button variant="link" as-child size="sm" class="text-xs sm:text-sm">
                        <NuxtLink :to="$localeRoute({ path: '/about' })">
                            {{ $t("pages.about.title") }}
                        </NuxtLink>
                    </Button>

                    <Button variant="link" as-child size="sm" class="text-xs sm:text-sm">
                        <NuxtLink :to="$localeRoute({ path: '/metrics' })">
                            {{ $t("pages.metrics.title") }}
                        </NuxtLink>
                    </Button>
                </div>

                <div class="text-xs space-y-1 text-left sm:text-right pointer-events-auto hidden sm:block">
                    <p>{{ $t("pages.home.controls.space") }} 🔼</p>
                    <p>{{ $t("pages.home.controls.planet") }} 🌍</p>
                </div>
            </div>
        </div>
    </main>
</template>
