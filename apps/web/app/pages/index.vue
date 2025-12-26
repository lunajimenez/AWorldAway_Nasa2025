<script setup lang="ts">
    import { Pointer, RotateCcw, Rocket, Globe2, ChevronRight, Menu, X, Volume2, VolumeX } from "lucide-vue-next";

    definePageMeta({
        title: "pages.home.title",
    });

    const modal = useModal();

    const { isLoading, hideLoading } = useAppLoading();

    const mountReference = useTemplateRef("MountRef");
    const { init, cleanup, Camera } = useControls();
    
    // Space ambient audio
    const spaceAudio = useSpaceAudio();

    // Mobile menu state
    const isMobileMenuOpen = ref(false);

    onMounted(() => {
        nextTick(async () => {
            init(mountReference.value);
            
            // Initialize space audio
            spaceAudio.init();

            await hideLoading();
        });
    });

    onUnmounted(() => {
        cleanup(mountReference.value);
        spaceAudio.cleanup();
    });
</script>

<template>
    <main class="relative w-full h-screen h-[100dvh] overflow-hidden">
        <CommonHomeLoadingBanner v-model="isLoading" />

        <!-- Three.js Canvas Mount -->
        <div ref="MountRef" class="w-full h-full" />

        <!-- Nebula Overlay for extra depth -->
        <div class="nebula-overlay" />

        <!-- Main UI Overlay -->
        <div
            class="absolute top-0 left-0 z-10 h-full w-full flex flex-col justify-between p-3 xs:p-4 sm:p-6 lg:p-8 pointer-events-none"
        >
            <!-- Header Section -->
            <header class="flex flex-col gap-3 sm:gap-4">
                <!-- Top Row: Brand + CTA -->
                <div class="flex items-start justify-between gap-2 sm:gap-4">
                    <!-- Brand + Title - Compact on mobile -->
                    <div class="pointer-events-auto glass rounded-xl sm:rounded-2xl px-3 py-2.5 sm:px-5 sm:py-4 flex-1 max-w-[280px] sm:max-w-md">
                        <div class="flex items-center gap-2 sm:gap-3 mb-1 sm:mb-2">
                            <div class="relative flex-shrink-0">
                                <div class="absolute inset-0 bg-indigo-500/30 blur-lg sm:blur-xl rounded-full animate-pulse-glow" />
                                <Globe2 class="w-6 h-6 sm:w-8 sm:h-8 lg:w-10 lg:h-10 text-indigo-400 relative z-10" />
                            </div>
                            <h1 class="text-base sm:text-xl lg:text-2xl xl:text-3xl font-bold tracking-tight text-white text-glow leading-tight">
                                {{ $t("pages.home.ui.title") }}
                            </h1>
                        </div>
                        <p class="text-xs sm:text-sm lg:text-base text-white/70 leading-relaxed line-clamp-2 sm:line-clamp-none">
                            {{ $t("pages.home.ui.subtitle") }}
                        </p>
                    </div>

                    <!-- CTA Button - Responsive sizing -->
                    <div class="pointer-events-auto flex-shrink-0">
                        <button
                            class="btn-cta-primary flex items-center gap-2 sm:gap-3 group !px-3 !py-2.5 sm:!px-6 sm:!py-3.5 lg:!px-8 lg:!py-4"
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
                            <Rocket class="w-4 h-4 sm:w-5 sm:h-5 transition-transform group-hover:translate-x-0.5" />
                            <span class="font-semibold text-sm sm:text-base lg:text-lg whitespace-nowrap">
                                {{ $t("pages.home.ui.try_out") }}
                            </span>
                            <ChevronRight class="w-4 h-4 sm:w-5 sm:h-5 opacity-60 transition-transform group-hover:translate-x-1 hidden sm:block" />
                        </button>
                    </div>
                </div>
            </header>

            <!-- Footer Section - Cockpit Style Navigation -->
            <footer class="flex flex-col gap-3 sm:gap-4">
                <!-- Mobile: Compact floating bar -->
                <div class="flex sm:hidden items-center justify-between gap-2 pointer-events-auto">
                    <!-- Compact Nav Toggle -->
                    <nav class="nav-cockpit px-2.5 py-2 flex items-center gap-2">
                        <!-- Camera Reset -->
                        <button
                            class="flex items-center justify-center w-9 h-9 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 transition-all duration-200 text-white/80 hover:text-white"
                            @click="Camera.reset()"
                        >
                            <RotateCcw class="w-4 h-4" />
                        </button>

                        <!-- Audio Mute Toggle -->
                        <button
                            class="flex items-center justify-center w-9 h-9 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 transition-all duration-200"
                            :class="spaceAudio.isMuted.value ? 'text-red-400/80' : 'text-white/80'"
                            @click="spaceAudio.toggleMute()"
                        >
                            <VolumeX v-if="spaceAudio.isMuted.value" class="w-4 h-4" />
                            <Volume2 v-else class="w-4 h-4" />
                        </button>

                        <!-- Language Selector -->
                        <CommonSettingsLocale />

                        <!-- Menu Toggle -->
                        <button
                            class="flex items-center justify-center w-9 h-9 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 transition-all duration-200 text-white/80 hover:text-white"
                            @click="isMobileMenuOpen = !isMobileMenuOpen"
                        >
                            <Menu v-if="!isMobileMenuOpen" class="w-4 h-4" />
                            <X v-else class="w-4 h-4" />
                        </button>
                    </nav>

                    <!-- Mobile Menu Dropdown -->
                    <Transition
                        enter-active-class="transition-all duration-200 ease-out"
                        enter-from-class="opacity-0 translate-y-2"
                        enter-to-class="opacity-100 translate-y-0"
                        leave-active-class="transition-all duration-150 ease-in"
                        leave-from-class="opacity-100 translate-y-0"
                        leave-to-class="opacity-0 translate-y-2"
                    >
                        <div
                            v-if="isMobileMenuOpen"
                            class="absolute bottom-16 left-3 right-3 nav-cockpit px-3 py-3 flex flex-col gap-2"
                        >
                            <NuxtLink
                                :to="$localeRoute({ path: '/about' })"
                                class="px-4 py-3 rounded-lg text-sm font-medium text-white/80 hover:text-white bg-white/5 hover:bg-white/10 transition-all duration-200 text-center"
                                @click="isMobileMenuOpen = false"
                            >
                                {{ $t("pages.about.title") }}
                            </NuxtLink>

                            <NuxtLink
                                :to="$localeRoute({ path: '/metrics' })"
                                class="px-4 py-3 rounded-lg text-sm font-medium text-white/80 hover:text-white bg-white/5 hover:bg-white/10 transition-all duration-200 text-center"
                                @click="isMobileMenuOpen = false"
                            >
                                {{ $t("pages.metrics.title") }}
                            </NuxtLink>
                        </div>
                    </Transition>
                </div>

                <!-- Desktop/Tablet: Full navigation bar -->
                <div class="hidden sm:flex sm:items-end sm:justify-between gap-4">
                    <!-- Navigation Controls -->
                    <nav class="nav-cockpit px-3 py-2.5 sm:px-4 sm:py-3 flex flex-wrap items-center gap-2 sm:gap-3 pointer-events-auto">
                        <!-- Camera Reset -->
                        <button
                            class="flex items-center gap-2 px-2.5 py-1.5 sm:px-3 sm:py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 transition-all duration-200 text-white/80 hover:text-white"
                            @click="Camera.reset()"
                        >
                            <Pointer class="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                            <RotateCcw class="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                        </button>

                        <!-- Audio Mute Toggle -->
                        <button
                            class="flex items-center justify-center w-8 h-8 sm:w-9 sm:h-9 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 transition-all duration-200"
                            :class="spaceAudio.isMuted.value ? 'text-red-400/80 hover:text-red-400' : 'text-white/80 hover:text-white'"
                            :title="spaceAudio.isMuted.value ? 'Unmute' : 'Mute'"
                            @click="spaceAudio.toggleMute()"
                        >
                            <VolumeX v-if="spaceAudio.isMuted.value" class="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                            <Volume2 v-else class="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                        </button>

                        <!-- Divider -->
                        <div class="w-px h-5 sm:h-6 bg-white/10" />

                        <!-- Language Selector -->
                        <CommonSettingsLocale />

                        <!-- Divider -->
                        <div class="w-px h-5 sm:h-6 bg-white/10" />

                        <!-- Navigation Links -->
                        <NuxtLink
                            :to="$localeRoute({ path: '/about' })"
                            class="px-3 py-1.5 sm:px-4 sm:py-2 rounded-lg text-xs sm:text-sm font-medium text-white/70 hover:text-white hover:bg-white/5 transition-all duration-200"
                        >
                            {{ $t("pages.about.title") }}
                        </NuxtLink>

                        <NuxtLink
                            :to="$localeRoute({ path: '/metrics' })"
                            class="px-3 py-1.5 sm:px-4 sm:py-2 rounded-lg text-xs sm:text-sm font-medium text-white/70 hover:text-white hover:bg-white/5 transition-all duration-200"
                        >
                            {{ $t("pages.metrics.title") }}
                        </NuxtLink>
                    </nav>

                    <!-- Controls Hint - Hidden on small tablets, visible on larger screens -->
                    <div class="hidden md:block glass rounded-xl px-3 py-2.5 lg:px-4 lg:py-3 text-xs space-y-1 lg:space-y-1.5 text-right pointer-events-auto">
                        <p class="text-white/60 flex items-center gap-2 justify-end">
                            <span class="text-white/80 text-[11px] lg:text-xs">{{ $t("pages.home.controls.space") }}</span>
                            <kbd class="px-1.5 py-0.5 lg:px-2 rounded bg-white/10 text-white/90 text-[9px] lg:text-[10px] font-mono">WASD</kbd>
                        </p>
                        <p class="text-white/60 flex items-center gap-2 justify-end">
                            <span class="text-white/80 text-[11px] lg:text-xs">{{ $t("pages.home.controls.planet") }}</span>
                            <kbd class="px-1.5 py-0.5 lg:px-2 rounded bg-white/10 text-white/90 text-[9px] lg:text-[10px] font-mono">DRAG</kbd>
                        </p>
                    </div>
                </div>
            </footer>
        </div>

        <!-- Mobile Menu Backdrop -->
        <Transition
            enter-active-class="transition-opacity duration-200"
            enter-from-class="opacity-0"
            enter-to-class="opacity-100"
            leave-active-class="transition-opacity duration-150"
            leave-from-class="opacity-100"
            leave-to-class="opacity-0"
        >
            <div
                v-if="isMobileMenuOpen"
                class="absolute inset-0 bg-black/30 z-[5] sm:hidden"
                @click="isMobileMenuOpen = false"
            />
        </Transition>
    </main>
</template>

<style scoped>
/* Safe area support for notched devices */
@supports (padding: max(0px)) {
    .h-\[100dvh\] {
        padding-bottom: env(safe-area-inset-bottom);
    }
}

/* Ensure touch targets are adequate on mobile */
@media (max-width: 640px) {
    .btn-cta-primary {
        min-height: 44px;
    }
}
</style>
