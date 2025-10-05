<script setup lang="ts">
    const isLoading = defineModel<boolean>({ default: true });

    const startTime = ref(Date.now());

    async function hideLoading() {
        const elapsed = Date.now() - startTime.value;
        const remaining = Math.max(0, CONSTANTS.HOME.MIN_LOADING_TIME - elapsed);

        await new Promise((resolve) => setTimeout(resolve, remaining));

        isLoading.value = false;
    }

    defineExpose({ hideLoading });
</script>

<template>
    <Transition name="fade">
        <div
            v-if="isLoading"
            class="fixed inset-0 z-50 flex flex-col items-center justify-center bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950"
        >
            <div class="relative mb-8">
                <div class="absolute inset-0 animate-pulse blur-2xl bg-primary/30 rounded-full" />
                <div
                    class="relative size-32 rounded-full bg-primary/10 flex items-center justify-center"
                >
                    <svg
                        class="size-20 text-primary animate-spin"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                    >
                        <circle
                            class="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            stroke-width="4"
                        />
                        <path
                            class="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                        />
                    </svg>
                </div>
            </div>

            <h1 class="text-3xl font-bold mb-2 tracking-wider font-pixelify">A WORLD AWAY</h1>
            <p class="text-sm text-muted-foreground animate-pulse">
                {{ $t("pages.home.loading.slogan") }}
            </p>
        </div>
    </Transition>
</template>

<style scoped>
    .fade-enter-active,
    .fade-leave-active {
        transition: opacity 0.5s ease;
    }

    .fade-enter-from,
    .fade-leave-to {
        opacity: 0;
    }

    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
    }
</style>
