<script setup lang="ts">
    import "vue-sonner/style.css";

    const colorMode = useColorMode();

    async function onBeforeEnter() {
        await refreshNuxtData();
    }

    onMounted(() => {
        colorMode.preference = "dark";
    });
</script>

<template>
    <!-- TODO: Add a simple loading state when the app starts -->
    <NuxtLayout>
        <NuxtPage :transition="{ name: 'my', mode: 'out-in', onBeforeEnter }" />
        <NuxtLoadingIndicator />
        <NuxtRouteAnnouncer />
        <Toaster class="pointer-events-auto" />
    </NuxtLayout>
</template>

<style>
    .my-enter-active,
    .my-leave-active {
        transition: opacity 0.3s;
    }
    .my-enter,
    .my-leave-active {
        opacity: 0;
    }
</style>
