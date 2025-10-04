<script setup lang="ts">
    import "vue-sonner/style.css";

    async function onBeforeEnter() {
        await refreshNuxtData();
    }
</script>

<template>
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
