<script setup lang="ts">
    import { Rocket } from "lucide-vue-next";
</script>

<template>
    <SheetContent class="overflow-y-auto">
        <SheetHeader class="space-y-2 sm:space-y-3 pb-4 sm:pb-6 border-b">
            <div class="flex items-center gap-2 sm:gap-3">
                <div class="flex h-8 w-8 sm:h-10 sm:w-10 items-center justify-center rounded-lg bg-primary/10 shrink-0">
                    <Rocket class="size-4 sm:size-5 text-primary" />
                </div>
                <div class="flex flex-col items-start">
                    <SheetTitle class="text-lg sm:text-2xl">Title</SheetTitle>
                    <SheetDescription class="text-left text-xs sm:text-sm">Description</SheetDescription>
                </div>
            </div>
        </SheetHeader>
        <div class="px-2 sm:px-4 py-4 space-y-4 sm:space-y-6">
            <CommonSettingsModel />
        </div>
    </SheetContent>
</template>
