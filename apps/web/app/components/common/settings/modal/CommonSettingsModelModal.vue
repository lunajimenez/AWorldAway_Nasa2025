<script setup lang="ts">
    import { Rocket } from "lucide-vue-next";
</script>

<template>
    <SheetContent>
        <SheetHeader class="space-y-3 pb-6 border-b">
            <div class="flex items-center gap-3">
                <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                    <Rocket class="size-5 text-primary" />
                </div>
                <div class="flex flex-col items-start">
                    <SheetTitle class="text-2xl">Title</SheetTitle>
                    <SheetDescription class="text-left">Description</SheetDescription>
                </div>
            </div>
        </SheetHeader>
        <div class="px-4 space-y-6">
            <CommonSettingsModel />
        </div>
    </SheetContent>
</template>
