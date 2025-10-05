<script setup lang="ts">
    import { ChevronRight } from "lucide-vue-next";

    interface PredictionResult {
        threshold: number;
        features_expected: string[];
        input_received: Record<string, string>;
        prediction: {
            dataset?: string | null;
            object_id?: string | null;
            pred_confirmed: number;
            score_confirmed: number;
        };
    }

    const props = defineProps<{
        result: PredictionResult | null;
    }>();

    const modal = useModal();

    const isConfirmed = computed(() => props.result?.prediction.pred_confirmed === 1);
    const scorePercent = computed(() =>
        props.result ? (props.result.prediction.score_confirmed * 100).toFixed(2) : "0",
    );

    function getScoreColor(score: number) {
        if (score >= 0.7) return "text-green-600 dark:text-green-400";
        if (score >= 0.4) return "text-yellow-600 dark:text-yellow-400";
        return "text-red-600 dark:text-red-400";
    }
</script>

<template>
    <DialogContent class="sm:max-w-[600px]">
        <DialogHeader>
            <DialogTitle class="flex items-center gap-2">
                🔭 {{ $t("pages.predict.result.title") }}
            </DialogTitle>
            <DialogDescription>
                {{ $t("pages.predict.result.description") }}
            </DialogDescription>
        </DialogHeader>

        <div v-if="result" class="space-y-6">
            <div class="text-center p-6 bg-muted rounded-lg space-y-4">
                <div class="text-6xl">
                    {{ isConfirmed ? "🪐" : "❌" }}
                </div>
                <h3 class="text-2xl font-bold">
                    {{
                        isConfirmed
                            ? $t("pages.predict.result.confirmed")
                            : $t("pages.predict.result.not_confirmed")
                    }}
                </h3>
                <div class="space-y-2">
                    <p class="text-sm text-muted-foreground">
                        {{ $t("pages.predict.result.confidence") }}
                    </p>
                    <div
                        class="text-5xl font-bold"
                        :class="getScoreColor(result.prediction.score_confirmed)"
                    >
                        {{ scorePercent }}%
                    </div>
                    <Progress :model-value="result.prediction.score_confirmed * 100" class="h-3" />
                </div>
            </div>

            <div class="space-y-3">
                <h4 class="font-semibold flex items-center gap-2">
                    📊 {{ $t("pages.predict.result.analysis_details") }}
                </h4>
                <div class="grid grid-cols-2 gap-3 text-sm">
                    <div class="p-3 bg-muted rounded-md">
                        <p class="text-muted-foreground">
                            {{ $t("pages.predict.result.threshold") }}
                        </p>
                        <p class="font-semibold">{{ (result.threshold * 100).toFixed(2) }}%</p>
                    </div>
                    <div class="p-3 bg-muted rounded-md">
                        <p class="text-muted-foreground">
                            {{ $t("pages.predict.result.classification") }}
                        </p>
                        <p class="font-semibold">
                            {{
                                isConfirmed
                                    ? $t("pages.predict.result.exoplanet")
                                    : $t("pages.predict.result.false_positive")
                            }}
                        </p>
                    </div>
                </div>
            </div>

            <Collapsible class="space-y-2">
                <CollapsibleTrigger class="flex items-center gap-2 text-sm font-medium w-full">
                    <ChevronRight class="h-4 w-4 transition-transform" />
                    {{ $t("pages.predict.result.input_parameters") }}
                </CollapsibleTrigger>
                <CollapsibleContent class="space-y-2">
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                        <div
                            v-for="[key, value] in Object.entries(result.input_received)"
                            :key="key"
                            class="p-2 bg-muted rounded-md"
                        >
                            <p class="text-muted-foreground">{{ key }}</p>
                            <p class="font-mono">{{ value }}</p>
                        </div>
                    </div>
                </CollapsibleContent>
            </Collapsible>

            <div
                class="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800"
            >
                <p class="text-sm text-blue-900 dark:text-blue-100">
                    <span class="font-semibold"
                        >💡 {{ $t("pages.predict.result.interpretation.title") }}:</span
                    >
                    {{
                        result.prediction.score_confirmed >= 0.7
                            ? $t("pages.predict.result.interpretation.high")
                            : result.prediction.score_confirmed >= 0.4
                            ? $t("pages.predict.result.interpretation.medium")
                            : $t("pages.predict.result.interpretation.low")
                    }}
                </p>
            </div>
        </div>

        <DialogFooter>
            <Button variant="outline" @click="() => (modal.open.value = false)">
                {{ $t("pages.predict.result.close") }}
            </Button>
        </DialogFooter>
    </DialogContent>
</template>
