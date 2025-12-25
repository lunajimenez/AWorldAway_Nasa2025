<script setup lang="ts">
    import { ChevronRight, Sparkles, Loader2 } from "lucide-vue-next";
    import { marked } from "marked";

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
        isNewPrediction?: boolean;
    }>();

    const { locale } = useI18n();
    const modal = useModal();
    const { getInterpretation, saveInterpretation } = useInterpretationCache();

    const isConfirmed = computed(() => props.result?.prediction.pred_confirmed === 1);
    const scorePercent = computed(() =>
        props.result ? (props.result.prediction.score_confirmed * 100).toFixed(2) : "0",
    );

    // AI Interpretation state
    const interpretation = ref<string | null>(null);
    const isLoadingInterpretation = ref(false);
    const interpretationError = ref<string | null>(null);

    // Parse markdown to HTML
    const interpretationHtml = computed(() => {
        if (!interpretation.value) return "";
        return marked.parse(interpretation.value, { breaks: true });
    });

    // Load cached interpretation on mount (only if not a new prediction)
    onMounted(() => {
        if (props.result?.input_received && !props.isNewPrediction) {
            const cached = getInterpretation(props.result.input_received);
            if (cached) {
                interpretation.value = cached;
            }
        }
    });

    // Watch for result changes to load cached interpretation
    watch(() => props.result, (newResult) => {
        if (newResult?.input_received) {
            // If it's a new prediction, don't load from cache
            if (props.isNewPrediction) {
                interpretation.value = null;
            } else {
                const cached = getInterpretation(newResult.input_received);
                if (cached) {
                    interpretation.value = cached;
                } else {
                    interpretation.value = null;
                }
            }
        }
    }, { immediate: true });

    async function getAIInterpretation() {
        if (!props.result) return;

        isLoadingInterpretation.value = true;
        interpretationError.value = null;

        try {
            const response = await $fetch("/api/interpret", {
                method: "POST",
                body: {
                    prediction: props.result.prediction,
                    inputData: props.result.input_received,
                    threshold: props.result.threshold,
                    locale: locale.value,
                },
            });

            interpretation.value = response.interpretation;
            
            // Save to cache
            saveInterpretation(props.result.input_received, response.interpretation);
        } catch (error: any) {
            console.error("Error getting interpretation:", error);
            interpretationError.value = error.message || "Failed to get interpretation";
        } finally {
            isLoadingInterpretation.value = false;
        }
    }

    function getScoreColor(score: number) {
        if (score >= 0.7) return "text-green-600 dark:text-green-400";
        if (score >= 0.4) return "text-yellow-600 dark:text-yellow-400";
        return "text-red-600 dark:text-red-400";
    }
</script>

<template>
    <DialogContent class="sm:max-w-[600px] max-h-[90vh] overflow-y-auto">
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

            <!-- AI Interpretation Section -->
            <div class="space-y-3">
                <div class="flex items-center justify-between">
                    <h4 class="font-semibold flex items-center gap-2">
                        <Sparkles class="h-4 w-4 text-purple-500" />
                        {{ $t("pages.predict.result.ai_interpretation.title") }}
                    </h4>
                    <Button
                        v-if="!interpretation && !isLoadingInterpretation"
                        size="sm"
                        class="gap-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white border-0"
                        @click="getAIInterpretation"
                    >
                        <Sparkles class="h-4 w-4" />
                        {{ $t("pages.predict.result.ai_interpretation.button") }}
                    </Button>
                </div>

                <!-- Loading State -->
                <div
                    v-if="isLoadingInterpretation"
                    class="relative overflow-hidden p-6 rounded-xl border border-purple-300/50 dark:border-purple-700/50 bg-gradient-to-br from-purple-50 via-indigo-50 to-violet-50 dark:from-purple-950/40 dark:via-indigo-950/40 dark:to-violet-950/40"
                >
                    <div class="absolute inset-0 bg-gradient-to-r from-transparent via-purple-200/20 to-transparent dark:via-purple-500/10 animate-shimmer"></div>
                    <div class="flex items-center gap-4">
                        <div class="relative">
                            <div class="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center">
                                <Loader2 class="h-5 w-5 animate-spin text-white" />
                            </div>
                        </div>
                        <div class="space-y-1">
                            <p class="font-medium text-purple-900 dark:text-purple-100">
                                {{ $t("pages.predict.result.ai_interpretation.loading") }}
                            </p>
                            <p class="text-xs text-purple-700 dark:text-purple-300 opacity-75">
                                Powered by GPT-OSS 120B
                            </p>
                        </div>
                    </div>
                </div>

                <!-- Error State -->
                <div
                    v-else-if="interpretationError"
                    class="p-4 rounded-xl border border-red-300/50 dark:border-red-700/50 bg-gradient-to-br from-red-50 to-rose-50 dark:from-red-950/40 dark:to-rose-950/40"
                >
                    <p class="text-sm text-red-900 dark:text-red-100 mb-3">
                        {{ interpretationError }}
                    </p>
                    <Button size="sm" variant="outline" class="border-red-300 dark:border-red-700" @click="getAIInterpretation">
                        {{ $t("pages.predict.result.ai_interpretation.retry") }}
                    </Button>
                </div>

                <!-- Interpretation Result -->
                <div
                    v-else-if="interpretation"
                    class="relative overflow-hidden rounded-xl border border-purple-300/50 dark:border-purple-700/50 bg-gradient-to-br from-purple-50 via-indigo-50 to-violet-50 dark:from-purple-950/30 dark:via-indigo-950/30 dark:to-violet-950/30"
                >
                    <!-- Header -->
                    <div class="px-5 py-3 border-b border-purple-200/50 dark:border-purple-700/50 bg-gradient-to-r from-purple-100/50 to-indigo-100/50 dark:from-purple-900/30 dark:to-indigo-900/30">
                        <div class="flex items-center gap-2">
                            <div class="w-6 h-6 rounded-full bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center">
                                <Sparkles class="h-3 w-3 text-white" />
                            </div>
                            <span class="text-xs font-semibold text-purple-800 dark:text-purple-200 uppercase tracking-wide">
                                Análisis Científico
                            </span>
                        </div>
                    </div>
                    
                    <!-- Content -->
                    <div class="p-5">
                        <div 
                            class="prose prose-sm dark:prose-invert max-w-none prose-p:text-purple-900 dark:prose-p:text-purple-100 prose-strong:text-purple-800 dark:prose-strong:text-purple-200 prose-headings:text-purple-900 dark:prose-headings:text-purple-100 prose-li:text-purple-900 dark:prose-li:text-purple-100"
                            v-html="interpretationHtml"
                        />
                    </div>

                    <!-- Footer -->
                    <div class="px-5 py-2 border-t border-purple-200/50 dark:border-purple-700/50 bg-purple-100/30 dark:bg-purple-900/20">
                        <p class="text-[10px] text-purple-600 dark:text-purple-400 flex items-center gap-1">
                            <Sparkles class="h-3 w-3" />
                            Generado con openai/gpt-oss-120b
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <DialogFooter>
            <Button variant="outline" @click="() => (modal.open.value = false)">
                {{ $t("pages.predict.result.close") }}
            </Button>
        </DialogFooter>
    </DialogContent>
</template>

