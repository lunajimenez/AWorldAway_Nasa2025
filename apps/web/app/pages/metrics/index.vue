<script setup lang="ts">
    import { ChevronsLeft } from "lucide-vue-next";

    interface Response {
        threshold: number;
        metrics: Metrics;
    }

    interface Metrics {
        ROC_AUC: number;
        PR_AUC: number;
        Best_Threshold_F1: number;
        Accuracy: number;
        Macro_F1: number;
        Positives_in_Test: number;
        Negatives_in_Test: number;
        N_features_numeric: number;
        N_features_categorical: number;
        Confusion_Matrix: Array<number[]>;
        Classification_Report: ClassificationReport;
        Rows_Total: number;
        Rows_Labeled: number;
    }

    interface ClassificationReport {
        NOT_CONFIRMED: Confirmed;
        CONFIRMED: Confirmed;
        accuracy: number;
        "macro avg": Confirmed;
        "weighted avg": Confirmed;
    }

    interface Confirmed {
        precision: number;
        recall: number;
        "f1-score": number;
        support: number;
    }

    const {
        public: { apiBase },
    } = useRuntimeConfig();

    const { t } = useI18n({ useScope: "local" });
    const { data } = useFetch<Response>("/api/model/metrics", { baseURL: apiBase });
</script>

<template>
    <div class="min-h-screen bg-background dark">
        <div class="container mx-auto px-4 sm:px-6 py-6 sm:py-12 space-y-4 sm:space-y-6">
            <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div class="space-y-1 sm:space-y-2">
                    <h1 class="text-2xl sm:text-4xl font-bold tracking-tight text-balance">
                        {{ t("title") }}
                    </h1>
                    <p class="text-muted-foreground text-sm sm:text-lg">
                        {{ t("description") }}
                    </p>
                </div>

                <div class="flex space-x-2">
                    <Button variant="outline" as-child size="sm" class="sm:size-default group hover:border-primary/50 transition-all duration-300">
                        <NuxtLink :to="$localeRoute({ path: '/' })">
                            <ChevronsLeft class="mr-1 w-4 h-4 transition-transform group-hover:-translate-x-1" />
                            <span class="text-xs sm:text-sm">{{ t("goBack") }}</span>
                        </NuxtLink>
                    </Button>
                    <CommonSettingsLocale />
                </div>
            </div>

            <template v-if="data">
                <MetricsOverview :metrics="data?.metrics" />

                <div class="grid gap-4 sm:gap-6 lg:grid-cols-2">
                    <MetricsConfusionMatrix :confusion-matrix="data.metrics.Confusion_Matrix" />

                    <MetricsClasificationReport :report="data.metrics.Classification_Report" />
                </div>

                <MetricsDatasetInformation :metrics="data.metrics" :threshold="data.threshold" />
            </template>
        </div>
    </div>
</template>

<i18n lang="yaml">
en-US:
    title: "Model Performance Dashboard"
    description: "Binary Classification Model Evaluation Metrics"
    goBack: "Go back"

es-ES:
    title: "Panel de Rendimiento del Modelo"
    description: "Métricas de Evaluación del Modelo de Clasificación Binaria"
    goBack: "Volver"
</i18n>
