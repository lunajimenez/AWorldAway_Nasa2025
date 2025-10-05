<script setup lang="ts">
    import { TrendingUp, Target, Zap, Award } from "lucide-vue-next";

    interface Props {
        metrics: {
            ROC_AUC: number;
            PR_AUC: number;
            Accuracy: number;
            Macro_F1: number;
        };
    }

    const props = defineProps<Props>();

    const { t } = useI18n({ useScope: "local" });

    const keyMetrics = [
        {
            label: t("rocAuc.label"),
            value: props.metrics.ROC_AUC,
            icon: TrendingUp,
            description: t("rocAuc.description"),
        },
        {
            label: t("prAuc.label"),
            value: props.metrics.PR_AUC,
            icon: Target,
            description: t("prAuc.description"),
        },
        {
            label: t("accuracy.label"),
            value: props.metrics.Accuracy,
            icon: Zap,
            description: t("accuracy.description"),
        },
        {
            label: t("macroF1.label"),
            value: props.metrics.Macro_F1,
            icon: Award,
            description: t("macroF1.description"),
        },
    ];
</script>

<template>
    <div class="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <template v-for="metric in keyMetrics" :key="metric.label">
            <Card class="p-6 bg-card border-border hover:border-primary/50 transition-colors">
                <div class="flex items-start justify-between">
                    <div class="space-y-1">
                        <p class="text-sm font-medium text-muted-foreground">{{ metric.label }}</p>
                        <p class="text-3xl font-bold text-primary">
                            {{ (metric.value * 100).toFixed(2) }}%
                        </p>
                        <p class="text-xs text-muted-foreground">{{ metric.description }}</p>
                    </div>
                    <div class="p-2 bg-primary/10 rounded-lg">
                        <component :is="metric.icon" class="h-5 w-5 text-primary" />
                    </div>
                </div>
            </Card>
        </template>
    </div>
</template>

<i18n lang="yaml">
en-US:
    rocAuc:
        label: "ROC AUC"
        description: "Area Under ROC Curve"
    prAuc:
        label: "PR AUC"
        description: "Precision-Recall AUC"
    accuracy:
        label: "Accuracy"
        description: "Overall Accuracy"
    macroF1:
        label: "Macro F1"
        description: "Macro-averaged F1 Score"

es-ES:
    rocAuc:
        label: "ROC AUC"
        description: "Área Bajo la Curva ROC"
    prAuc:
        label: "PR AUC"
        description: "AUC de Precisión-Recall"
    accuracy:
        label: "Precisión"
        description: "Precisión General"
    macroF1:
        label: "F1 Macro"
        description: "Puntuación F1 Promedio Macro"
</i18n>
