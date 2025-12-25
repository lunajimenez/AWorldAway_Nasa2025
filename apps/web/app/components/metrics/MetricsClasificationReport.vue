<script setup lang="ts">
    interface Props {
        report: {
            NOT_CONFIRMED: {
                precision: number;
                recall: number;
                "f1-score": number;
                support: number;
            };
            CONFIRMED: {
                precision: number;
                recall: number;
                "f1-score": number;
                support: number;
            };
            "macro avg": {
                precision: number;
                recall: number;
                "f1-score": number;
                support: number;
            };
            "weighted avg": {
                precision: number;
                recall: number;
                "f1-score": number;
                support: number;
            };
        };
    }

    const props = defineProps<Props>();

    const { t } = useI18n({ useScope: "local" });

    const rows = [
        { label: t("notConfirmed"), data: props.report.NOT_CONFIRMED, highlight: false },
        { label: t("confirmed"), data: props.report.CONFIRMED, highlight: false },
        { label: t("macroAvg"), data: props.report["macro avg"], highlight: true },
        { label: t("weightedAvg"), data: props.report["weighted avg"], highlight: true },
    ];
</script>

<template>
    <Card class="bg-card border-border">
        <CardHeader>
            <CardTitle>{{ t("title") }}</CardTitle>
            <CardDescription>{{ t("description") }}</CardDescription>
        </CardHeader>
        <CardContent>
            <div class="overflow-x-auto -mx-6 px-6">
                <div class="min-w-[400px] space-y-2">
                    <div class="grid grid-cols-5 gap-2 pb-2 border-b border-border">
                        <div class="text-xs sm:text-sm font-semibold text-muted-foreground">{{ t("class") }}</div>
                        <div class="text-xs sm:text-sm font-semibold text-muted-foreground text-right">
                            {{ t("precision") }}
                        </div>
                        <div class="text-xs sm:text-sm font-semibold text-muted-foreground text-right">
                            {{ t("recall") }}
                        </div>
                        <div class="text-xs sm:text-sm font-semibold text-muted-foreground text-right">
                            {{ t("f1Score") }}
                        </div>
                        <div class="text-xs sm:text-sm font-semibold text-muted-foreground text-right">
                            {{ t("support") }}
                        </div>
                    </div>

                    <div
                        v-for="(row, idx) in rows"
                        :key="idx"
                        :class="[
                            'grid grid-cols-5 gap-2 py-2 sm:py-3 rounded-lg items-center',
                            row.highlight ? 'bg-primary/5 px-2' : '',
                        ]"
                    >
                        <div
                            :class="[
                                'text-xs sm:text-sm font-medium truncate',
                                row.highlight ? 'text-primary' : 'text-foreground',
                            ]"
                        >
                            {{ row.label }}
                        </div>
                        <div class="text-xs sm:text-sm text-right font-mono text-foreground">
                            {{ (row.data.precision * 100).toFixed(1) }}%
                        </div>
                        <div class="text-xs sm:text-sm text-right font-mono text-foreground">
                            {{ (row.data.recall * 100).toFixed(1) }}%
                        </div>
                        <div class="text-xs sm:text-sm text-right font-mono text-foreground">
                            {{ (row.data["f1-score"] * 100).toFixed(1) }}%
                        </div>
                        <div class="text-xs sm:text-sm text-right font-mono text-muted-foreground">
                            {{ row.data.support.toFixed(0) }}
                        </div>
                    </div>
                </div>
            </div>
        </CardContent>
    </Card>
</template>

<i18n lang="yaml">
en-US:
    title: "Classification Report"
    description: "Per-class performance metrics"
    class: "Class"
    precision: "Precision"
    recall: "Recall"
    f1Score: "F1-Score"
    support: "Support"
    notConfirmed: "NOT CONFIRMED"
    confirmed: "CONFIRMED"
    macroAvg: "Macro Avg"
    weightedAvg: "Weighted Avg"

es-ES:
    title: "Reporte de Clasificación"
    description: "Métricas de rendimiento por clase"
    class: "Clase"
    precision: "Precisión"
    recall: "Exhaustividad"
    f1Score: "Puntuación F1"
    support: "Soporte"
    notConfirmed: "NO CONFIRMADO"
    confirmed: "CONFIRMADO"
    macroAvg: "Promedio Macro"
    weightedAvg: "Promedio Ponderado"
</i18n>
