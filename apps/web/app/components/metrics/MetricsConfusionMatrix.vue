<script setup lang="ts">
    interface Props {
        confusionMatrix: number[][];
    }

    const props = withDefaults(defineProps<Props>(), {
        confusionMatrix: () => [
            [0, 0],
            [0, 0],
        ],
    });

    const { t } = useI18n({ useScope: "local" });

    const [row0, row1] = props.confusionMatrix;
    const [tn, fp] = row0!;
    const [fn, tp] = row1!;

    const total = tn! + fp! + fn! + tp!;

    const cells = [
        { label: t("trueNegative"), value: tn, color: "bg-chart-4/20 border-chart-4" },
        { label: t("falsePositive"), value: fp, color: "bg-destructive/20 border-destructive" },
        { label: t("falseNegative"), value: fn, color: "bg-destructive/20 border-destructive" },
        { label: t("truePositive"), value: tp, color: "bg-chart-1/20 border-chart-1" },
    ];
</script>

<template>
    <Card class="bg-card border-border">
        <CardHeader class="pb-2 sm:pb-6">
            <CardTitle class="text-base sm:text-lg">{{ t("title") }}</CardTitle>
            <CardDescription class="text-xs sm:text-sm">{{ t("description") }}</CardDescription>
        </CardHeader>
        <CardContent>
            <div class="space-y-3 sm:space-y-4">
                <div class="grid grid-cols-2 gap-2 sm:gap-4">
                    <div
                        v-for="cell in cells"
                        :key="cell.label"
                        :class="[
                            'p-3 sm:p-6 rounded-lg border-2 flex flex-col items-center justify-center space-y-1 sm:space-y-2',
                            cell.color,
                        ]"
                    >
                        <p class="text-[10px] sm:text-sm font-medium text-muted-foreground text-center">
                            {{ cell.label }}
                        </p>
                        <p class="text-xl sm:text-3xl font-bold text-foreground">{{ cell.value }}</p>
                        <p class="text-[10px] sm:text-xs text-muted-foreground">
                            {{ ((cell.value! / total) * 100).toFixed(1) }}%
                        </p>
                    </div>
                </div>

                <div class="flex items-center justify-between pt-3 sm:pt-4 border-t border-border">
                    <div class="text-center">
                        <p class="text-[10px] sm:text-sm text-muted-foreground">{{ t("predictedNegative") }}</p>
                        <p class="text-sm sm:text-lg font-semibold text-foreground">{{ tn! + fn! }}</p>
                    </div>
                    <div class="text-center">
                        <p class="text-[10px] sm:text-sm text-muted-foreground">{{ t("predictedPositive") }}</p>
                        <p class="text-sm sm:text-lg font-semibold text-foreground">{{ fp! + tp! }}</p>
                    </div>
                </div>
            </div>
        </CardContent>
    </Card>
</template>

<i18n lang="yaml">
en-US:
    title: "Confusion Matrix"
    description: "Model prediction distribution across classes"
    trueNegative: "True Negative"
    falsePositive: "False Positive"
    falseNegative: "False Negative"
    truePositive: "True Positive"
    predictedNegative: "Predicted Negative"
    predictedPositive: "Predicted Positive"

es-ES:
    title: "Matriz de Confusión"
    description: "Distribución de predicciones del modelo entre clases"
    trueNegative: "Verdadero Negativo"
    falsePositive: "Falso Positivo"
    falseNegative: "Falso Negativo"
    truePositive: "Verdadero Positivo"
    predictedNegative: "Predicho Negativo"
    predictedPositive: "Predicho Positivo"
</i18n>
