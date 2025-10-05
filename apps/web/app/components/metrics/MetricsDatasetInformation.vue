<script setup lang="ts">
    import { Database, Layers, Hash, Sliders } from "lucide-vue-next";

    interface Props {
        metrics: {
            Rows_Total: number;
            Rows_Labeled: number;
            N_features_numeric: number;
            N_features_categorical: number;
            Positives_in_Test: number;
            Negatives_in_Test: number;
        };
        threshold: number;
    }

    const props = defineProps<Props>();

    const { t } = useI18n({ useScope: "local" });

    const infoItems = [
        {
            icon: Database,
            label: t("totalRows.label"),
            value: props.metrics.Rows_Total.toLocaleString(),
            description: t("totalRows.description"),
        },
        {
            icon: Layers,
            label: t("labeledRows.label"),
            value: props.metrics.Rows_Labeled.toLocaleString(),
            description: t("labeledRows.description"),
        },
        {
            icon: Hash,
            label: t("numericFeatures.label"),
            value: props.metrics.N_features_numeric.toString(),
            description: t("numericFeatures.description"),
        },
        {
            icon: Hash,
            label: t("categoricalFeatures.label"),
            value: props.metrics.N_features_categorical.toString(),
            description: t("categoricalFeatures.description"),
        },
        {
            icon: Sliders,
            label: t("bestThreshold.label"),
            value: props.threshold.toFixed(4),
            description: t("bestThreshold.description"),
        },
        {
            icon: Database,
            label: t("classBalance.label"),
            value: `${props.metrics.Negatives_in_Test} / ${props.metrics.Positives_in_Test}`,
            description: t("classBalance.description"),
        },
    ];
</script>

<template>
    <Card class="bg-card border-border">
        <CardHeader>
            <CardTitle>{{ t("title") }}</CardTitle>
            <CardDescription>{{ t("description") }}</CardDescription>
        </CardHeader>
        <CardContent>
            <div class="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                <div
                    v-for="(item, idx) in infoItems"
                    :key="idx"
                    class="flex items-start gap-3 p-4 rounded-lg bg-muted/30 border border-border"
                >
                    <div class="p-2 bg-primary/10 rounded-lg">
                        <component :is="item.icon" class="h-4 w-4 text-primary" />
                    </div>
                    <div class="space-y-1">
                        <p class="text-xs text-muted-foreground">{{ item.label }}</p>
                        <p class="text-lg font-bold text-foreground font-mono">{{ item.value }}</p>
                        <p class="text-xs text-muted-foreground">{{ item.description }}</p>
                    </div>
                </div>
            </div>
        </CardContent>
    </Card>
</template>

<i18n lang="yaml">
en-US:
    title: "Dataset & Model Information"
    description: "Training data and model configuration details"
    totalRows:
        label: "Total Rows"
        description: "Complete dataset size"
    labeledRows:
        label: "Labeled Rows"
        description: "Rows with labels"
    numericFeatures:
        label: "Numeric Features"
        description: "Continuous variables"
    categoricalFeatures:
        label: "Categorical Features"
        description: "Discrete variables"
    bestThreshold:
        label: "Best Threshold"
        description: "Optimal F1 threshold"
    classBalance:
        label: "Class Balance"
        description: "Negative / Positive"

es-ES:
    title: "Información del Dataset y Modelo"
    description: "Detalles de los datos de entrenamiento y configuración del modelo"
    totalRows:
        label: "Filas Totales"
        description: "Tamaño completo del dataset"
    labeledRows:
        label: "Filas Etiquetadas"
        description: "Filas con etiquetas"
    numericFeatures:
        label: "Features Numéricas"
        description: "Variables continuas"
    categoricalFeatures:
        label: "Features Categóricas"
        description: "Variables discretas"
    bestThreshold:
        label: "Mejor Umbral"
        description: "Umbral óptimo F1"
    classBalance:
        label: "Balance de Clases"
        description: "Negativo / Positivo"
</i18n>
<script setup lang="ts">
    import { Database, Layers, Hash, Sliders } from "lucide-vue-next";

    interface Props {
        metrics: {
            Rows_Total: number;
            Rows_Labeled: number;
            N_features_numeric: number;
            N_features_categorical: number;
            Positives_in_Test: number;
            Negatives_in_Test: number;
        };
        threshold: number;
    }

    const props = defineProps<Props>();

    const { t } = useI18n({ useScope: "local" });

    const infoItems = [
        {
            icon: Database,
            label: t("totalRows.label"),
            value: props.metrics.Rows_Total.toLocaleString(),
            description: t("totalRows.description"),
        },
        {
            icon: Layers,
            label: t("labeledRows.label"),
            value: props.metrics.Rows_Labeled.toLocaleString(),
            description: t("labeledRows.description"),
        },
        {
            icon: Hash,
            label: t("numericFeatures.label"),
            value: props.metrics.N_features_numeric.toString(),
            description: t("numericFeatures.description"),
        },
        {
            icon: Hash,
            label: t("categoricalFeatures.label"),
            value: props.metrics.N_features_categorical.toString(),
            description: t("categoricalFeatures.description"),
        },
        {
            icon: Sliders,
            label: t("bestThreshold.label"),
            value: props.threshold.toFixed(4),
            description: t("bestThreshold.description"),
        },
        {
            icon: Database,
            label: t("classBalance.label"),
            value: `${props.metrics.Negatives_in_Test} / ${props.metrics.Positives_in_Test}`,
            description: t("classBalance.description"),
        },
    ];
</script>

<template>
    <Card class="bg-card border-border">
        <CardHeader>
            <CardTitle>{{ t("title") }}</CardTitle>
            <CardDescription>{{ t("description") }}</CardDescription>
        </CardHeader>
        <CardContent>
            <div class="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                <div
                    v-for="(item, idx) in infoItems"
                    :key="idx"
                    class="flex items-start gap-3 p-4 rounded-lg bg-muted/30 border border-border"
                >
                    <div class="p-2 bg-primary/10 rounded-lg">
                        <component :is="item.icon" class="h-4 w-4 text-primary" />
                    </div>
                    <div class="space-y-1">
                        <p class="text-xs text-muted-foreground">{{ item.label }}</p>
                        <p class="text-lg font-bold text-foreground font-mono">{{ item.value }}</p>
                        <p class="text-xs text-muted-foreground">{{ item.description }}</p>
                    </div>
                </div>
            </div>
        </CardContent>
    </Card>
</template>

<i18n lang="yaml">
en-US:
    title: "Dataset & Model Information"
    description: "Training data and model configuration details"
    totalRows:
        label: "Total Rows"
        description: "Complete dataset size"
    labeledRows:
        label: "Labeled Rows"
        description: "Rows with labels"
    numericFeatures:
        label: "Numeric Features"
        description: "Continuous variables"
    categoricalFeatures:
        label: "Categorical Features"
        description: "Discrete variables"
    bestThreshold:
        label: "Best Threshold"
        description: "Optimal F1 threshold"
    classBalance:
        label: "Class Balance"
        description: "Negative / Positive"

es-ES:
    title: "Información del Dataset y Modelo"
    description: "Detalles de los datos de entrenamiento y configuración del modelo"
    totalRows:
        label: "Filas Totales"
        description: "Tamaño completo del dataset"
    labeledRows:
        label: "Filas Etiquetadas"
        description: "Filas con etiquetas"
    numericFeatures:
        label: "Features Numéricas"
        description: "Variables continuas"
    categoricalFeatures:
        label: "Features Categóricas"
        description: "Variables discretas"
    bestThreshold:
        label: "Mejor Umbral"
        description: "Umbral óptimo F1"
    classBalance:
        label: "Balance de Clases"
        description: "Negativo / Positivo"
</i18n>
