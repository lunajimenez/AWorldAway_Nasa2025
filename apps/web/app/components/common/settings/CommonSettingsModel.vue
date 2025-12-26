<script setup lang="ts">
    import { toTypedSchema } from "@vee-validate/zod";
    import { useForm } from "vee-validate";
    import { z } from "zod";
    import { Eye, Loader2 } from "lucide-vue-next";

    const { t } = useI18n();

    const SOURCE_MISSIONS = [
        { value: "Kepler", label: "pages.predict.form.missions.kepler" },
        { value: "K2", label: "pages.predict.form.missions.k2" },
        { value: "TESS", label: "pages.predict.form.missions.tess" },
    ] as const;

    const MISSION_VALUES = SOURCE_MISSIONS.map((m) => m.value);

    const Schema = computed(() =>
        z.object({
            orbital_period_days: z.coerce
                .number({ error: t("pages.predict.form.validation.required") })
                .min(0.4518373720000007, { message: t("pages.predict.form.validation.orbital_period_range") })
                .max(569.9815810000001, { message: t("pages.predict.form.validation.orbital_period_range") })
                .refine(
                    (val) => !isNaN(val) && val > 0,
                    { message: t("pages.predict.form.validation.positive_number") },
                )
                .default(365.25),

            transit_duration_hours: z.coerce
                .number({ error: t("pages.predict.form.validation.required") })
                .min(0.4853028172653019, { message: t("pages.predict.form.validation.transit_duration_range") })
                .max(22.98957577832262, { message: t("pages.predict.form.validation.transit_duration_range") })
                .refine(
                    (val) => !isNaN(val) && val > 0,
                    { message: t("pages.predict.form.validation.positive_number") },
                )
                .default(13.5),

            planet_radius_earth: z.coerce
                .number({ error: t("pages.predict.form.validation.required") })
                .min(0.59, { message: t("pages.predict.form.validation.planet_radius_range") })
                .max(156.00040000000044, { message: t("pages.predict.form.validation.planet_radius_range") })
                .refine(
                    (val) => !isNaN(val) && val > 0,
                    { message: t("pages.predict.form.validation.positive_number") },
                )
                .default(1.0),

            equilibrium_temperature_K: z.coerce
                .number({ error: t("pages.predict.form.validation.required") })
                .min(0.0, { message: t("pages.predict.form.validation.equilibrium_temp_range") })
                .max(29339.3904417822, { message: t("pages.predict.form.validation.equilibrium_temp_range") })
                .refine(
                    (val) => !isNaN(val) && val > 0,
                    { message: t("pages.predict.form.validation.positive_number") },
                )
                .default(288),

            transit_depth_ppm: z.coerce
                .number({ error: t("pages.predict.form.validation.required") })
                .min(0.0, { message: t("pages.predict.form.validation.transit_depth_range") })
                .max(357953.2000000003, { message: t("pages.predict.form.validation.transit_depth_range") })
                .refine(
                    (val) => !isNaN(val) && val > 0,
                    { message: t("pages.predict.form.validation.positive_number") },
                )
                .default(84),

            insolation_flux_Earth: z.coerce
                .number({ error: t("pages.predict.form.validation.required") })
                .min(0.0, { message: t("pages.predict.form.validation.insolation_flux_range") })
                .max(40519.208836177786, { message: t("pages.predict.form.validation.insolation_flux_range") })
                .refine(
                    (val) => !isNaN(val) && val > 0,
                    { message: t("pages.predict.form.validation.positive_number") },
                )
                .default(1.0),

            stellar_radius_solar: z.coerce
                .number({ error: t("pages.predict.form.validation.required") })
                .min(0.25, { message: t("pages.predict.form.validation.stellar_radius_range") })
                .max(8.055360000000007, { message: t("pages.predict.form.validation.stellar_radius_range") })
                .refine(
                    (val) => !isNaN(val) && val > 0,
                    { message: t("pages.predict.form.validation.positive_number") },
                )
                .default(1.0),

            stellar_temperature_K: z.coerce
                .number({ error: t("pages.predict.form.validation.required") })
                .min(3217.52, { message: t("pages.predict.form.validation.stellar_temp_range") })
                .max(8991.560000000001, { message: t("pages.predict.form.validation.stellar_temp_range") })
                .refine(
                    (val) => !isNaN(val) && val >= 1000,
                    { message: t("pages.predict.form.validation.stellar_temp_min") },
                )
                .default(5778),
        }),
    );

    const ResponseSchema = z.object({
        features_expected: z.array(z.string()),
        threshold: z.coerce.number(),
        input_received: z.record(z.string(), z.string()),
        prediction: z.object({
            dataset: z.string().nullable().optional(),
            object_id: z.string().nullable().optional(),
            pred_confirmed: z.coerce.number(),
            score_confirmed: z.coerce.number(),
        }),
    });

    const { handleSubmit } = useForm({
        validationSchema: toTypedSchema(Schema.value),
        initialValues: Schema.value.parse({}),
    });

    const isLoading = ref(false);
    const modal = useModal();
    type PredictionResult = {
        input_received: Record<string, string>;
        features_expected: string[];
        prediction: {
            dataset?: string | null;
            object_id?: string | null;
            pred_confirmed: number;
            score_confirmed: number;
        };
        threshold: number;
    };

    const LAST_RESULT_KEY = 'exoplanet_last_prediction_result';

    // Load last result from localStorage on mount
    const loadLastResult = (): PredictionResult | null => {
        if (import.meta.client) {
            try {
                const stored = localStorage.getItem(LAST_RESULT_KEY);
                if (stored) {
                    return JSON.parse(stored) as PredictionResult;
                }
            } catch (e) {
                console.error('Error loading last result from localStorage:', e);
            }
        }
        return null;
    };

    // Save last result to localStorage
    const saveLastResult = (result: PredictionResult) => {
        if (import.meta.client) {
            try {
                localStorage.setItem(LAST_RESULT_KEY, JSON.stringify(result));
            } catch (e) {
                console.error('Error saving last result to localStorage:', e);
            }
        }
    };

    const lastResult = ref<PredictionResult | null>(loadLastResult());

    const { clearInterpretation } = useInterpretationCache();

    const {
        public: { apiBase },
    } = useRuntimeConfig();
    
    const onSubmit = handleSubmit((values) => {
        isLoading.value = true;

        $fetch("/api/model/predict-one", {
            baseURL: apiBase,
            query: values,
        })
            .then((response) => {
                const { features_expected, prediction, threshold, input_received } =
                    ResponseSchema.parse(response);

                // Clear cached interpretation for this input
                clearInterpretation(input_received);

                // Store the result
                lastResult.value = {
                    input_received,
                    features_expected,
                    prediction,
                    threshold,
                };
                
                // Persist to localStorage
                saveLastResult(lastResult.value);

                // Open modal with results (new prediction, so clear interpretation)
                openResultsModal(true);
            })
            .catch((error) => console.error(error))
            .finally(() => (isLoading.value = false));
    });

    function openResultsModal(isNewPrediction: boolean = false) {
        if (!lastResult.value) return;
        
        modal.loadComponent({
            loader: () => import("@/components/common/modal/CommonPredictionModal.vue"),
            key: "prediction:modal",
            props: {
                result: lastResult.value,
                isNewPrediction,
            },
        });
        modal.open.value = true;
    }
</script>

<template>
    <form class="space-y-3 sm:space-y-4" @submit="onSubmit">
        <div class="space-y-3 sm:space-y-4">
            <h3 class="text-sm sm:text-lg font-semibold flex items-center gap-1.5 sm:gap-2">
                🌍 {{ $t("pages.predict.form.sections.orbital") }}
            </h3>
            <div class="grid grid-cols-1 gap-3 sm:gap-4">
                <FormField v-slot="{ componentField }" name="orbital_period_days">
                    <FormItem>
                        <FormLabel>
                            {{ $t("pages.predict.form.fields.orbital_period.label") }}
                        </FormLabel>
                        <FormControl>
                            <Input
                                :placeholder="
                                    $t('pages.predict.form.fields.orbital_period.placeholder')
                                "
                                v-bind="componentField"
                            />
                        </FormControl>
                        <FormDescription>
                            {{ $t("pages.predict.form.fields.orbital_period.description") }}
                        </FormDescription>
                        <FormMessage />
                    </FormItem>
                </FormField>

                <FormField v-slot="{ componentField }" name="transit_duration_hours">
                    <FormItem>
                        <FormLabel>
                            {{ $t("pages.predict.form.fields.transit_duration.label") }}
                        </FormLabel>
                        <FormControl>
                            <Input
                                :placeholder="
                                    $t('pages.predict.form.fields.transit_duration.placeholder')
                                "
                                v-bind="componentField"
                            />
                        </FormControl>
                        <FormDescription>
                            {{ $t("pages.predict.form.fields.transit_duration.description") }}
                        </FormDescription>
                        <FormMessage />
                    </FormItem>
                </FormField>
            </div>
        </div>

        <div class="space-y-3 sm:space-y-4">
            <h3 class="text-sm sm:text-lg font-semibold flex items-center gap-1.5 sm:gap-2">
                🪐 {{ $t("pages.predict.form.sections.planet") }}
            </h3>
            <div class="grid grid-cols-1 gap-3 sm:gap-4 items-start">
                <FormField v-slot="{ componentField }" name="planet_radius_earth">
                    <FormItem>
                        <FormLabel>
                            {{ $t("pages.predict.form.fields.planet_radius.label") }}
                        </FormLabel>
                        <FormControl>
                            <Input
                                :placeholder="
                                    $t('pages.predict.form.fields.planet_radius.placeholder')
                                "
                                v-bind="componentField"
                            />
                        </FormControl>
                        <FormDescription>
                            {{ $t("pages.predict.form.fields.planet_radius.description") }}
                        </FormDescription>
                        <FormMessage />
                    </FormItem>
                </FormField>

                <FormField v-slot="{ componentField }" name="equilibrium_temperature_K">
                    <FormItem>
                        <FormLabel>
                            {{ $t("pages.predict.form.fields.equilibrium_temp.label") }}
                        </FormLabel>
                        <FormControl>
                            <Input
                                :placeholder="
                                    $t('pages.predict.form.fields.equilibrium_temp.placeholder')
                                "
                                v-bind="componentField"
                            />
                        </FormControl>
                        <FormDescription>
                            {{ $t("pages.predict.form.fields.equilibrium_temp.description") }}
                        </FormDescription>
                        <FormMessage />
                    </FormItem>
                </FormField>
            </div>
        </div>

        <div class="space-y-3 sm:space-y-4">
            <h3 class="text-sm sm:text-lg font-semibold flex items-center gap-1.5 sm:gap-2">
                📉 {{ $t("pages.predict.form.sections.transit") }}
            </h3>
            <div class="grid grid-cols-1 gap-3 sm:gap-4 items-start">
                <FormField v-slot="{ componentField }" name="transit_depth_ppm">
                    <FormItem>
                        <FormLabel>
                            {{ $t("pages.predict.form.fields.transit_depth.label") }}
                        </FormLabel>
                        <FormControl>
                            <Input
                                :placeholder="
                                    $t('pages.predict.form.fields.transit_depth.placeholder')
                                "
                                v-bind="componentField"
                            />
                        </FormControl>
                        <FormDescription>
                            {{ $t("pages.predict.form.fields.transit_depth.description") }}
                        </FormDescription>
                        <FormMessage />
                    </FormItem>
                </FormField>

                <FormField v-slot="{ componentField }" name="insolation_flux_Earth">
                    <FormItem>
                        <FormLabel>
                            {{ $t("pages.predict.form.fields.insolation_flux.label") }}
                        </FormLabel>
                        <FormControl>
                            <Input
                                :placeholder="
                                    $t('pages.predict.form.fields.insolation_flux.placeholder')
                                "
                                v-bind="componentField"
                            />
                        </FormControl>
                        <FormDescription>
                            {{ $t("pages.predict.form.fields.insolation_flux.description") }}
                        </FormDescription>
                        <FormMessage />
                    </FormItem>
                </FormField>
            </div>
        </div>

        <div class="space-y-3 sm:space-y-4">
            <h3 class="text-sm sm:text-lg font-semibold flex items-center gap-1.5 sm:gap-2">
                ⭐ {{ $t("pages.predict.form.sections.stellar") }}
            </h3>
            <div class="grid grid-cols-1 gap-3 sm:gap-4 items-start">
                <FormField v-slot="{ componentField }" name="stellar_radius_solar">
                    <FormItem>
                        <FormLabel>
                            {{ $t("pages.predict.form.fields.stellar_radius.label") }}
                        </FormLabel>
                        <FormControl>
                            <Input
                                :placeholder="
                                    $t('pages.predict.form.fields.stellar_radius.placeholder')
                                "
                                v-bind="componentField"
                            />
                        </FormControl>
                        <FormDescription>
                            {{ $t("pages.predict.form.fields.stellar_radius.description") }}
                        </FormDescription>
                        <FormMessage />
                    </FormItem>
                </FormField>

                <FormField v-slot="{ componentField }" name="stellar_temperature_K">
                    <FormItem>
                        <FormLabel>
                            {{ $t("pages.predict.form.fields.stellar_temp.label") }}
                        </FormLabel>
                        <FormControl>
                            <Input
                                :placeholder="
                                    $t('pages.predict.form.fields.stellar_temp.placeholder')
                                "
                                v-bind="componentField"
                            />
                        </FormControl>
                        <FormDescription>
                            {{ $t("pages.predict.form.fields.stellar_temp.description") }}
                        </FormDescription>
                        <FormMessage />
                    </FormItem>
                </FormField>
            </div>
        </div>

        <!-- Action Buttons Section -->
        <div class="pt-2 sm:pt-4 border-t border-white/10">
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 sm:gap-3">
                <!-- Primary: Predict Button -->
                <Button 
                    type="submit" 
                    class="w-full h-10 sm:h-11 text-xs sm:text-sm font-medium transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]" 
                    :disabled="isLoading"
                >
                    <Loader2 v-if="isLoading" class="w-3.5 h-3.5 sm:w-4 sm:h-4 mr-1.5 sm:mr-2 animate-spin" />
                    <span>{{ $t("pages.predict.form.submit") }}</span>
                </Button>

                <!-- Secondary: View Results Button (only shows when there's a result) -->
                <Button 
                    v-if="lastResult" 
                    type="button" 
                    variant="outline"
                    class="w-full h-10 sm:h-11 text-xs sm:text-sm font-medium gap-1.5 sm:gap-2 transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] border-white/20 hover:border-white/40 hover:bg-white/5"
                    @click="openResultsModal(false)"
                >
                    <Eye class="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                    <span>{{ $t("pages.predict.form.view_results") }}</span>
                </Button>

                <!-- Language Selector - full width on its own row in mobile when lastResult exists -->
                <div :class="lastResult ? 'col-span-1 sm:col-span-2' : ''">
                    <CommonSettingsLocale class="w-full" />
                </div>
            </div>
        </div>
    </form>
</template>

