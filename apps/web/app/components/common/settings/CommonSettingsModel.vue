<script setup lang="ts">
    import { toTypedSchema } from "@vee-validate/zod";
    import { useForm } from "vee-validate";
    import { z } from "zod";

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
                .number()
                .min(0.4518373720000007, t("pages.predict.form.validation.required"))
                .max(569.9815810000001, t("pages.predict.form.validation.required"))
                .refine(
                    (val) => !isNaN(val) && val > 0,
                    t("pages.predict.form.validation.positive_number"),
                )
                .default(365.25),

            transit_duration_hours: z.coerce
                .number()
                .min(0.4853028172653019, t("pages.predict.form.validation.required"))
                .max(22.98957577832262, t("pages.predict.form.validation.required"))
                .refine(
                    (val) => !isNaN(val) && val > 0,
                    t("pages.predict.form.validation.positive_number"),
                )
                .default(13.5),

            planet_radius_earth: z.coerce
                .number()
                .min(0.59, t("pages.predict.form.validation.required"))
                .max(156.00040000000044, t("pages.predict.form.validation.required"))
                .refine(
                    (val) => !isNaN(val) && val > 0,
                    t("pages.predict.form.validation.positive_number"),
                )
                .default(1.0),

            equilibrium_temperature_K: z.coerce
                .number()
                .min(0.0, t("pages.predict.form.validation.required"))
                .max(29339.3904417822, t("pages.predict.form.validation.required"))
                .refine(
                    (val) => !isNaN(val) && val > 0,
                    t("pages.predict.form.validation.positive_number"),
                )
                .default(288),

            transit_depth_ppm: z.coerce
                .number()
                .min(0.0, t("pages.predict.form.validation.required"))
                .max(357953.2000000003, t("pages.predict.form.validation.required"))
                .refine(
                    (val) => !isNaN(val) && val > 0,
                    t("pages.predict.form.validation.positive_number"),
                )
                .default(84),

            insolation_flux_Earth: z.coerce
                .number()
                .min(0.0, t("pages.predict.form.validation.required"))
                .max(40519.208836177786, t("pages.predict.form.validation.required"))
                .refine(
                    (val) => !isNaN(val) && val > 0,
                    t("pages.predict.form.validation.positive_number"),
                )
                .default(1.0),

            stellar_radius_solar: z.coerce
                .number()
                .min(0.25, t("pages.predict.form.validation.required"))
                .max(8.055360000000007, t("pages.predict.form.validation.required"))
                .refine(
                    (val) => !isNaN(val) && val > 0,
                    t("pages.predict.form.validation.positive_number"),
                )
                .default(1.0),

            stellar_temperature_K: z.coerce
                .number()
                .min(3217.52, t("pages.predict.form.validation.required"))
                .max(8991.560000000001, t("pages.predict.form.validation.required"))
                .refine(
                    (val) => !isNaN(val) && val >= 1000,
                    t("pages.predict.form.validation.stellar_temp_min"),
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

                modal.loadComponent({
                    loader: () => import("@/components/common/modal/CommonPredictionModal.vue"),
                    key: "prediction:modal",
                    props: {
                        result: {
                            input_received,
                            features_expected,
                            prediction,
                            threshold,
                        },
                    },
                });
            })
            .catch((error) => console.error(error))
            .finally(() => (isLoading.value = false));
    });
</script>

<template>
    <form class="space-y-4" @submit="onSubmit">
        <div class="space-y-4">
            <h3 class="text-lg font-semibold flex items-center gap-2">
                🌍 {{ $t("pages.predict.form.sections.orbital") }}
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
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

        <div class="space-y-4">
            <h3 class="text-lg font-semibold flex items-center gap-2">
                🪐 {{ $t("pages.predict.form.sections.planet") }}
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
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

        <div class="space-y-4">
            <h3 class="text-lg font-semibold flex items-center gap-2">
                📉 {{ $t("pages.predict.form.sections.transit") }}
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
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

        <div class="space-y-4">
            <h3 class="text-lg font-semibold flex items-center gap-2">
                ⭐ {{ $t("pages.predict.form.sections.stellar") }}
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
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

        <div class="flex space-x-2">
            <Button type="submit" class="grow" :disabled="isLoading">
                <span>{{ $t("pages.predict.form.submit") }}</span>
            </Button>

            <CommonSettingsLocale class="grow" />
        </div>
    </form>
</template>
