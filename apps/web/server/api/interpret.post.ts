export default defineEventHandler(async (event) => {
    const config = useRuntimeConfig();
    const body = await readBody(event);

    const { prediction, inputData, threshold, locale } = body;

    if (!config.groqApiKey) {
        throw createError({
            statusCode: 500,
            message: "Groq API key not configured",
        });
    }

    const isSpanish = locale === "es-ES";

    const systemPrompt = isSpanish
        ? `Eres un astrofísico senior especializado en exoplanetología y análisis de datos de misiones como Kepler y TESS. Tu tarea es proporcionar una interpretación técnica y científica profunda de los resultados de un modelo de Machine Learning.

Debes analizar las CORRELACIONES entre los parámetros. No te limites a describir los valores; explica CÓMO interactúan entre sí para validar o invalidar la naturaleza planetaria del candidato.

Considera aspectos como:
- Relación entre Radio del Planeta y Profundidad del Tránsito (Proporcionalidad).
- Coherencia entre Periodo Orbital y Temperatura de Equilibrio.
- Proporción entre el Radio Estelar y el Radio del Planeta (detección de posibles binarias eclipsantes).
- Relación entre la Duración del Tránsito y el Periodo (geometría orbital).

Proporciona una explicación de 3 párrafos:
1. **Análisis de la Naturaleza del Objeto**: Basado en el tamaño, temperatura y flujo, ¿qué clase de objeto es? (¿Rocoso, Gigante gaseoso, Super-Tierra?).
2. **Correlación de Parámetros**: Explica por qué la combinación de estos datos específicos confiere o resta credibilidad al hallazgo según la física orbital.
3. **Veredicto del Modelo**: Interpreta la confianza del ML en el contexto de posibles falsos positivos comunes (como binarias de fondo o variabilidad estelar).

IMPORTANTE: 
- NO uses fórmulas LaTeX (evita \\frac, \\sqrt, \\simeq, etc.).
- Usa símbolos Unicode simples: R⊕ (radio terrestre), R☉ (radio solar), ≈ (aproximado), × (multiplicación), ² (cuadrado), √ (raíz).
- Escribe las ecuaciones en texto plano, ejemplo: "Tdur ≈ (P/π) × (R★/a)" en lugar de LaTeX.
- Usa Markdown para formato: **negritas**, *cursivas*, listas con guiones.

Usa un lenguaje profesional pero didáctico.`
        : `You are a senior astrophysicist specializing in exoplanetology and data analysis from missions like Kepler and TESS. Your task is to provide a deep technical and scientific interpretation of the results from a Machine Learning model.

You must analyze the CORRELATIONS between the parameters. Do not just describe the values; explain HOW they interact with each other to validate or invalidate the planetary nature of the candidate.

Consider aspects such as:
- Relationship between Planet Radius and Transit Depth (Proportionality).
- Consistency between Orbital Period and Equilibrium Temperature.
- Ratio between Stellar Radius and Planet Radius (detection of potential eclipsing binaries).
- Relationship between Transit Duration and Period (orbital geometry).

Provide a 3-paragraph explanation:
1. **Object Nature Analysis**: Based on size, temperature, and flux, what class of object is it? (Rocky, Gas Giant, Super-Earth?).
2. **Parameter Correlation**: Explain why the combination of these specific data points adds or subtracts credibility to the find according to orbital physics.
3. **Model Verdict**: Interpret the ML confidence in the context of common false positives (like background eclipsing binaries or stellar variability).

IMPORTANT:
- DO NOT use LaTeX formulas (avoid \\frac, \\sqrt, \\simeq, etc.).
- Use simple Unicode symbols: R⊕ (Earth radius), R☉ (solar radius), ≈ (approximately), × (multiplication), ² (squared), √ (square root).
- Write equations in plain text, example: "Tdur ≈ (P/π) × (R★/a)" instead of LaTeX.
- Use Markdown for formatting: **bold**, *italics*, lists with dashes.

Use professional but didactic language.`;

    const userPrompt = isSpanish
        ? `Analiza rigurosamente este candidato a exoplaneta:

**DATOS DEL MODELO:**
- Clasificación Final: ${prediction.pred_confirmed === 1 ? "CONFIRMADO (Exoplaneta Protoprobable)" : "NO CONFIRMADO (Probable Falso Positivo)"}
- Confianza: ${(prediction.score_confirmed * 100).toFixed(2)}%
- Umbral de Sensibilidad: ${(threshold * 100).toFixed(2)}%

**PARÁMETROS FÍSICOS:**
- Periodo Orbital: ${inputData.orbital_period_days} días (Tiempo que tarda en rodear su estrella)
- Duración del Tránsito: ${inputData.transit_duration_hours} horas (Tiempo que bloquea la luz)
- Profundidad del Tránsito: ${inputData.transit_depth_ppm} ppm (Cantidad de luz bloqueada)
- Radio Planetario: ${inputData.planet_radius_earth} R⊕ (Tamaño relativo a la Tierra)
- Flujo de Insolación: ${inputData.insolation_flux_Earth} S⊕ (Energía recibida de su estrella)
- Temperatura de Equilibrio: ${inputData.equilibrium_temperature_K} K (Temperatura estimada del planeta)
- Radio de la Estrella: ${inputData.stellar_radius_solar} R☉ (Tamaño de la estrella anfitriona)
- Temperatura de la Estrella: ${inputData.stellar_temperature_K} K (Tipo espectral de la estrella)

Por favor, realiza un análisis de correlación científica detallado.`
        : `Rigorously analyze this exoplanet candidate:

**MODEL DATA:**
- Final Classification: ${prediction.pred_confirmed === 1 ? "CONFIRMED (Likely Exoplanet)" : "NOT CONFIRMED (Likely False Positive)"}
- Confidence: ${(prediction.score_confirmed * 100).toFixed(2)}%
- Sensitivity Threshold: ${(threshold * 100).toFixed(2)}%

**PHYSICAL PARAMETERS:**
- Orbital Period: ${inputData.orbital_period_days} days (Time to orbit its star)
- Transit Duration: ${inputData.transit_duration_hours} hours (Time it blocks the light)
- Transit Depth: ${inputData.transit_depth_ppm} ppm (Amount of light blocked)
- Planetary Radius: ${inputData.planet_radius_earth} R⊕ (Size relative to Earth)
- Insolation Flux: ${inputData.insolation_flux_Earth} S⊕ (Energy received from its star)
- Equilibrium Temperature: ${inputData.equilibrium_temperature_K} K (Estimated planet temperature)
- Star Radius: ${inputData.stellar_radius_solar} R☉ (Size of the host star)
- Star Temperature: ${inputData.stellar_temperature_K} K (Spectral type of the star)

Please perform a detailed scientific correlation analysis.`;

    try {
        interface GroqResponse {
            choices: Array<{
                message: {
                    content: string;
                };
            }>;
        }

        const response = await $fetch<GroqResponse>("https://api.groq.com/openai/v1/chat/completions", {
            method: "POST",
            headers: {
                Authorization: `Bearer ${config.groqApiKey}`,
                "Content-Type": "application/json",
            },
            body: {
                model: "openai/gpt-oss-120b",
                messages: [
                    { role: "system", content: systemPrompt },
                    { role: "user", content: userPrompt },
                ],
                max_tokens: 1536,
                temperature: 0.5,
            },
        });

        return {
            interpretation: response.choices[0]?.message?.content || "No interpretation available",
        };
    } catch (error: any) {
        console.error("Groq API Error:", error);
        throw createError({
            statusCode: 500,
            message: error.message || "Failed to get interpretation from AI",
        });
    }
});
