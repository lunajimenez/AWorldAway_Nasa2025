import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================================
# 0. CONFIGURACIÓN DE RUTAS
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data/raw"
PROCESSED_DATA_PATH = BASE_DIR / "data/processed"

PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. CARGAR DATOS RAW
# ============================================================================
print("\n1. CARGANDO DATOS RAW...")
csv_file = RAW_DATA_PATH / "cumulative_2025.10.01_11.42.42.csv"
df = pd.read_csv(csv_file, comment='#')

print(f"✓ Dataset original: {df.shape[0]} filas x {df.shape[1]} columnas")

# ============================================================================
# 2. SELECCIONAR COLUMNAS (14 features + target, SIN koi_score)
# ============================================================================
print("\n" + "="*80)
print("2. SELECCIONANDO COLUMNAS CLAVE")
print("="*80)

# Features seleccionadas (basadas en análisis previo de importancia)
columnas_seleccionadas = [
    # Señales de Tránsito (4)
    'koi_period',           # Período orbital
    'koi_duration',         # Duración del tránsito
    'koi_depth',            # Profundidad del tránsito
    'koi_model_snr',        # Signal-to-Noise Ratio
    
    # Características Planetarias (3)
    'koi_prad',             # Radio planetario
    'koi_teq',              # Temperatura de equilibrio
    'koi_insol',            # Flujo de insolación
    
    # Características Estelares (2)
    'koi_srad',             # Radio estelar (importante!)
    'koi_steff',            # Temperatura estelar
    
    # Flags de Falso Positivo (4)
    'koi_fpflag_nt',        # No Transit-Like
    'koi_fpflag_ss',        # Stellar Eclipse
    'koi_fpflag_co',        # Centroid Offset
    'koi_fpflag_ec',        # Ephemeris Match
    
    # Target
    'koi_disposition',      # Clasificación
    
    # NOTA: koi_score EXCLUIDO (solo +0.32% mejora según experimento)
]

df_filtered = df[columnas_seleccionadas].copy()

print(f"✓ Columnas seleccionadas: {len(columnas_seleccionadas) - 1} features + target")
print(f"✓ koi_score EXCLUIDO (basado en experimento)")
print(f"\nFeatures incluidas:")
for i, col in enumerate(columnas_seleccionadas[:-1], 1):
    print(f"  {i:2}. {col}")

# ============================================================================
# 3. ANÁLISIS DE CALIDAD DE DATOS
# ============================================================================
print("\n" + "="*80)
print("3. ANÁLISIS DE CALIDAD DE DATOS")
print("="*80)

# Valores faltantes
print("\nValores faltantes por columna:")
missing_info = df_filtered.isnull().sum()
missing_pct = (missing_info / len(df_filtered) * 100).round(2)

for col in df_filtered.columns:
    if missing_info[col] > 0:
        print(f"  {col:20} → {missing_info[col]:5} valores ({missing_pct[col]:5.2f}%)")

total_missing = missing_info.sum()
print(f"\nTotal valores faltantes: {total_missing}")

# Distribución de clases
print("\nDistribución de koi_disposition:")
print(df_filtered['koi_disposition'].value_counts().sort_index())
print("\nProporción:")
print((df_filtered['koi_disposition'].value_counts(normalize=True) * 100).round(2).sort_index())

# ============================================================================
# 4. ESTRATEGIA DE LIMPIEZA (CONSERVADORA)
# ============================================================================
print("\n" + "="*80)
print("4. APLICANDO ESTRATEGIA DE LIMPIEZA CONSERVADORA")
print("="*80)

print("\n📋 ESTRATEGIA ELEGIDA: Mantener datos originales sin transformación agresiva")
print("\nRazones:")
print("  • Los outliers en datos astronómicos son información valiosa")
print("  • Winsorization agresiva causó pérdida de rendimiento (-1%)")
print("  • Enfoque conservador demostró mejor generalización")

# Opción 1: Eliminar filas con valores faltantes (más simple)
filas_completas = df_filtered.dropna().shape[0]
pct_completo = (filas_completas / len(df_filtered) * 100)

print(f"\n✓ Filas sin valores faltantes: {filas_completas} ({pct_completo:.1f}%)")

if pct_completo >= 90:
    print(f"✓ Estrategia seleccionada: ELIMINAR filas con NaN ({100-pct_completo:.1f}% pérdida)")
    df_clean = df_filtered.dropna()
else:
    print(f"⚠️  Solo {pct_completo:.1f}% filas completas, aplicando imputación KNN...")
    from sklearn.impute import KNNImputer
    
    features_numericas = [col for col in df_filtered.columns if col != 'koi_disposition']
    
    knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
    df_numeric = pd.DataFrame(
        knn_imputer.fit_transform(df_filtered[features_numericas]),
        columns=features_numericas,
        index=df_filtered.index
    )
    df_clean = df_numeric.copy()
    df_clean['koi_disposition'] = df_filtered['koi_disposition']

print(f"✓ Dataset limpio: {df_clean.shape[0]} filas x {df_clean.shape[1]} columnas")

# ============================================================================
# 5. VERIFICACIÓN DE CALIDAD
# ============================================================================
print("\n" + "="*80)
print("5. VERIFICACIÓN DE CALIDAD")
print("="*80)

# Verificar que no hay NaN
assert df_clean.isnull().sum().sum() == 0
print("✓ Sin valores faltantes")

# Verificar distribución de clases se mantuvo
print("\nDistribución final de clases:")
print(df_clean['koi_disposition'].value_counts().sort_index())

# Verificar que koi_score NO está
assert 'koi_score' not in df_clean.columns, "ERROR: koi_score no debería estar"
print("✓ koi_score correctamente excluido")

# Estadísticas básicas
print("\nEstadísticas de algunas features clave:")
print(df_clean[['koi_period', 'koi_depth', 'koi_prad', 'koi_teq']].describe().round(2))

# ============================================================================
# 6. GUARDAR DATASET
# ============================================================================
print("\n" + "="*80)
print("6. GUARDANDO DATASET FINAL")
print("="*80)

output_file = PROCESSED_DATA_PATH / 'KOI_All_Filtrated.csv'
df_clean.to_csv(output_file, index=False)

print(f"✓ Dataset guardado en: {output_file}")
print(f"✓ Nombre: KOI_All_Filtrated.csv")
print(f"✓ Dimensiones: {df_clean.shape[0]} filas x {df_clean.shape[1]} columnas")
print(f"✓ Features: {df_clean.shape[1] - 1} (sin contar target)")

# ============================================================================
# 7. RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)

print(f"""
📊 DATASET GENERADO: KOI_All_Filtrated.csv

CARACTERÍSTICAS:
  • Filas: {df_clean.shape[0]}
  • Columnas: {df_clean.shape[1]} (14 features + 1 target)
  • Valores faltantes: 0

""")

print("="*80)
print("✅ GENERACIÓN COMPLETADA EXITOSAMENTE")
print("="*80)

# ============================================================================
# 8. CREAR REPORTE DE METADATOS
# ============================================================================
metadata = {
    "filename": "KOI_All_Filtrated.csv",
    "source": "cumulative_2025.10.01_11.42.42.csv",
    "rows": int(df_clean.shape[0]),
    "columns": int(df_clean.shape[1]),
    "features": int(df_clean.shape[1] - 1),
    "strategy": "conservative_filtering",
    "transformations": [
        "Missing values handled (drop or KNN imputation)",
        "No winsorization (maintains original values)",
        "No feature engineering",
        "No scaling"
    ],
    "excluded_features": ["koi_score"],
    "exclusion_reason": "Only +0.32% improvement based on experiment, prefer simpler model",
    "expected_performance": {
        "ROC_AUC": "~0.9828",
        "comparison": "vs 0.9850 with koi_score (-0.22% difference)"
    },
    "features_included": [col for col in df_clean.columns if col != 'koi_disposition'],
    "class_distribution": df_clean['koi_disposition'].value_counts().to_dict(),
}

import json
metadata_file = PROCESSED_DATA_PATH / 'KOI_All_Filtrated_metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n💾 Metadata guardado en: {metadata_file}")