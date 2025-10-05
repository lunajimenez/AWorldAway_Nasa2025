# unified_exoplanet_ml.py
# -*- coding: utf-8 -*-
"""
Pipeline Unificado de ML para Exoplanetas - NASA Space Apps 2025
Combina feature engineering astronómico + selección automática + ensemble
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier, 
    StackingClassifier,
    VotingClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    average_precision_score,
    classification_report, 
    confusion_matrix,
    precision_recall_curve, 
    roc_auc_score,
    f1_score,
    balanced_accuracy_score
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_predict,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Opcional pero recomendado
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("Warning: LightGBM not installed. Using fallback models.")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not installed. Using fallback models.")

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

# Constantes astronómicas
R_SUN_TO_RE = 109.2          # radios terrestres por radio solar
AU_PER_R_SUN = 0.00465047    # AU por radio solar

# Features base esperadas (de tu modelo original)
FEATURES_UNIFIED_BASE = [
    "orbital_period_days",
    "transit_duration_hours", 
    "transit_depth_ppm",
    "planet_radius_earth",
    "equilibrium_temperature_K",
    "insolation_flux_Earth",
    "stellar_radius_solar",
    "stellar_temperature_K",
]

# Mapeo de etiquetas (del modelo del compañero)
LABEL_ALIASES = {
    "PC": "CANDIDATE",
    "KP": "CANDIDATE", 
    "APC": "CANDIDATE",
    "CP": "CONFIRMED",
    "FP": "FALSE_POSITIVE",
    "FALSE POSITIVE": "FALSE_POSITIVE",
    "REFUTED": "FALSE_POSITIVE",
    "FA": "FALSE_POSITIVE",
}

# Columnas a excluir siempre
EXCLUDE_ALWAYS = {
    "final_disposition", "final_disposition_raw", "disp_label", 
    "y_confirmed", "y_multiclass", "rowid", "id", "idx", "index"
}

# ============================================================================
# FEATURE ENGINEERING ASTRONÓMICO (Tu código mejorado)
# ============================================================================

def add_astronomy_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Feature engineering astronómico avanzado
    Combina tus features originales + nuevas features astronómicas
    """
    if verbose:
        print("[ASTRONOMY] Adding advanced astronomical features...")
    
    df = df.copy()
    new_features = []
    
    # 1. RATIOS FÍSICOS FUNDAMENTALES
    if all(c in df.columns for c in ['planet_radius_earth', 'stellar_radius_solar']):
        # Ratio planeta/estrella (relacionado con transit_depth)
        stellar_radius_earth = df['stellar_radius_solar'] * R_SUN_TO_RE
        df['planet_star_radius_ratio'] = df['planet_radius_earth'] / stellar_radius_earth
        new_features.append('planet_star_radius_ratio')
        
        # Transit depth teórico
        df['transit_depth_theoretical'] = (df['planet_star_radius_ratio'] ** 2) * 1e6
        new_features.append('transit_depth_theoretical')
        
        # Discrepancia depth (detector de falsos positivos)
        if 'transit_depth_ppm' in df.columns:
            df['depth_discrepancy'] = np.abs(
                df['transit_depth_ppm'] - df['transit_depth_theoretical']
            )
            df['depth_ratio'] = df['transit_depth_ppm'] / (df['transit_depth_theoretical'] + 1e-10)
            new_features.extend(['depth_discrepancy', 'depth_ratio'])
    
    # 2. SIGNAL-TO-NOISE RATIO DEL TRÁNSITO
    if all(c in df.columns for c in ['transit_depth_ppm', 'transit_duration_hours']):
        df['transit_snr'] = df['transit_depth_ppm'] / np.sqrt(df['transit_duration_hours'] + 1)
        new_features.append('transit_snr')
    
    # 3. PARÁMETRO DE IMPACTO ESTIMADO
    if all(c in df.columns for c in ['transit_duration_hours', 'orbital_period_days']):
        duration_fraction = df['transit_duration_hours'] / (df['orbital_period_days'] * 24)
        df['impact_parameter_proxy'] = np.sqrt(1 - np.minimum(duration_fraction * np.pi, 1))
        df['duration_period_ratio'] = duration_fraction
        new_features.extend(['impact_parameter_proxy', 'duration_period_ratio'])
    
    # 4. LUMINOSIDAD Y DENSIDAD ESTELAR
    if all(c in df.columns for c in ['stellar_radius_solar', 'stellar_temperature_K']):
        df['stellar_luminosity'] = (
            (df['stellar_radius_solar'] ** 2) * 
            ((df['stellar_temperature_K'] / 5778) ** 4)
        )
        df['stellar_density'] = (
            ((df['stellar_temperature_K'] / 5778) ** 4) / 
            (df['stellar_radius_solar'] ** 3 + 1e-10)
        )
        new_features.extend(['stellar_luminosity', 'stellar_density'])
    
    # 5. ZONA HABITABLE AVANZADA
    if all(c in df.columns for c in ['insolation_flux_Earth', 'stellar_temperature_K']):
        # Zona habitable ajustada por tipo estelar
        hz_inner = 0.95 * np.sqrt(df['stellar_temperature_K'] / 5778)
        hz_outer = 1.37 * np.sqrt(df['stellar_temperature_K'] / 5778)
        
        df['hz_score'] = np.exp(
            -((df['insolation_flux_Earth'] - 1.0) ** 2) / (0.5 ** 2)
        )
        df['in_habitable_zone'] = (
            (df['insolation_flux_Earth'] >= hz_inner * 0.5) & 
            (df['insolation_flux_Earth'] <= hz_outer * 2.0)
        ).astype(int)
        new_features.extend(['hz_score', 'in_habitable_zone'])
    
    # 6. LOGARITMOS (normalizan distribuciones)
    log_cols = [
        'orbital_period_days', 'transit_depth_ppm', 
        'planet_radius_earth', 'transit_duration_hours'
    ]
    for col in log_cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log10(df[col] + 1e-10)
            new_features.append(f'log_{col}')
    
    # 7. INTERACCIONES NO LINEALES
    if all(c in df.columns for c in ['transit_depth_ppm', 'transit_duration_hours']):
        df['depth_duration_product'] = (
            np.log1p(df['transit_depth_ppm']) * 
            np.log1p(df['transit_duration_hours'])
        )
        new_features.append('depth_duration_product')
    
    if all(c in df.columns for c in ['equilibrium_temperature_K', 'planet_radius_earth']):
        df['temp_radius_interaction'] = (
            df['equilibrium_temperature_K'] * 
            df['planet_radius_earth']
        )
        new_features.append('temp_radius_interaction')
    
    # 8. CATEGORÍAS DE PLANETAS
    if 'planet_radius_earth' in df.columns:
        df['planet_type'] = pd.cut(
            df['planet_radius_earth'],
            bins=[0, 1.0, 1.75, 4.0, 15.0, 100],
            labels=[0, 1, 2, 3, 4]  # Earth, Super-Earth, Neptune, Jupiter, Super-Jupiter
        ).cat.codes
        new_features.append('planet_type')
    
    # 9. PROBABILIDAD GEOMÉTRICA DE TRÁNSITO
    if all(c in df.columns for c in ['stellar_radius_solar', 'orbital_period_days']):
        # Kepler's third law approximation
        semi_major_axis = (df['orbital_period_days'] / 365.25) ** (2/3)  # en AU
        df['transit_probability'] = (
            df['stellar_radius_solar'] * 0.00465047 / semi_major_axis
        )
        new_features.append('transit_probability')
    
    if verbose:
        print(f"[ASTRONOMY] Added {len(new_features)} astronomical features")
        
    return df

# ============================================================================
# PREPROCESAMIENTO INTELIGENTE (Del compañero mejorado)
# ============================================================================

def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura que las etiquetas estén correctamente formateadas"""
    out = df.copy()
    
    if "final_disposition" in out.columns:
        disp = out["final_disposition"].astype(str).str.upper()
    elif "final_disposition_raw" in out.columns:
        disp = out["final_disposition_raw"].astype(str).str.upper()
        disp = disp.replace(LABEL_ALIASES)
    else:
        disp = pd.Series(["UNKNOWN"] * len(out), index=out.index)
    
    out["disp_label"] = disp
    out["y_confirmed"] = (disp == "CONFIRMED").astype(int)
    
    # También preparar multiclass
    label_map = {"CONFIRMED": 0, "CANDIDATE": 1, "FALSE_POSITIVE": 2, "UNKNOWN": -1}
    out["y_multiclass"] = disp.map(label_map).fillna(-1).astype(int)
    
    return out

def split_feature_types(
    df: pd.DataFrame,
    exclude_cols: List[str],
    auto_drop_ids: bool = True,
    id_uniqueness_ratio: float = 0.9,
    missing_threshold: float = 0.98
) -> Tuple[List[str], List[str], List[str]]:
    """
    Separa columnas en numéricas y categóricas, con filtros inteligentes
    """
    cols = [c for c in df.columns if c not in exclude_cols and c not in EXCLUDE_ALWAYS]
    num_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cols if c not in num_cols]
    dropped_cols = []
    
    # Eliminar columnas tipo ID (alta cardinalidad)
    if auto_drop_ids and len(df) > 0:
        to_drop = []
        for c in cat_cols:
            n_unique = df[c].nunique(dropna=True)
            if n_unique / max(1, len(df)) >= id_uniqueness_ratio:
                to_drop.append(c)
        if to_drop:
            dropped_cols.extend(to_drop)
            cat_cols = [c for c in cat_cols if c not in to_drop]
    
    # Filtrar columnas con demasiados NaN
    def drop_too_missing(candidates: List[str], threshold=missing_threshold):
        keep, drop = [], []
        na_frac = df[candidates].isna().mean()
        for c in candidates:
            if na_frac[c] >= threshold:
                drop.append(c)
            else:
                keep.append(c)
        return keep, drop
    
    num_cols, drop_n = drop_too_missing(num_cols)
    cat_cols, drop_c = drop_too_missing(cat_cols)
    dropped_cols.extend(drop_n + drop_c)
    
    # Filtrar columnas constantes
    def drop_constants(candidates: List[str]):
        keep, drop = [], []
        for c in candidates:
            vals = df[c].dropna().unique()
            if len(vals) <= 1:
                drop.append(c)
            else:
                keep.append(c)
        return keep, drop
    
    num_cols, drop_n2 = drop_constants(num_cols)
    cat_cols, drop_c2 = drop_constants(cat_cols)
    dropped_cols.extend(drop_n2 + drop_c2)
    
    print(f"[FEATURES] Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}, Dropped: {len(dropped_cols)}")
    
    return num_cols, cat_cols, dropped_cols

# ============================================================================
# CONSTRUCCIÓN DE MODELOS Y ENSEMBLE
# ============================================================================

def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Pipeline de preprocesamiento robusto"""
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())  # Añadido scaling
    ])
    
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    
    return preprocessor

def build_ensemble_model(
    task: str = "binary",
    random_state: int = 42
) -> Pipeline:
    """
    Construye un ensemble robusto de múltiples modelos
    """
    models = []
    
    # 1. LightGBM (tu modelo optimizado)
    if HAS_LGBM:
        lgbm = LGBMClassifier(
            n_estimators=2500,
            learning_rate=0.02,
            max_depth=12,
            num_leaves=127,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=1.0,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1
        )
        models.append(('lgbm', lgbm))
    
    # 2. RandomForest (mejorado del compañero)
    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    models.append(('rf', rf))
    
    # 3. XGBoost
    if HAS_XGB:
        xgb_model = xgb.XGBClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        models.append(('xgb', xgb_model))
    
    # 4. Crear ensemble
    if len(models) > 1:
        # Opción A: Stacking (mejor para combinar fortalezas)
        ensemble = StackingClassifier(
            estimators=models,
            final_estimator=LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=random_state
            ),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
    else:
        # Si solo hay un modelo, usarlo directamente
        ensemble = models[0][1]
    
    return ensemble

def compute_best_threshold(
    y_true: np.ndarray, 
    probs: np.ndarray,
    metric: str = "f1"
) -> float:
    """
    Calcula el mejor threshold optimizando F1 o balanced accuracy
    """
    prec, rec, thresholds = precision_recall_curve(y_true, probs)
    
    if len(thresholds) == 0:
        return 0.5
        
    if metric == "f1":
        f1s = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
        best_idx = int(np.nanargmax(f1s))
    else:  # balanced_accuracy
        bal_acc = []
        for thr in thresholds:
            preds = (probs >= thr).astype(int)
            bal_acc.append(balanced_accuracy_score(y_true, preds))
        best_idx = int(np.argmax(bal_acc))
    
    return float(thresholds[best_idx])

def compute_cv_threshold(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5
) -> float:
    """
    Calcula threshold robusto usando cross-validation
    """
    cv_probs = cross_val_predict(
        model, X, y, 
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        method='predict_proba',
        n_jobs=-1
    )
    
    if len(cv_probs.shape) > 1:
        cv_probs = cv_probs[:, 1]
    
    return compute_best_threshold(y, cv_probs, metric="f1")

# ============================================================================
# PIPELINE PRINCIPAL DE ENTRENAMIENTO
# ============================================================================

def train_unified_pipeline(
    csv_path: Path,
    outdir: Path,
    test_size: float = 0.2,
    seed: int = 42,
    task: str = "binary",
    use_astronomy: bool = True,
    use_all_features: bool = True,
    exclude_cols: List[str] = None,
    auto_drop_ids: bool = True
) -> Dict:
    """
    Pipeline principal que combina lo mejor de ambos modelos
    """
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar datos
    print(f"\n{'='*60}")
    print("UNIFIED EXOPLANET ML PIPELINE")
    print(f"{'='*60}")
    print(f"[DATA] Loading {csv_path}...")
    
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[DATA] Shape: {df.shape}")
    
    # 2. Preparar etiquetas
    df = ensure_labels(df)
    
    # 3. Aplicar feature engineering astronómico
    if use_astronomy:
        df = add_astronomy_features(df, verbose=True)
    
    # 4. Filtrar datos etiquetados
    if task == "binary":
        labeled = df[df["disp_label"].isin(["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"])].copy()
        y = labeled["y_confirmed"].values
        class_names = ["NOT_CONFIRMED", "CONFIRMED"]
    else:
        labeled = df[df["y_multiclass"] >= 0].copy()
        y = labeled["y_multiclass"].values
        class_names = ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]
    
    if len(labeled) == 0:
        raise ValueError("No labeled data found!")
    
    print(f"[DATA] Labeled samples: {len(labeled)}")
    print(f"[DATA] Class distribution:\n{pd.Series(y).value_counts()}")
    
    # 5. Selección de features
    if exclude_cols is None:
        exclude_cols = []
    
    if use_all_features:
        # Usar todas las columnas disponibles (como el compañero)
        num_cols, cat_cols, dropped_cols = split_feature_types(
            labeled, 
            exclude_cols=exclude_cols,
            auto_drop_ids=auto_drop_ids
        )
    else:
        # Usar solo las features seleccionadas manualmente
        num_cols = [c for c in FEATURES_UNIFIED_BASE if c in labeled.columns]
        cat_cols = ['source_mission'] if 'source_mission' in labeled.columns else []
        dropped_cols = []
    
    if not num_cols and not cat_cols:
        raise ValueError("No features available!")
    
    # 6. Preparar X
    X = labeled[num_cols + cat_cols]
    
    # 7. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    print(f"\n[SPLIT] Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 8. Construir pipeline completo
    print(f"\n[MODEL] Building ensemble pipeline...")
    
    preprocessor = build_preprocessor(num_cols, cat_cols)
    ensemble = build_ensemble_model(task=task, random_state=seed)
    
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("ensemble", ensemble)
    ])
    
    # 9. Entrenar
    print(f"[TRAIN] Training ensemble model...")
    pipe.fit(X_train, y_train)
    
    # 10. Calcular threshold óptimo
    print(f"[OPTIMIZE] Computing optimal threshold...")
    
    if task == "binary":
        # Usar validación para threshold robusto
        best_threshold = compute_cv_threshold(pipe, X_train, y_train, cv=5)
        print(f"[OPTIMIZE] Best threshold: {best_threshold:.4f}")
    else:
        best_threshold = None
    
    # 11. Evaluar en test
    print(f"\n[EVALUATE] Testing model...")
    
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    
    if task == "binary" and best_threshold is not None:
        y_pred_opt = (y_proba[:, 1] >= best_threshold).astype(int)
        y_proba_binary = y_proba[:, 1]
    else:
        y_pred_opt = y_pred
        y_proba_binary = None
    
    # 12. Métricas
    results = {}
    
    # Métricas básicas
    results["accuracy"] = accuracy_score(y_test, y_pred_opt)
    results["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred_opt)
    results["f1_macro"] = f1_score(y_test, y_pred_opt, average='macro')
    
    # ROC-AUC
    if task == "binary" and y_proba_binary is not None:
        results["roc_auc"] = roc_auc_score(y_test, y_proba_binary)
        results["pr_auc"] = average_precision_score(y_test, y_proba_binary)
    elif task == "multiclass":
        results["roc_auc"] = roc_auc_score(y_test, y_proba, multi_class='ovr')
    
    # Report detallado
    report = classification_report(
        y_test, y_pred_opt,
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_opt)
    
    # 13. Imprimir resultados
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:          {results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"F1-Macro:          {results['f1_macro']:.4f}")
    if 'roc_auc' in results:
        print(f"ROC-AUC:           {results['roc_auc']:.4f}")
    if 'pr_auc' in results:
        print(f"PR-AUC:            {results['pr_auc']:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_opt, target_names=class_names))
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # 14. Guardar modelo y resultados
    print(f"\n[SAVE] Saving model and results...")
    
    # Modelo
    dump(pipe, outdir / "unified_model.joblib")
    
    # Configuración
    config = {
        "source_csv": str(csv_path),
        "task": task,
        "use_astronomy": use_astronomy,
        "use_all_features": use_all_features,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "dropped_columns": dropped_cols,
        "threshold": best_threshold,
        "test_metrics": results,
        "random_state": seed,
        "test_size": test_size
    }
    
    (outdir / "model_config.json").write_text(
        json.dumps(config, indent=2), 
        encoding="utf-8"
    )
    
    # Métricas detalladas
    detailed_metrics = {
        **results,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_features_numeric": len(num_cols),
        "n_features_categorical": len(cat_cols),
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test)
    }
    
    (outdir / "metrics_detailed.json").write_text(
        json.dumps(detailed_metrics, indent=2),
        encoding="utf-8"
    )
    
    # Report de texto
    report_text = f"""
UNIFIED EXOPLANET ML PIPELINE - RESULTS
{'='*60}
Dataset: {csv_path}
Task: {task}
Features: {len(num_cols)} numeric, {len(cat_cols)} categorical

Test Results:
- Accuracy: {results['accuracy']:.4f}
- Balanced Accuracy: {results['balanced_accuracy']:.4f}
- F1-Macro: {results['f1_macro']:.4f}
{f"- ROC-AUC: {results['roc_auc']:.4f}" if 'roc_auc' in results else ""}
{f"- PR-AUC: {results['pr_auc']:.4f}" if 'pr_auc' in results else ""}
{f"- Optimal Threshold: {best_threshold:.4f}" if best_threshold else ""}

Confusion Matrix:
{cm}

Classification Report:
{classification_report(y_test, y_pred_opt, target_names=class_names)}
"""
    
    (outdir / "report.txt").write_text(report_text, encoding="utf-8")
    
    # 15. Score del dataset completo
    print(f"\n[SCORE] Scoring full dataset...")
    
    X_full = labeled[num_cols + cat_cols]
    y_proba_full = pipe.predict_proba(X_full)
    
    if task == "binary" and best_threshold:
        y_pred_full = (y_proba_full[:, 1] >= best_threshold).astype(int)
        score_col = y_proba_full[:, 1]
    else:
        y_pred_full = pipe.predict(X_full)
        score_col = y_proba_full.max(axis=1)
    
    scored = labeled.copy()
    scored["score"] = score_col
    scored["prediction"] = y_pred_full
    scored.to_csv(outdir / "dataset_scored.csv", index=False)
    
    # Ranking de candidatos
    if task == "binary":
        candidates = scored[scored["disp_label"] == "CANDIDATE"].sort_values(
            "score", ascending=False
        )
        candidates.to_csv(outdir / "candidates_ranked.csv", index=False)
        print(f"[SAVE] Top candidates saved to candidates_ranked.csv")
    
    print(f"\n[COMPLETE] All results saved to {outdir}")
    
    return results

# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Exoplanet ML Pipeline - NASA Space Apps 2025"
    )
    parser.add_argument(
        "--csv", 
        type=str, 
        required=True,
        help="Path to harmonized CSV file"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./outputs_unified",
        help="Output directory"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["binary", "multiclass"],
        default="binary",
        help="Classification task type"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--no-astronomy",
        action="store_true",
        help="Disable astronomical feature engineering"
    )
    parser.add_argument(
        "--manual-features",
        action="store_true", 
        help="Use only manually selected features (not all available)"
    )
    parser.add_argument(
        "--exclude-cols",
        type=str,
        default="",
        help="Comma-separated list of columns to exclude"
    )
    parser.add_argument(
        "--no-auto-drop",
        action="store_true",
        help="Disable automatic dropping of high-cardinality columns"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    exclude_cols = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
    
    results = train_unified_pipeline(
        csv_path=Path(args.csv),
        outdir=Path(args.out),
        test_size=args.test_size,
        seed=args.seed,
        task=args.task,
        use_astronomy=not args.no_astronomy,
        use_all_features=not args.manual_features,
        exclude_cols=exclude_cols,
        auto_drop_ids=not args.no_auto_drop
    )
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Final F1-Macro: {results['f1_macro']:.4f}")
    if 'roc_auc' in results:
        print(f"Final ROC-AUC: {results['roc_auc']:.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()