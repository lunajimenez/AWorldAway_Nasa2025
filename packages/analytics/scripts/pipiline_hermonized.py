
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NASA Space Apps 2025 - Exoplanet Classification Pipeline
Dataset Unificado (KOI + TOI + K2)
Optimizado para 17,263 exoplanetas
"""
import argparse
import os
import sys
from typing import List, Tuple, Optional
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier

try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except Exception:
    _HAS_SMOTE = False

_HAS_LGBM = True
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except Exception:
    _HAS_LGBM = False

import matplotlib.pyplot as plt


# ============================================================================
# FEATURES PARA DATASET UNIFICADO
# ============================================================================
FEATURES_UNIFIED = [
    "orbital_period_days",
    "transit_duration_hours",
    "transit_depth_ppm",
    "planet_radius_earth",
    "equilibrium_temperature_K",
    "insolation_flux_Earth",
    "stellar_radius_solar",
    "stellar_temperature_K",
]

# Feature categórica opcional
CATEGORICAL_FEATURES = ["source_mission"]  # KEPLER, TESS

TARGET_DEFAULT = "final_disposition"  # CONFIRMED, FALSE_POSITIVE, CANDIDATE


# ============================================================================
# CONFIGURACIÓN DE LIMPIEZA DE DATOS
# ============================================================================
CLEANING_RULES = {
    # [min, max] - valores fuera de este rango se eliminan
    "orbital_period_days": [0.1, 10000],           # 2.4h a ~27 años
    "transit_duration_hours": [0.1, 24],           # 6min a 1 día
    "transit_depth_ppm": [1, 100000],              # 1ppm a 10%
    "planet_radius_earth": [0.1, 30],              # Super-Tierra a Júpiter grande
    "equilibrium_temperature_K": [50, 5000],       # Muy frío a muy caliente
    "insolation_flux_Earth": [0.01, 10000],        # Muy lejos a muy cerca
    "stellar_radius_solar": [0.1, 50],             # Enana a Gigante
    "stellar_temperature_K": [2000, 50000],        # Enana roja a estrella azul
}


def ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def clean_anomalies(df: pd.DataFrame, rules: dict, verbose: bool = True) -> pd.DataFrame:
    """Limpia valores anómalos y outliers extremos"""
    initial_rows = len(df)
    
    for col, (min_val, max_val) in rules.items():
        if col in df.columns:
            before = len(df)
            mask = (df[col] >= min_val) & (df[col] <= max_val)
            df = df[mask].copy()
            removed = before - len(df)
            if removed > 0 and verbose:
                print(f"[CLEAN] {col}: removed {removed} outliers (range: [{min_val}, {max_val}])")
    
    total_removed = initial_rows - len(df)
    if verbose:
        print(f"[CLEAN] Total rows removed: {total_removed} ({100*total_removed/initial_rows:.2f}%)")
        print(f"[CLEAN] Rows remaining: {len(df)}")
    
    return df


def add_feature_engineering(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Feature engineering astronómico avanzado
    Crea features derivadas con alto poder predictivo
    """
    if verbose:
        print("[FEAT-ENG] Creating derived features...")
    
    new_features = []
    
    # 1. RATIOS FÍSICOS (muy importantes)
    if 'planet_radius_earth' in df.columns and 'stellar_radius_solar' in df.columns:
        # Ratio planeta/estrella (relacionado con transit_depth)
        df['planet_star_radius_ratio'] = df['planet_radius_earth'] / (df['stellar_radius_solar'] * 109.2)
        new_features.append('planet_star_radius_ratio')
        
        # Transit depth teórico (para detectar inconsistencias)
        df['transit_depth_theoretical'] = (df['planet_star_radius_ratio'] ** 2) * 1e6  # en ppm
        new_features.append('transit_depth_theoretical')
        
        # Diferencia entre depth observado y teórico (detector de falsos positivos)
        if 'transit_depth_ppm' in df.columns:
            df['depth_discrepancy'] = np.abs(df['transit_depth_ppm'] - df['transit_depth_theoretical'])
            new_features.append('depth_discrepancy')
    
    # 2. LOGARITMOS (normalizan distribuciones asimétricas)
    log_cols = ['orbital_period_days', 'transit_depth_ppm', 'planet_radius_earth']
    for col in log_cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log10(df[col] + 1e-10)
            new_features.append(f'log_{col}')
    
    # 3. INTERACCIONES IMPORTANTES
    if 'orbital_period_days' in df.columns and 'transit_duration_hours' in df.columns:
        # Ratio duración/período (geometría del tránsito)
        df['duration_period_ratio'] = df['transit_duration_hours'] / (df['orbital_period_days'] * 24)
        new_features.append('duration_period_ratio')
    
    # 4. ZONA HABITABLE (relevante para clasificación)
    if 'insolation_flux_Earth' in df.columns:
        # Clasificador simple de zona habitable (0.5 a 2.0 veces la Tierra)
        df['in_habitable_zone'] = ((df['insolation_flux_Earth'] >= 0.5) & 
                                   (df['insolation_flux_Earth'] <= 2.0)).astype(int)
        new_features.append('in_habitable_zone')
    
    # 5. CATEGORÍAS DE TAMAÑO
    if 'planet_radius_earth' in df.columns:
        df['planet_size_category'] = pd.cut(
            df['planet_radius_earth'],
            bins=[0, 1.25, 2.0, 4.0, 30],
            labels=[0, 1, 2, 3],
            include_lowest=True
        )
        # Manejar NaN antes de convertir a int
        df['planet_size_category'] = df['planet_size_category'].cat.codes
        df['planet_size_category'] = df['planet_size_category'].replace(-1, np.nan)
        new_features.append('planet_size_category')
    
    # 6. TEMPERATURA CATEGORIZADA
    if 'equilibrium_temperature_K' in df.columns:
        df['temp_category'] = pd.cut(
            df['equilibrium_temperature_K'],
            bins=[0, 200, 400, 1000, 5000],
            labels=[0, 1, 2, 3],
            include_lowest=True
        )
        # Manejar NaN antes de convertir a int
        df['temp_category'] = df['temp_category'].cat.codes
        df['temp_category'] = df['temp_category'].replace(-1, np.nan)
        new_features.append('temp_category')
    
    if verbose:
        print(f"[FEAT-ENG] Added {len(new_features)} new features:")
        for feat in new_features:
            print(f"  - {feat}")
    
    return df, new_features


def handle_missing_values(df: pd.DataFrame, strategy: str = "median", verbose: bool = True) -> pd.DataFrame:
    """
    Maneja valores faltantes de forma inteligente
    Strategy: 'drop', 'median', 'mean', 'knn'
    """
    missing_before = df.isnull().sum()
    missing_cols = missing_before[missing_before > 0]
    
    if len(missing_cols) == 0:
        if verbose:
            print("[MISSING] No missing values found")
        return df
    
    if verbose:
        print(f"[MISSING] Found missing values in {len(missing_cols)} columns:")
        for col, count in missing_cols.items():
            pct = 100 * count / len(df)
            print(f"  - {col}: {count} ({pct:.2f}%)")
    
    if strategy == "drop":
        df = df.dropna()
        if verbose:
            print(f"[MISSING] Dropped rows with NaN. New shape: {df.shape}")
    
    elif strategy == "median":
        for col in missing_cols.index:
            if df[col].dtype in ['float64', 'int64']:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                if verbose:
                    print(f"[MISSING] Filled {col} with median: {median_val:.2f}")
    
    elif strategy == "mean":
        for col in missing_cols.index:
            if df[col].dtype in ['float64', 'int64']:
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                if verbose:
                    print(f"[MISSING] Filled {col} with mean: {mean_val:.2f}")
    
    return df


def encode_categorical(df: pd.DataFrame, cat_cols: List[str], verbose: bool = True) -> pd.DataFrame:
    """One-hot encoding para features categóricas"""
    encoded_features = []
    
    for col in cat_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            encoded_features.extend(dummies.columns.tolist())
            df = df.drop(columns=[col])
            if verbose:
                print(f"[ENCODE] One-hot encoded '{col}' → {len(dummies.columns)} new columns")
    
    return df, encoded_features


def load_and_prepare(
    csv_path: str,
    target_col: str,
    use_binary: bool,
    positive_label: str,
    use_categorical: bool = True,
    use_feature_engineering: bool = True,
    missing_strategy: str = "median",
    clean_data: bool = True,
    sep: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    
    # Load data
    if sep:
        df = pd.read_csv(csv_path, sep=sep)
    else:
        df = pd.read_csv(csv_path)
    
    print(f"[INFO] Loaded data shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    # Verify target exists
    if target_col not in df.columns:
        raise KeyError(f"Target '{target_col}' not found. Available: {list(df.columns)}")
    
    # Clean anomalies
    if clean_data:
        df = clean_anomalies(df, CLEANING_RULES, verbose=True)
    
    # Handle missing values BEFORE feature engineering
    df = handle_missing_values(df, strategy=missing_strategy, verbose=True)
    
    # Feature engineering
    new_features = []
    if use_feature_engineering:
        df, new_features = add_feature_engineering(df, verbose=True)
    
    # Categorical encoding
    encoded_features = []
    if use_categorical:
        df, encoded_features = encode_categorical(df, CATEGORICAL_FEATURES, verbose=True)
    
    # Build final feature list
    features = FEATURES_UNIFIED.copy()
    features += new_features
    features += encoded_features
    
    # Keep only existing features
    features = [f for f in features if f in df.columns]
    
    # Separate X and y
    X = df[features].copy()
    y = df[target_col].astype(str)
    
    print(f"\n[INFO] Final feature count: {len(features)}")
    print(f"[INFO] Features: {features}")
    
    # Binary or multiclass
    if use_binary:
        y = (y == positive_label).astype(int)
        classes_info = f"Binary: '{positive_label}' vs others"
        class_names = ["negative", "positive"]
    else:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)
        class_names = list(le.classes_)
        classes_info = f"Multiclass: {class_names}"
    
    print(f"[INFO] {classes_info}")
    print(f"[INFO] Class distribution:\n{y.value_counts()}")
    print(f"[INFO] Final X shape: {X.shape}")
    
    return X, y, class_names, features


def build_model(random_state: int = 42, n_samples: int = 20000):
    """
    Modelo optimizado para ~17K samples y 3 clases balanceadas
    HIPERPARÁMETROS MEJORADOS PARA MAYOR PRECISIÓN
    """
    if _HAS_LGBM:
        print("[INFO] Using LightGBM (optimized)")
        model = LGBMClassifier(
            n_estimators=2500,          # OPTIMIZADO: de 1500 a 2500
            learning_rate=0.02,         # OPTIMIZADO: de 0.03 a 0.02
            max_depth=12,               # OPTIMIZADO: de 8 a 12
            num_leaves=127,             # OPTIMIZADO: de 63 a 127
            min_child_samples=15,       # OPTIMIZADO: de 20 a 15
            subsample=0.8,              # OPTIMIZADO: de 0.85 a 0.8
            colsample_bytree=0.8,       # OPTIMIZADO: de 0.85 a 0.8
            reg_alpha=0.3,              # OPTIMIZADO: de 0.1 a 0.3
            reg_lambda=1.0,             # OPTIMIZADO: de 0.5 a 1.0
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        print("[INFO] Using RandomForest (fallback)")
        model = RandomForestClassifier(
            n_estimators=800,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state,
        )
    return model


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, folds: int = 10) -> float:
    """CV con más folds (tienes 17K samples)"""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"[CV] F1-macro: mean={scores.mean():.4f} ± {scores.std():.4f}")
    print(f"[CV] Individual folds: {[f'{s:.4f}' for s in scores]}")
    return scores.mean()


def train_and_evaluate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    class_names: List[str],
    feature_names: List[str],
    test_size: float = 0.2,
    use_smote: bool = False,
    outdir: str = "outputs",
    random_state: int = 42,
):
    ensure_dirs([outdir, os.path.join(outdir, "figs"), os.path.join(outdir, "reports")])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"\n[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"[INFO] Train distribution:\n{pd.Series(y_train).value_counts()}")

    if use_smote:
        if not _HAS_SMOTE:
            print("[WARN] SMOTE not available, skipping", file=sys.stderr)
        else:
            print("[INFO] Applying SMOTE...")
            sm = SMOTE(random_state=random_state)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print(f"[INFO] After SMOTE: {len(X_train)} samples")

    # Train
    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    has_proba = hasattr(model, "predict_proba")
    y_proba = model.predict_proba(X_test) if has_proba else None

    # Metrics
    rep = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(rep)
    print(f"Balanced Accuracy: {bal_acc:.4f}")

    # ROC-AUC
    roc_msg = ""
    try:
        if y_proba is not None:
            if len(class_names) > 2:
                roc = roc_auc_score(y_test, y_proba, multi_class="ovr")
            else:
                roc = roc_auc_score(y_test, y_proba[:, 1])
            roc_msg = f"ROC-AUC: {roc:.4f}"
            print(roc_msg)
    except Exception as e:
        roc_msg = f"ROC-AUC not computed: {e}"
        print(roc_msg)

    # Save report
    report_path = os.path.join(outdir, "reports", "metrics.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("EXOPLANET CLASSIFICATION - FINAL RESULTS\n")
        f.write("="*60 + "\n")
        f.write(rep + "\n")
        f.write(f"Balanced Accuracy: {bal_acc:.4f}\n")
        if roc_msg:
            f.write(roc_msg + "\n")
    print(f"\n[SAVE] Metrics → {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    fig_cm = plt.figure(figsize=(8, 6))
    ax = fig_cm.add_subplot(111)
    im = ax.imshow(cm, aspect="auto", cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", 
                   color=text_color, fontsize=11, fontweight='bold')
    
    fig_cm.tight_layout()
    cm_path = os.path.join(outdir, "figs", "confusion_matrix.png")
    fig_cm.savefig(cm_path, dpi=200)
    plt.close(fig_cm)
    print(f"[SAVE] Confusion Matrix → {cm_path}")

    # Feature importances
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            raise AttributeError("No feature_importances_")
        
        indices = np.argsort(importances)[::-1]
        
        # Plot top 20
        top_n = min(20, len(indices))
        fig_imp = plt.figure(figsize=(12, 8))
        ax2 = fig_imp.add_subplot(111)
        ax2.barh(range(top_n), importances[indices[:top_n]], color='steelblue')
        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels([feature_names[i] for i in indices[:top_n]])
        ax2.set_xlabel("Importance", fontsize=12)
        ax2.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        fig_imp.tight_layout()
        
        imp_path = os.path.join(outdir, "figs", "feature_importances.png")
        fig_imp.savefig(imp_path, dpi=200)
        plt.close(fig_imp)
        print(f"[SAVE] Feature Importances → {imp_path}")

        print(f"\n[INFO] Top {min(15, len(indices))} Most Important Features:")
        for rank in range(min(15, len(indices))):
            idx = indices[rank]
            print(f"  {rank+1:2d}. {feature_names[idx]:35s} {importances[idx]:.4f}")
        
        # Save to file
        imp_report_path = os.path.join(outdir, "reports", "feature_importances.txt")
        with open(imp_report_path, "w", encoding="utf-8") as f:
            f.write("FEATURE IMPORTANCES\n")
            f.write("="*60 + "\n")
            for rank in range(len(indices)):
                idx = indices[rank]
                f.write(f"{rank+1:3d}. {feature_names[idx]:35s} {importances[idx]:.6f}\n")
        print(f"[SAVE] Feature Importances (full) → {imp_report_path}")
        
    except Exception as e:
        print(f"[WARN] Could not analyze importances: {e}", file=sys.stderr)
    
    return model


def parse_args():
    p = argparse.ArgumentParser(
        description="NASA Space Apps - Exoplanet ML Pipeline (Unified Dataset)"
    )
    p.add_argument("--csv", required=True, help="Path to unified CSV")
    p.add_argument("--target", default=TARGET_DEFAULT, 
                   help=f"Target column (default: {TARGET_DEFAULT})")
    p.add_argument("--binary", action="store_true", 
                   help="Binary classification (CONFIRMED vs rest)")
    p.add_argument("--positive", default="CONFIRMED", 
                   help="Positive label for binary mode")
    p.add_argument("--smote", action="store_true", 
                   help="Apply SMOTE (not recommended, data is balanced)")
    p.add_argument("--no-feature-engineering", action="store_true",
                   help="Disable feature engineering")
    p.add_argument("--no-categorical", action="store_true",
                   help="Don't use source_mission as feature")
    p.add_argument("--missing-strategy", default="median",
                   choices=["drop", "median", "mean"],
                   help="Strategy for missing values")
    p.add_argument("--no-clean", action="store_true",
                   help="Skip anomaly cleaning")
    p.add_argument("--test-size", type=float, default=0.2, 
                   help="Test fraction (default: 0.2)")
    p.add_argument("--outdir", default="outputs_unified", 
                   help="Output directory")
    p.add_argument("--sep", default=None, help="CSV separator")
    return p.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("NASA SPACE APPS 2025 - EXOPLANET CLASSIFICATION")
    print("Unified Dataset Pipeline (KOI + TOI)")
    print("="*60)
    print(f"Dataset: {args.csv}")
    print(f"Target: {args.target}")
    print(f"Mode: {'Binary' if args.binary else 'Multiclass'}")
    print("="*60 + "\n")
    
    # Load and prepare
    X, y, class_names, feature_names = load_and_prepare(
        csv_path=args.csv,
        target_col=args.target,
        use_binary=args.binary,
        positive_label=args.positive,
        use_categorical=not args.no_categorical,
        use_feature_engineering=not args.no_feature_engineering,
        missing_strategy=args.missing_strategy,
        clean_data=not args.no_clean,
        sep=args.sep,
    )

    # Build model
    model = build_model(random_state=42, n_samples=len(X))

    # Cross-validation
    print("\n" + "="*60)
    print("CROSS-VALIDATION (10-Fold)")
    print("="*60)
    cv_score = cross_validate_model(model, X, y, folds=10)

    # Train and evaluate
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL")
    print("="*60)
    fitted_model = train_and_evaluate(
        model=model,
        X=X,
        y=y,
        class_names=class_names,
        feature_names=feature_names,
        test_size=args.test_size,
        use_smote=args.smote,
        outdir=args.outdir,
        random_state=42,
    )

    # Save model bundle
    bundle = {
        "model": fitted_model,
        "features": feature_names,
        "class_names": class_names,
        "target": args.target,
        "binary": args.binary,
        "positive_label": args.positive,
        "cv_score": cv_score,
    }
    model_path = os.path.join(args.outdir, "exoplanet_model.joblib")
    joblib.dump(bundle, model_path)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Model saved: {model_path}")
    print(f"Reports: {os.path.join(args.outdir, 'reports')}")
    print(f"Figures: {os.path.join(args.outdir, 'figs')}")
    print(f"CV F1-macro: {cv_score:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()