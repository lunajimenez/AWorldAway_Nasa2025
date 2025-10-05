#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kepler KOI ML Pipeline - Adaptado para datos filtrados
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


# FEATURES ACTUALIZADAS PARA TUS DATOS FILTRADOS
FEATURES_KOI = [
    "koi_period",      # Ya existía
    "koi_duration",    # Ya existía
    "koi_prad",        # Ya existía
    "koi_teq",         # Ya existía
    "koi_insol",       # Ya existía
    "koi_srad",        # Ya existía
    "koi_steff",       # Ya existía
    "koi_depth",       # Ya existía
    "koi_model_snr",   # Ya existía
    "koi_score",       # NUEVA - la incluimos
]

# Estas columnas las eliminaremos si existen
FLAG_COLS = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]

TARGET_DEFAULT = "koi_disposition"


def ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def pick_existing_features(df: pd.DataFrame, features: List[str]) -> List[str]:
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns will be skipped: {missing}", file=sys.stderr)
    return [c for c in features if c in df.columns]


def maybe_binary_labels(y: pd.Series, positive_label: str) -> pd.Series:
    return (y == positive_label).astype(int)


def read_table(path: str, sep: Optional[str], on_bad_lines: str = "error") -> pd.DataFrame:
    tried = []
    def _try(**kw):
        tried.append(str(kw))
        return pd.read_csv(path, **kw)

    if sep is not None:
        try:
            df = _try(sep=sep, on_bad_lines=on_bad_lines)
            print(f"[INFO] Read OK with sep='{sep}'")
            return df
        except Exception as e:
            print(f"[WARN] Failed with sep='{sep}': {e}", file=sys.stderr)

    try:
        df = _try(on_bad_lines=on_bad_lines)
        print(f"[INFO] Read OK with default comma")
        return df
    except Exception as e:
        print(f"[WARN] Default comma failed: {e}", file=sys.stderr)

    try:
        df = _try(sep=None, engine="python", on_bad_lines=on_bad_lines)
        print(f"[INFO] Read OK with auto-detect")
        return df
    except Exception as e:
        print(f"[WARN] Auto-detect failed: {e}", file=sys.stderr)

    for alt in ["\t", ";", "|"]:
        try:
            df = _try(sep=alt, engine="python", on_bad_lines=on_bad_lines)
            print(f"[INFO] Read OK with sep='{alt}'")
            return df
        except Exception:
            pass

    if on_bad_lines != "skip":
        try:
            df = _try(sep=None, engine="python", on_bad_lines="skip")
            print("[INFO] Read OK with on_bad_lines='skip'")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to parse table. Tried: {tried}")

    raise RuntimeError("Failed to read file")


def load_and_prepare(
    csv_path: str,
    target_col: str,
    use_binary: bool,
    positive_label: str,
    extra_features: Optional[List[str]] = None,
    sep: Optional[str] = None,
    on_bad_lines: str = "error",
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    
    df = read_table(csv_path, sep=sep, on_bad_lines=on_bad_lines)
    print(f"[INFO] Loaded data shape: {df.shape}")
    print(f"[INFO] Columns found: {list(df.columns)}")

    # Build feature list
    features = FEATURES_KOI.copy()
    if extra_features:
        features += extra_features

    # Drop flag columns if they exist
    drop_cols = [c for c in FLAG_COLS if c in df.columns]
    if drop_cols:
        print(f"[INFO] Dropping flag columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # Verificar que target existe
    if target_col not in df.columns:
        raise KeyError(f"Target '{target_col}' not found. Available: {list(df.columns)}")
    
    # Keep only selected features + target
    keep_cols = pick_existing_features(df, features) + [target_col]
    df = df[keep_cols].copy()
    
    print(f"[INFO] Using {len(keep_cols)-1} features")

    # Verificar valores faltantes
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        print("[WARN] Found missing values:")
        print(missing_counts[missing_counts > 0])
        print("[WARN] Dropping rows with missing values...")
        df = df.dropna()
        print(f"[INFO] New shape after dropping NaNs: {df.shape}")

    # Separate X and y
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str)

    # Binary option
    if use_binary:
        y = maybe_binary_labels(y, positive_label)
        classes_info = f"Binary: positive='{positive_label}' vs others"
    else:
        classes_info = f"Multiclass: {sorted(pd.unique(y))}"
    print(f"[INFO] {classes_info}")
    
    # Check class distribution
    print("[INFO] Class distribution:")
    print(y.value_counts())

    # Encode labels if multiclass
    if not use_binary:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        class_names = list(le.classes_)
    else:
        y_enc = y.values
        class_names = ["negative", "positive"]

    print(f"[INFO] Final data ready. X shape: {X.shape}")
    
    return X, pd.Series(y_enc), class_names, list(X.columns)


def build_model(random_state: int = 42):
    if _HAS_LGBM:
        print("[INFO] Using LightGBM")
        model = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            class_weight="balanced",
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        )
    else:
        print("[INFO] Using RandomForest")
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    return model


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, folds: int = 5) -> float:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    print(f"[CV] F1-macro: mean={scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()


def train_and_evaluate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    class_names: List[str],
    test_size: float = 0.2,
    use_smote: bool = False,
    outdir: str = "outputs",
    random_state: int = 42,
):
    ensure_dirs([outdir, os.path.join(outdir, "figs"), os.path.join(outdir, "reports")])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    if use_smote:
        if not _HAS_SMOTE:
            print("[WARN] SMOTE not available, skipping", file=sys.stderr)
        else:
            print("[INFO] Applying SMOTE to training set")
            sm = SMOTE(random_state=random_state)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print(f"[INFO] After SMOTE: {len(X_train)} samples")

    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    has_proba = hasattr(model, "predict_proba")
    y_proba = model.predict_proba(X_test) if has_proba else None

    # Metrics
    rep = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print("\n" + "="*50)
    print("[TEST] Classification Report:")
    print("="*50)
    print(rep)
    print(f"[TEST] Balanced Accuracy: {bal_acc:.4f}")

    # ROC-AUC
    roc_msg = ""
    try:
        if y_proba is not None:
            if len(class_names) > 2:
                roc = roc_auc_score(y_test, y_proba, multi_class="ovr")
            else:
                roc = roc_auc_score(y_test, y_proba[:, 1])
            roc_msg = f"[TEST] ROC-AUC: {roc:.4f}"
            print(roc_msg)
    except Exception as e:
        roc_msg = f"[TEST] ROC-AUC not computed: {e}"
        print(roc_msg)

    # Save report
    report_path = os.path.join(outdir, "reports", "metrics.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write("Classification Report\n")
        f.write("="*50 + "\n")
        f.write(rep + "\n")
        f.write(f"Balanced Accuracy: {bal_acc:.4f}\n")
        if roc_msg:
            f.write(roc_msg + "\n")
    print(f"[INFO] Metrics saved to: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    fig_cm = plt.figure(figsize=(6, 5))
    ax = fig_cm.add_subplot(111)
    im = ax.imshow(cm, aspect="auto", cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
    fig_cm.tight_layout()
    cm_path = os.path.join(outdir, "figs", "confusion_matrix.png")
    fig_cm.savefig(cm_path, dpi=150)
    plt.close(fig_cm)
    print(f"[INFO] Confusion matrix saved to: {cm_path}")

    # Feature importances
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            raise AttributeError("No feature_importances_")
        indices = np.argsort(importances)[::-1]
        fig_imp = plt.figure(figsize=(10, 6))
        ax2 = fig_imp.add_subplot(111)
        ax2.bar(range(len(indices)), importances[indices])
        ax2.set_xticks(range(len(indices)))
        ax2.set_xticklabels([X.columns[i] for i in indices], rotation=45, ha="right")
        ax2.set_title("Feature Importances")
        ax2.set_ylabel("Importance")
        fig_imp.tight_layout()
        imp_path = os.path.join(outdir, "figs", "feature_importances.png")
        fig_imp.savefig(imp_path, dpi=150)
        plt.close(fig_imp)
        print(f"[INFO] Feature importances saved to: {imp_path}")

        print("\n[INFO] Top 10 Features:")
        top_k = min(10, len(indices))
        for rank in range(top_k):
            idx = indices[rank]
            print(f"  {rank+1:2d}. {X.columns[idx]:20s} ({importances[idx]:.4f})")
    except Exception as e:
        print(f"[WARN] Could not plot importances: {e}", file=sys.stderr)
    
    return model


def parse_args():
    p = argparse.ArgumentParser(description="KOI ML Pipeline - Datos Filtrados")
    p.add_argument("--csv", required=True, help="Path to CSV")
    p.add_argument("--target", default=TARGET_DEFAULT, help=f"Target column (default: {TARGET_DEFAULT})")
    p.add_argument("--binary", action="store_true", help="Binary classification")
    p.add_argument("--positive", default="CONFIRMED", help="Positive label")
    p.add_argument("--smote", action="store_true", help="Apply SMOTE")
    p.add_argument("--test_size", type=float, default=0.2, help="Test fraction")
    p.add_argument("--outdir", default="outputs", help="Output directory")
    p.add_argument("--sep", default=None, help="CSV separator")
    p.add_argument("--on_bad_lines", default="error", choices=["error", "warn", "skip"])
    return p.parse_args()


def main():
    args = parse_args()
    
    print("="*50)
    print("KOI ML Pipeline - Starting")
    print("="*50)
    
    X, y, class_names, feature_names = load_and_prepare(
        csv_path=args.csv,
        target_col=args.target,
        use_binary=args.binary,
        positive_label=args.positive,
        extra_features=None,
        sep=args.sep,
        on_bad_lines=args.on_bad_lines,
    )

    model = build_model(random_state=42)

    print("\n[INFO] Starting Cross-Validation...")
    cross_validate_model(model, X, y, folds=5)

    print("\n[INFO] Training final model...")
    fitted_model = train_and_evaluate(
        model=model,
        X=X,
        y=y,
        class_names=class_names,
        test_size=args.test_size,
        use_smote=args.smote,
        outdir=args.outdir,
        random_state=42,
    )

    # Save model
    bundle = {
        "model": fitted_model,
        "features": feature_names,
        "class_names": class_names,
        "target": args.target,
        "binary": args.binary,
        "positive_label": args.positive,
    }
    model_path = os.path.join(args.outdir, "koi_model.joblib")
    joblib.dump(bundle, model_path)
    print(f"\n[INFO] Model saved to: {model_path}")
    print("="*50)
    print("Pipeline Complete")
    print("="*50)


if __name__ == "__main__":
    main()