#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kepler KOI ML Pipeline (Hackathon-ready)
----------------------------------------
- Loads KOI table (CSV)
- Selects physically meaningful features
- Handles missing values
- Encodes labels (multi-class or binary)
- Cross-validation (StratifiedKFold) for F1-macro
- Trains classifier (LightGBM if available, else RandomForest)
- Evaluates on holdout test set
- Saves metrics and plots (feature importances, confusion matrix)

Usage:
    python kepler_ml_pipeline.py --csv path/to/kepler.csv [--binary] [--positive CONFIRMED] [--smote]
"""
import argparse
import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
import joblib

# Optional SMOTE (only if available)
try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except Exception:
    _HAS_SMOTE = False

# Try LightGBM; if not available we fall back to RandomForest
_HAS_LGBM = True
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except Exception:
    _HAS_LGBM = False

import matplotlib.pyplot as plt


FEATURES_BASE = [
    # Transit/orbital parameters
    "koi_period",       # Orbital Period [days]
    "koi_impact",       # Impact Parameter
    "koi_duration",     # Transit Duration [hrs]
    "koi_depth",        # Transit Depth [ppm]
    "koi_model_snr",    # Transit Signal-to-Noise

    # Planet properties (derived)
    "koi_prad",         # Planetary Radius [Earth radii]
    "koi_teq",          # Equilibrium Temperature [K]
    "koi_insol",        # Insolation Flux [Earth flux]

    # Stellar properties
    "koi_steff",        # Stellar Effective Temperature [K]
    "koi_slogg",        # Stellar Surface Gravity [log10(cm/s**2)]
    "koi_srad",         # Stellar Radius [Solar radii]
    "koi_kepmag",       # Kepler-band [mag]
]

ID_COLS = ["kepid", "kepoi_name", "kepler_name", "koi_tce_delivname", "ra", "dec"]
FLAG_COLS = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]

TARGET_DEFAULT = "koi_disposition"  # 'CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'
TARGET_DEFAULT = "koi_pdisposition"  # 'FALSE POSITIVE', 'CANDIDATE'

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


def load_and_prepare(
    csv_path: str,
    target_col: str,
    use_binary: bool,
    positive_label: str,
    extra_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded data shape: {df.shape}")

    # Build feature list
    features = FEATURES_BASE.copy()
    if extra_features:
        features += extra_features

    # Drop obviously problematic or leakage-prone columns
    drop_cols = [c for c in (ID_COLS + FLAG_COLS) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Keep only selected features + target
    keep_cols = pick_existing_features(df, features) + [target_col]
    df = df[keep_cols].copy()

    # Separate X and y
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str)

    # Binary option
    if use_binary:
        # Map positive class vs all others
        y = maybe_binary_labels(y, positive_label)
        classes_info = f"Binary task: positive='{positive_label}', negative='others'"
    else:
        classes_info = f"Multiclass task with classes: {sorted(pd.unique(y))}"
    print(f"[INFO] {classes_info}")

    # Impute missing values (median - robust for skewed distributions)
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Encode labels if multiclass
    if not use_binary:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        class_names = list(le.classes_)
    else:
        y_enc = y.values
        class_names = ["negative", "positive"]

    return X_imp, pd.Series(y_enc), class_names, imputer, list(X_imp.columns)
    #return X_imp, pd.Series(y_enc), class_names


def build_model(random_state: int = 42):
    if _HAS_LGBM:
        print("[INFO] Using LightGBM classifier")
        model = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            class_weight="balanced",
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        )
    else:
        print("[INFO] LightGBM not available, using RandomForest classifier")
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
) -> None:
    ensure_dirs([outdir, os.path.join(outdir, "figs"), os.path.join(outdir, "reports")])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if use_smote:
        if not _HAS_SMOTE:
            print("[WARN] SMOTE requested but imblearn not installed; skipping.", file=sys.stderr)
        else:
            print("[INFO] Applying SMOTE to training split")
            sm = SMOTE(random_state=random_state)
            X_train, y_train = sm.fit_resample(X_train, y_train)

    # Fit model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Probabilities (if available)
    has_proba = hasattr(model, "predict_proba")
    y_proba = model.predict_proba(X_test) if has_proba else None

    # Metrics
    rep = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print("\n[TEST] Classification report:\n", rep)
    print(f"[TEST] Balanced accuracy: {bal_acc:.4f}")

    # ROC-AUC (multi-class OVR) if we have probabilities and >2 classes
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
        f.write("Classification report\n")
        f.write(rep + "\n")
        f.write(f"Balanced accuracy: {bal_acc:.4f}\n")
        if roc_msg:
            f.write(roc_msg + "\n")
    print(f"[INFO] Saved metrics to: {report_path}")

    # Confusion matrix plot (matplotlib only; single plot; no explicit colors)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    fig_cm = plt.figure(figsize=(6, 5))
    ax = fig_cm.add_subplot(111)
    im = ax.imshow(cm, aspect="auto")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig_cm.tight_layout()
    cm_path = os.path.join(outdir, "figs", "confusion_matrix.png")
    fig_cm.savefig(cm_path, dpi=150)
    plt.close(fig_cm)
    print(f"[INFO] Saved confusion matrix to: {cm_path}")

    # Feature importances (matplotlib only; single plot; no explicit colors)
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            raise AttributeError("Model has no feature_importances_")
        indices = np.argsort(importances)[::-1]
        fig_imp = plt.figure(figsize=(8, 6))
        ax2 = fig_imp.add_subplot(111)
        ax2.bar(range(len(indices)), importances[indices])
        ax2.set_xticks(range(len(indices)))
        ax2.set_xticklabels([X.columns[i] for i in indices], rotation=90)
        ax2.set_title("Feature Importances")
        fig_imp.tight_layout()
        imp_path = os.path.join(outdir, "figs", "feature_importances.png")
        fig_imp.savefig(imp_path, dpi=150)
        plt.close(fig_imp)
        print(f"[INFO] Saved feature importances to: {imp_path}")

        # Print top 10
        top_k = min(10, len(indices))
        print("[INFO] Top features:")
        for rank in range(top_k):
            idx = indices[rank]
            print(f"  {rank+1:2d}. {X.columns[idx]}  ({importances[idx]:.4f})")
    except Exception as e:
        print(f"[WARN] Could not plot importances: {e}", file=sys.stderr)

    return model


def parse_args():
    p = argparse.ArgumentParser(description="Kepler KOI ML Pipeline")
    p.add_argument("--csv", required=True, help="Path to KOI CSV file")
    p.add_argument("--target", default=TARGET_DEFAULT, help=f"Target column (default: {TARGET_DEFAULT})")
    p.add_argument("--binary", action="store_true", help="Use binary classification (positive vs others)")
    p.add_argument("--positive", default="CONFIRMED", help="Positive label for binary (default: CONFIRMED)")
    p.add_argument("--smote", action="store_true", help="Apply SMOTE to training data (requires imblearn)")
    p.add_argument("--test_size", type=float, default=0.2, help="Test size fraction (default: 0.2)")
    p.add_argument("--outdir", default="outputs", help="Output directory (default: outputs)")
    return p.parse_args()


def main():
    args = parse_args()
    X, y, class_names, imputer, feature_names = load_and_prepare(
        csv_path=args.csv,
        target_col=args.target,
        use_binary=args.binary,
        positive_label=args.positive,
        extra_features=None,
        sep=args.sep,
        on_bad_lines=args.on_bad_lines,
    )

    model = build_model(random_state=42)

    print("[INFO] Cross-validating model (F1-macro, 5-fold)...")
    cross_validate_model(model, X, y, folds=5)

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

    # --- NUEVO: guardar el mejor modelo + preprocesador + metadata ---
    bundle = {
        "model": fitted_model,
        "imputer": imputer,
        "features": feature_names,
        "class_names": class_names,
        "target": args.target,
        "binary": args.binary,
        "positive_label": args.positive,
    }
    model_path = os.path.join(args.outdir, "kepler_koi_model.joblib")
    joblib.dump(bundle, model_path)
    print(f"[INFO] Saved model bundle to: {model_path}")



if __name__ == "__main__":
    main()
