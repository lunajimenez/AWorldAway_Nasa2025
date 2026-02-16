# type: ignore

import argparse
import warnings
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

# Modelos
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)

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

EXCLUDE_ALWAYS = {
    "final_disposition",
    "final_disposition_raw",
    "koi_disposition",
    "disp_label",
    "y_confirmed",
    "rowid",
    "id",
    "idx",
    "index",
    "koi_score", 
}


def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "final_disposition" in out.columns:
        disp = out["final_disposition"].astype(str).str.upper()
    elif "final_disposition_raw" in out.columns:
        disp = out["final_disposition_raw"].astype(str).str.upper()
        disp = disp.replace(LABEL_ALIASES)
    elif "koi_disposition" in out.columns:
        disp = out["koi_disposition"].astype(str).str.upper()
        disp = disp.replace(LABEL_ALIASES)
    else:
        disp = pd.Series(["UNKNOWN"] * len(out), index=out.index)
    out["disp_label"] = disp
    out["y_confirmed"] = (disp == "CONFIRMED").astype(int)
    return out


def split_feature_types(
    df: pd.DataFrame,
    exclude_cols: List[str],
    auto_drop_ids: bool = True,
    id_uniqueness_ratio: float = 0.9,
) -> Tuple[List[str], List[str], List[str]]:
    cols = [c for c in df.columns if c not in exclude_cols and c not in EXCLUDE_ALWAYS]
    num_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cols if c not in num_cols]
    dropped_cols = []

    if auto_drop_ids and len(df) > 0:
        to_drop = []
        for c in cat_cols:
            n_unique = df[c].nunique(dropna=True)
            if n_unique / max(1, len(df)) >= id_uniqueness_ratio:
                to_drop.append(c)
        if to_drop:
            dropped_cols.extend(to_drop)
            cat_cols = [c for c in cat_cols if c not in to_drop]

    def drop_too_missing(candidates: List[str], threshold=0.98):
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
    return num_cols, cat_cols, dropped_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    # Agregamos StandardScaler para modelos sensibles a la escala (LR, SVM, KNN, MLP)
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def evaluate_models(
    csv_path: Path,
    test_size: float,
    seed: int,
    exclude_cols: List[str],
    auto_drop_ids: bool,
):
    print(f"Loading data from: {csv_path}")
    print("⚙️  Comparación de modelos utilizando dataset 'KOI_All_Filtrated.csv'")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Rename columns to match user request
    rename_map = {
        "koi_period": "orbital_period_days",
        "koi_duration": "transit_duration_hours", 
        "koi_depth": "transit_depth_ppm",
        "koi_prad": "planet_radius_earth",
        "koi_teq": "equilibrium_temperature_K",
        "koi_insol": "insolation_flux_Earth",
        "koi_srad": "stellar_radius_solar",
        "koi_steff": "stellar_temperature_K",
        "koi_model_snr": "signal_to_noise",
        "koi_disposition": "final_disposition"
    }
    df = df.rename(columns=rename_map)
    
    # Keep only requested columns that actually exist in the processed dataset
    keep_cols = [
        "orbital_period_days",
        "transit_duration_hours", 
        "transit_depth_ppm",
        "planet_radius_earth",
        "equilibrium_temperature_K",
        "insolation_flux_Earth",
        "stellar_radius_solar",
        "stellar_temperature_K",
        "signal_to_noise",
        # Flags (these exist in KOI_All_Filtrated.csv)
        "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
        # Target
        "final_disposition"
    ]
    
    # Filter columns that actually exist in the dataframe
    existing_cols = [c for c in keep_cols if c in df.columns]
    
    if 'koi_score' in df.columns:
        print("ℹ️  koi_score detectado pero será ignorado para consistencia con koi_ml.py")
        
    df = df[existing_cols]

    df = ensure_labels(df)

    labeled = df[df["disp_label"] != "UNKNOWN"].copy()
    if labeled.empty:
        raise ValueError("No labeled rows found.")
    
    num_cols, cat_cols, _ = split_feature_types(
        labeled, exclude_cols=exclude_cols, auto_drop_ids=auto_drop_ids
    )

    if not num_cols and not cat_cols:
        raise ValueError("No feature columns available.")

    print(f"Features: {len(num_cols)} numeric, {len(cat_cols)} categorical")
    print(f"Total labeled samples: {len(labeled)}")

    X = labeled[num_cols + cat_cols]
    y = labeled["y_confirmed"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Definir modelos a comparar
    models = [
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1
            ),
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(random_state=seed),
        ),
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        ),
        (
            "DecisionTree",
            DecisionTreeClassifier(class_weight="balanced", random_state=seed),
        ),
        (
            "KNeighbors",
            KNeighborsClassifier(n_jobs=-1),
        ),
        (
            "AdaBoost",
            AdaBoostClassifier(algorithm="SAMME", random_state=seed),
        ),
        (
            "MLP (Neural Net)",
            MLPClassifier(max_iter=500, random_state=seed)
        ),
        (
            "SVM (RBF Kernel)",
            SVC(probability=True, random_state=seed, class_weight="balanced")
        )
    ]

    results = []

    print("\n" + "=" * 100)
    print(f"{'Model':<20} | {'ACC':<7} | {'AUC':<7} | {'F1':<7} | {'Prec':<7} | {'Recall':<7}")
    print("=" * 100)

    for name, model in models:
        pipe = Pipeline(steps=[("prep", preprocessor), ("clf", model)])
        
        try:
            pipe.fit(X_train, y_train)
            
            # Predicciones
            preds = pipe.predict(X_test)
            
            # Probabilidades (si soporta predict_proba)
            if hasattr(model, "predict_proba"):
                probs = pipe.predict_proba(X_test)[:, 1]
                roc = roc_auc_score(y_test, probs)
            else:
                roc = 0.0 

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="binary")
            prec = precision_score(y_test, preds, average="binary", zero_division=0)
            rec = recall_score(y_test, preds, average="binary")

            results.append({
                "Model": name,
                "Accuracy": acc,
                "ROC_AUC": roc,
                "F1_Score": f1,
                "Precision": prec,
                "Recall": rec
            })

            print(f"{name:<20} | {acc:.4f}  | {roc:.4f}  | {f1:.4f}  | {prec:.4f}  | {rec:.4f}")

        except Exception as e:
            print(f"{name:<20} | FAILED: {str(e)}")

    print("=" * 100)
    
    if results:
        results_df = pd.DataFrame(results)
        best_model = results_df.loc[results_df['F1_Score'].idxmax()]
        print(f"\n🏆 Best Model by F1 Score: {best_model['Model']} (F1: {best_model['F1_Score']:.4f})")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Comparar múltiples modelos de ML en el dataset."
    )
    # Default to KOI_All_Filtrated.csv
    default_csv = Path(__file__).parent.parent / "data/processed/KOI_All_Filtrated.csv"
    
    ap.add_argument("--csv", type=str, default=str(default_csv), help="Ruta al CSV.")
    ap.add_argument("--test-size", type=float, default=0.2, help="Proporción de test.")
    ap.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    ap.add_argument("--exclude-cols", type=str, default="", help="Columnas a excluir.")
    ap.add_argument("--no-auto-drop-ids", action="store_true", help="No auto drop IDs.")
    
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exclude_cols = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
    
    evaluate_models(
        csv_path=Path(args.csv),
        test_size=args.test_size,
        seed=args.seed,
        exclude_cols=exclude_cols,
        auto_drop_ids=not args.no_auto_drop_ids,
    )
