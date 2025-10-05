# type: ignore

import argparse
import json
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

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
    "disp_label",
    "y_confirmed",
    "rowid",
    "id",
    "idx",
    "index",
}


def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
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

    # Heurística: columnas tipo ID de alta cardinalidad
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
    return num_cols, cat_cols, dropped_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
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


def compute_best_f1_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, probs)
    if len(prec) <= 1 or len(rec) <= 1 or len(thr) == 0:
        return 0.5
    f1s = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    best_idx = int(np.nanargmax(f1s))
    return float(thr[best_idx])


# ----------------------------
# Entrenamiento principal
# ----------------------------


def train_from_harmonized(
    csv_path: Path,
    outdir: Path,
    test_size: float,
    seed: int,
    exclude_cols: List[str],
    auto_drop_ids: bool,
):
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    df = ensure_labels(df)
    df.to_csv(outdir / "exoplanets_harmonized_used.csv", index=False)

    labeled = df[df["disp_label"] != "UNKNOWN"].copy()
    if labeled.empty:
        raise ValueError(
            "No hay filas etiquetadas (CONFIRMED/CANDIDATE/FALSE_POSITIVE)."
        )

    num_cols, cat_cols, dropped_cols = split_feature_types(
        labeled, exclude_cols=exclude_cols, auto_drop_ids=auto_drop_ids
    )

    if not num_cols and not cat_cols:
        raise ValueError("No hay columnas de features disponibles tras los filtros.")

    pre = build_preprocessor(num_cols, cat_cols)
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    pipe = Pipeline(steps=[("prep", pre), ("rf", rf)])

    X = labeled[num_cols + cat_cols]
    y = labeled["y_confirmed"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipe.fit(X_train, y_train)

    probs_test = pipe.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, probs_test)
    pr_auc = average_precision_score(y_test, probs_test)
    best_thr = compute_best_f1_threshold(y_test, probs_test)

    # ===== MÉTRICAS EXTENDIDAS =====
    pred_test = (probs_test >= best_thr).astype(int)
    report_dict = classification_report(
        y_test, pred_test, target_names=["NOT_CONFIRMED", "CONFIRMED"], output_dict=True
    )
    report_text = classification_report(
        y_test, pred_test, target_names=["NOT_CONFIRMED", "CONFIRMED"]
    )
    cm = confusion_matrix(y_test, pred_test)
    acc = accuracy_score(y_test, pred_test)
    macro_f1 = report_dict["macro avg"]["f1-score"]

    metrics_ext = {
        "ROC_AUC": float(roc),
        "PR_AUC": float(pr_auc),
        "Best_Threshold_F1": float(best_thr),
        "Accuracy": float(acc),
        "Macro_F1": float(macro_f1),
        "Confusion_Matrix": cm.tolist(),
        "Classification_Report": report_dict,
        "N_features_numeric": len(num_cols),
        "N_features_categorical": len(cat_cols),
        "Rows_Total": int(len(df)),
        "Rows_Labeled": int(len(labeled)),
    }
    (outdir / "metrics_detailed.json").write_text(
        json.dumps(metrics_ext, indent=2), encoding="utf-8"
    )
    (outdir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    # ===== GUARDADO MODELO =====
    dump(pipe, outdir / "model.joblib")
    cfg = {
        "source_csv": str(csv_path),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "dropped_columns": dropped_cols,
        "threshold": best_thr,
        "metrics": {"ROC_AUC": float(roc), "PR_AUC": float(pr_auc)},
        "random_state": seed,
        "test_size": test_size,
        "auto_drop_ids": auto_drop_ids,
        "exclude_cols": exclude_cols,
    }
    (outdir / "model_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )

    # ===== SCORING COMPLETO =====
    probs_all = pipe.predict_proba(X)[:, 1]
    pred_all = (probs_all >= best_thr).astype(int)
    scored = labeled.copy()
    scored["score_confirmed"] = probs_all
    scored["pred_confirmed"] = pred_all
    scored.to_csv(outdir / "exoplanets_scored.csv", index=False)

    rank_candidates = scored[scored["disp_label"] != "CONFIRMED"].sort_values(
        "score_confirmed", ascending=False
    )
    rank_candidates.to_csv(outdir / "exoplanets_candidates_ranked.csv", index=False)

    importances = pipe.named_steps["rf"].feature_importances_
    out_feature_names = num_cols + cat_cols
    fi = pd.DataFrame(
        {"feature": out_feature_names[: len(importances)], "importance": importances}
    ).sort_values("importance", ascending=False)
    fi.to_csv(outdir / "feature_importances.csv", index=False)

    # ===== MÉTRICAS RESUMEN =====
    (outdir / "metrics.json").write_text(
        json.dumps(
            {
                "ROC_AUC": float(roc),
                "PR_AUC": float(pr_auc),
                "Best_Threshold_F1": float(best_thr),
                "Accuracy": float(acc),
                "Macro_F1": float(macro_f1),
                "Positives_in_Test": int(y_test.sum()),
                "Negatives_in_Test": int((1 - y_test).sum()),
                "N_features_numeric": len(num_cols),
                "N_features_categorical": len(cat_cols),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ===== PRINT FINAL =====
    print(
        json.dumps(
            {
                "ROC_AUC": round(float(roc), 4),
                "PR_AUC": round(float(pr_auc), 4),
                "Best_Threshold_F1": round(float(best_thr), 4),
                "Accuracy": round(float(acc), 4),
                "Macro_F1": round(float(macro_f1), 4),
                "N_num": len(num_cols),
                "N_cat": len(cat_cols),
            },
            indent=2,
        )
    )


# ----------------------------
# CLI
# ----------------------------


def parse_args():
    ap = argparse.ArgumentParser(
        description="Entrenar RandomForest desde CSV armonizado usando TODAS las columnas útiles."
    )
    ap.add_argument("--csv", type=str, required=True, help="Ruta al CSV ya armonizado.")
    ap.add_argument(
        "--out", type=str, default="./outputs_strict", help="Carpeta de salida."
    )
    ap.add_argument("--test-size", type=float, default=0.2, help="Proporción de test.")
    ap.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    ap.add_argument(
        "--exclude-cols",
        type=str,
        default="",
        help="Lista de columnas a excluir, separadas por coma. Ej: 'koi_name,filename'",
    )
    ap.add_argument(
        "--no-auto-drop-ids",
        action="store_true",
        help="Desactiva la heurística que elimina categóricas con altísima cardinalidad.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    exclude_cols = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
    train_from_harmonized(
        csv_path=Path(args.csv),
        outdir=Path(args.out),
        test_size=args.test_size,
        seed=args.seed,
        exclude_cols=exclude_cols,
        auto_drop_ids=not args.no_auto_drop_ids,
    )


if __name__ == "__main__":
    main()
