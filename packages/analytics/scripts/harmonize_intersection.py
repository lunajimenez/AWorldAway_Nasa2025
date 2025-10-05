# armonizado.py
# -*- coding: utf-8 -*-
"""
Pipeline de armonización + diagnósticos + limpieza con unidades y derivadas físicas.
Produce:
  - <outdir>/Harmonized_FULL.csv        -> todo lo armonizado (máxima cobertura)
  - <outdir>/Harmonized_STRICT.csv      -> filtro conservador listo para ML
  - <outdir>/Harmonized_DIAG.json       -> métricas de calidad, conversiones y conteos
  - <outdir>/Harmonized_MISSING.csv     -> % faltantes por columna
  - <outdir>/Harmonized_OUTLIERS.csv    -> ratios IQR por columna numérica

Uso típico:
python armonizado.py ^
  --koi DATA/KOI_All_Filtrated.csv ^
  --k2  DATA/K2_All_Filtrated.csv ^
  --toi DATA/TOI_All_Filtrated.csv ^
  --outdir FilteredData ^
  --winsorize --add_logs
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Opcional: opt-in al comportamiento futuro (evita downcasting silencioso)
pd.set_option('future.no_silent_downcasting', True)

# ----------------------------
# Constantes
# ----------------------------
R_SUN_TO_RE = 109.2          # radios terrestres por radio solar
AU_PER_R_SUN = 0.00465047    # AU por radio solar

# ----------------------------
# Utilidades genéricas
# ----------------------------
def read_csv_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")

def normalize_numeric(series: pd.Series) -> pd.Series:
    """
    Limpia y convierte a float:
    - Si ya es numérica, solo coerce a float.
    - Si es texto, quita espacios, miles (','), '%' y convierte.
    Parche robusto para evitar errores del .str accessor.
    """
    if series is None:
        return pd.Series(dtype="float64")

    s = pd.Series(series, copy=True)

    # Caso 1: ya numérica
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    # Caso 2: tratar como texto
    s = s.astype("string")
    s = s.str.strip()

    # Normaliza vacíos/sentinelas → NA
    s = s.replace(
        {"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NA": pd.NA, "None": pd.NA},
        regex=False
    ).astype("string")

    # Elimina separador de miles y '%' (si corresponde)
    s = s.str.replace(",", "", regex=False).astype("string")
    s = s.str.replace("%", "", regex=False).astype("string")

    # Convierte a float
    return pd.to_numeric(s, errors="coerce")

def winsorize(s: pd.Series, p_low=0.5, p_high=99.5) -> pd.Series:
    if s.notna().sum() == 0:
        return s
    lo, hi = np.nanpercentile(s, [p_low, p_high])
    return s.clip(lo, hi)

def iqr_outlier_ratio(s: pd.Series):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series({"lower": 0, "upper": 0, "total": int(s.notna().sum()), "ratio": 0.0})
    lower = int((s < (q1 - 1.5*iqr)).sum())
    upper = int((s > (q3 + 1.5*iqr)).sum())
    total = int(s.notna().sum())
    ratio = (lower + upper) / total if total else 0.0
    return pd.Series({"lower": lower, "upper": upper, "total": total, "ratio": ratio})

# ----------------------------
# Mapeos flexibles (regex) por misión
# ----------------------------
RX = lambda *p: [re.compile(x, re.I) for x in p]

MAPS = {
    "KOI": {
        "period": RX(r"^koi_period$", r"pl_orbper", r"orb.*per", r"\bperiod\b"),
        "dur":    RX(r"^koi_duration$", r"pl_trandurh?$", r"tran.*dur", r"\bduration"),
        "depth":  RX(r"^koi_depth$", r"pl_trandep", r"depth.*ppm", r"\bdepth\b"),
        "prad":   RX(r"^koi_prad$", r"pl_rade", r"planet.*rad", r"\bradius\b"),
        "teq":    RX(r"^koi_teq$", r"pl_eqt", r"eq.*temp"),
        "insol":  RX(r"^koi_insol$", r"pl_insol", r"insol"),
        "rstar":  RX(r"^koi_srad$", r"st_rad", r"rstar"),
        "teff":   RX(r"^koi_steff$", r"st_teff", r"teff"),
        "disp":   RX(r"^koi_disposition$", r"disposition", r"tfopwg_disp", r"pdisposition", r"final.*dispos.*")
    },
    "K2": {
        "period": RX(r"^pl_orbper$", r"orb.*per", r"\bperiod\b"),
        "dur":    RX(r"^pl_trandurh?$", r"tran.*dur", r"\bduration"),
        "depth":  RX(r"^pl_trandep$", r"depth.*ppm", r"\bdepth\b"),
        "prad":   RX(r"^pl_rade$", r"planet.*rad", r"\bradius\b"),
        "teq":    RX(r"^pl_eqt$", r"eq.*temp"),
        "insol":  RX(r"^pl_insol$", r"insol"),
        "rstar":  RX(r"^st_rad$", r"rstar"),
        "teff":   RX(r"^st_teff$", r"teff"),
        "disp":   RX(r"^disposition$", r"final.*dispos.*"),
    },
    "TOI": {
        "period": RX(r"^pl_orbper$", r"orb.*per", r"\bperiod\b"),
        "dur":    RX(r"^pl_trandurh$", r"pl_trandur$", r"tran.*dur", r"\bduration"),
        "depth":  RX(r"^pl_trandep$", r"depth.*ppm", r"\bdepth\b"),
        "prad":   RX(r"^pl_rade$", r"planet.*rad", r"\bradius\b"),
        "teq":    RX(r"^pl_eqt$", r"eq.*temp"),
        "insol":  RX(r"^pl_insol$", r"insol"),
        "rstar":  RX(r"^st_rad$", r"rstar"),
        "teff":   RX(r"^st_teff$", r"teff"),
        "disp":   RX(r"^tfopwg_disp$", r"disposition", r"final.*dispos.*"),
    }
}

def pick_col(df: pd.DataFrame, patterns) -> str | None:
    cols = list(df.columns)
    for p in patterns:
        for c in cols:
            if p.search(c):
                return c
    return None

# ----------------------------
# Disposition normalizer
# ----------------------------
def unify_disposition(val, source=None):
    if pd.isna(val): return np.nan
    s = str(val).strip().upper().replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    # TESS/TOI
    if source == "TOI":
        if s in ("CP","KP","CONFIRMED"): return "CONFIRMED"
        if s in ("PC","APC","CANDIDATE"): return "CANDIDATE"
        if s in ("FP","FA","FALSE POSITIVE","FALSE_POSITIVE"): return "FALSE_POSITIVE"
    # Genéricos
    if "CONFIR" in s or s in ("CP","KP"): return "CONFIRMED"
    if "CAND" in s or s in ("PC","APC"): return "CANDIDATE"
    if "FALSE" in s or "REFUT" in s or s in ("FP","FA"): return "FALSE_POSITIVE"
    if s in ("CONFIRMED","CANDIDATE","FALSE_POSITIVE"): return s
    return np.nan

# ----------------------------
# Heurísticas de UNIDADES
# ----------------------------
def convert_depth_to_ppm(depth_series: pd.Series) -> tuple[pd.Series, dict]:
    """Convierte profundidad a ppm usando heurística basada en cuantiles."""
    meta = {"rule": "assumed_ppm", "scale": 1.0}
    s = depth_series.copy()
    if s.notna().sum() == 0:
        return s, meta
    q90 = np.nanpercentile(s, 90)
    # Regla simple robusta:
    # - q90 < 0.05 -> fracción → ppm = frac * 1e6
    # - 0.05 ≤ q90 ≤ 10 -> porcentaje → ppm = % * 1e4
    # - si q90 >> 10 -> ya está en ppm
    if q90 < 0.05:
        s = s * 1e6
        meta = {"rule": "fraction_to_ppm", "scale": 1e6}
    elif 0.05 <= q90 <= 10:
        s = s * 1e4
        meta = {"rule": "percent_to_ppm", "scale": 1e4}
    else:
        meta = {"rule": "already_ppm", "scale": 1.0}
    return s, meta

def convert_duration_to_hours(dur_series: pd.Series, col_name_hint: str | None) -> tuple[pd.Series, dict]:
    """Convierte duración a horas si parece estar en días."""
    meta = {"rule": "assumed_hours", "scale": 1.0}
    s = dur_series.copy()
    if s.notna().sum() == 0:
        return s, meta
    # Pista por nombre
    if col_name_hint and re.search("durh", col_name_hint, re.I):
        return s, {"rule": "already_hours_by_name", "scale": 1.0}
    q50 = np.nanmedian(s)
    # Duraciones típicas de tránsito: 1–12 h. Si mediana < 0.6 (días), multiplica por 24
    if q50 is not None and q50 < 0.6:
        s = s * 24.0
        meta = {"rule": "days_to_hours", "scale": 24.0}
    else:
        meta = {"rule": "already_hours_by_value", "scale": 1.0}
    return s, meta

# ----------------------------
# Derivadas físicas
# ----------------------------
def add_physics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    Rstar_Re = out["stellar_radius_solar"] * R_SUN_TO_RE
    with np.errstate(invalid="ignore", divide="ignore"):
        out["rp_over_rstar"] = out["planet_radius_earth"] / Rstar_Re
        out["depth_theory_ppm"] = 1e6 * (out["rp_over_rstar"] ** 2)
        out["depth_ratio"] = out["transit_depth_ppm"] / out["depth_theory_ppm"]
    return out

# ----------------------------
# Preparación por misión
# ----------------------------
def prepare_mission(df_raw: pd.DataFrame, mission: str, diag: dict) -> pd.DataFrame:
    m = MAPS[mission]
    df = df_raw.copy()

    # Local picker
    def pick(name): return pick_col(df, m[name])

    c_period = pick("period")
    c_dur    = pick("dur")
    c_depth  = pick("depth")
    c_prad   = pick("prad")
    c_teq    = pick("teq")
    c_insol  = pick("insol")
    c_rstar  = pick("rstar")
    c_teff   = pick("teff")
    c_disp   = pick("disp")

    diag[mission] = {"found": {
        "period": c_period, "dur": c_dur, "depth": c_depth, "prad": c_prad,
        "teq": c_teq, "insol": c_insol, "rstar": c_rstar, "teff": c_teff, "disp": c_disp
    }}

    out = pd.DataFrame()
    if "source_mission" in df.columns:
        out["source_mission"] = df["source_mission"].values
    else:
        out["source_mission"] = mission if mission != "KOI" else "KEPLER"

    def get_series(col):
        if col and col in df.columns:
            return df[col]
        # misma longitud y dtype string para que normalize_numeric no falle
        return pd.Series([pd.NA] * len(df), dtype="string")

    # Numéricos
    out["orbital_period_days"]  = normalize_numeric(get_series(c_period))

    dur_raw = normalize_numeric(get_series(c_dur))
    dur_conv, dur_meta = convert_duration_to_hours(dur_raw, c_dur or "")
    diag[mission]["duration_meta"] = dur_meta
    out["transit_duration_hours"] = dur_conv

    depth_raw = normalize_numeric(get_series(c_depth))
    depth_conv, depth_meta = convert_depth_to_ppm(depth_raw)
    diag[mission]["depth_meta"] = depth_meta
    out["transit_depth_ppm"]    = depth_conv

    out["planet_radius_earth"]  = normalize_numeric(get_series(c_prad))
    out["equilibrium_temperature_K"] = normalize_numeric(get_series(c_teq))
    out["insolation_flux_Earth"]     = normalize_numeric(get_series(c_insol))
    out["stellar_radius_solar"]      = normalize_numeric(get_series(c_rstar))
    out["stellar_temperature_K"]     = normalize_numeric(get_series(c_teff))

    # Etiquetas
    disp_raw = get_series(c_disp).astype("string")
    out["final_disposition_raw"] = disp_raw
    out["final_disposition"] = disp_raw.apply(lambda v: unify_disposition(v, source=mission))

    return out

# ----------------------------
# Filtros de calidad + versiones
# ----------------------------
ESSENTIAL_STRICT = [
    "orbital_period_days",
    "transit_duration_hours",
    "transit_depth_ppm",
    "planet_radius_earth",
    "stellar_radius_solar",
    "stellar_temperature_K",
    "final_disposition"
]

def strict_filter(df: pd.DataFrame) -> pd.DataFrame:
    keep = df.dropna(subset=[c for c in ESSENTIAL_STRICT if c in df.columns])

    # filtros físicos conservadores (rangos plausibles)
    conds = []
    if "orbital_period_days" in keep:    conds.append((keep["orbital_period_days"] > 0) & (keep["orbital_period_days"] < 2000))
    if "transit_duration_hours" in keep: conds.append((keep["transit_duration_hours"] > 0.2) & (keep["transit_duration_hours"] < 30))
    if "transit_depth_ppm" in keep:      conds.append((keep["transit_depth_ppm"] > 20) & (keep["transit_depth_ppm"] < 200000))  # 20 ppm – 20%
    if "planet_radius_earth" in keep:    conds.append((keep["planet_radius_earth"] > 0.2) & (keep["planet_radius_earth"] < 30))
    if "stellar_radius_solar" in keep:   conds.append((keep["stellar_radius_solar"] > 0.1) & (keep["stellar_radius_solar"] < 100))
    if "stellar_temperature_K" in keep:  conds.append((keep["stellar_temperature_K"] > 2300) & (keep["stellar_temperature_K"] < 15000))

    if conds:
        mask = np.logical_and.reduce(conds)
        keep = keep[mask]

    # Profundidad observada vs teórica (si disponible)
    if "depth_ratio" in keep:
        keep = keep[(keep["depth_ratio"] > 0.01) & (keep["depth_ratio"] < 100)]

    return keep

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Armonización + diagnósticos (KOI/K2/TOI)")
    ap.add_argument("--koi", type=str, required=True, help="Ruta KOI_All_Filtrated.csv")
    ap.add_argument("--k2",  type=str, required=True, help="Ruta K2_All_Filtrated.csv")
    ap.add_argument("--toi", type=str, required=True, help="Ruta TOI_All_Filtrated.csv")
    ap.add_argument("--outdir", type=str, required=True, help="Directorio de salida")
    ap.add_argument("--winsorize", action="store_true", help="Aplicar winsorización p0.5–p99.5")
    ap.add_argument("--add_logs", action="store_true", help="Agregar log10 a period/radius/depth")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    diag = {"notes": [], "missions": {}}

    # -----------------------
    # 1) Leer crudos
    koi_raw = read_csv_any(Path(args.koi))
    k2_raw  = read_csv_any(Path(args.k2))
    toi_raw = read_csv_any(Path(args.toi))

    # -----------------------
    # 2) Agregar columna fija con el identificador de misión
    koi_raw["source_mission"] = "KEPLER"  # KOI → KEPLER
    k2_raw["source_mission"]  = "K2"
    toi_raw["source_mission"] = "TOI"

    # -----------------------
    # 3) Preparar por misión
    koi_p = prepare_mission(koi_raw, "KOI", diag["missions"])
    k2_p  = prepare_mission(k2_raw,  "K2",  diag["missions"])
    toi_p = prepare_mission(toi_raw, "TOI", diag["missions"])

    # Concatenar y derivadas
    all_df = pd.concat([koi_p, k2_p, toi_p], ignore_index=True)
    all_df = add_physics(all_df)

    # Opcional: winsor y logs (añade columnas, no sustituye)
    if args.winsorize:
        for col in ["orbital_period_days","planet_radius_earth","transit_depth_ppm","transit_duration_hours"]:
            if col in all_df:
                all_df[f"{col}_wz"] = winsorize(all_df[col].astype(float))
        diag["notes"].append("winsorize p0.5–p99.5 aplicado a columnas clave (sufijo _wz).")

    if args.add_logs:
        for col in ["orbital_period_days","planet_radius_earth","transit_depth_ppm"]:
            if col in all_df:
                x = all_df[col].astype(float)
                all_df[f"log10_{col}"] = np.where(x > 0, np.log10(x), np.nan)
        diag["notes"].append("log10 agregado para period/radius/depth (prefijo log10_).")

    # DIAGNÓSTICOS: missingness y outliers
    miss = all_df.isna().mean().sort_values(ascending=False).rename("missing_ratio").reset_index()
    miss.columns = ["column","missing_ratio"]
    miss.to_csv(outdir / "Harmonized_MISSING.csv", index=False)

    num_cols = all_df.select_dtypes(include=[np.number]).columns.tolist()
    outliers = all_df[num_cols].apply(iqr_outlier_ratio).T.sort_values("ratio", ascending=False)
    outliers.to_csv(outdir / "Harmonized_OUTLIERS.csv", index=True)

    # Métricas de etiquetas
    disp_counts = all_df["final_disposition"].value_counts(dropna=False).to_dict()

    # Guardar FULL
    full_fp = outdir / "Harmonized_FULL.csv"
    all_df.to_csv(full_fp, index=False)

    # STRCIT para ML
    strict_df = strict_filter(all_df)
    strict_fp = outdir / "Harmonized_STRICT.csv"
    strict_df.to_csv(strict_fp, index=False)

    # DIAG JSON
    diag["row_counts"] = {
        "KOI_raw": int(len(koi_raw)),
        "K2_raw": int(len(k2_raw)),
        "TOI_raw": int(len(toi_raw)),
        "FULL": int(len(all_df)),
        "STRICT": int(len(strict_df))
    }
    diag["labels"] = disp_counts
    diag["top_missing"] = miss.head(15).to_dict(orient="records")
    diag["top_outliers"] = outliers.head(15).reset_index().rename(columns={"index":"column"}).to_dict(orient="records")
    (outdir / "Harmonized_DIAG.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    print("OK")
    print("FULL :", full_fp)
    print("STRICT:", strict_fp)
    print("MISSING CSV:", outdir / "Harmonized_MISSING.csv")
    print("OUTLIERS CSV:", outdir / "Harmonized_OUTLIERS.csv")
    print("DIAG JSON:", outdir / "Harmonized_DIAG.json")


if __name__ == "__main__":
    main()