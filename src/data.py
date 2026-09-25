from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    ACTIVE_DATA_PATH,
    BASE_DIR,
    CHEMICAL_PRICE_PER_KG,
    HALOGEN_COLUMNS,
    INPUT_COLUMNS,
    RANDOM_STATE,
    REQUIRED_COLUMNS,
    TARGET_COLUMNS,
)


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_validation: pd.DataFrame
    y_test: pd.DataFrame


def find_data_file(directory: Path = BASE_DIR) -> Path:
    if directory.resolve() == BASE_DIR.resolve() and ACTIVE_DATA_PATH.exists():
        return ACTIVE_DATA_PATH

    candidates = sorted(
        (
            p
            for pattern in ("*.csv", "*.xlsx")
            for p in directory.glob(pattern)
            if not p.name.startswith("~$")
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "Aucun fichier de données CSV ou Excel trouvé dans le projet."
        )
    return candidates[0]


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    source = Path(path) if path else find_data_file()
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source, encoding="utf-8-sig")
    elif source.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(source, engine="openpyxl")
    else:
        raise ValueError(f"Format de données non pris en charge : {source.suffix}")
    return clean_and_validate(frame)


def clean_and_validate(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.columns = [str(column).strip() for column in df.columns]

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError("Colonnes obligatoires absentes : " + ", ".join(missing_columns))

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=TARGET_COLUMNS).drop_duplicates().reset_index(drop=True)
    _validate_physical_units(df)

    # Ces deux sorties sont des identités métier, pas des cibles à apprendre.
    df["Toxicity_Index_pct"] = df[HALOGEN_COLUMNS].sum(axis=1, min_count=1)
    df["Cout_liquide_USD"] = (
        df["Concentration_liquide_chimique_kg"] * CHEMICAL_PRICE_PER_KG
    )
    return df


def _validate_physical_units(df: pd.DataFrame) -> None:
    """Détecte les valeurs incompatibles avec les unités déclarées."""
    invalid: dict[str, int] = {}
    bounded = {
        "Concentration_liquide_chimique_kg": (0.0, 10_000.0),
        "Temperature_C": (-273.15, 3_000.0),
        "Temps_reaction_min": (0.0, 100_000.0),
        "Heating_rate_C_min": (0.0, 10_000.0),
        "Taille_particules_mm": (0.0, 10_000.0),
    }
    for column, (lower, upper) in bounded.items():
        values = df[column].dropna()
        count = int(((values <= lower) | (values > upper)).sum())
        if count:
            invalid[column] = count
    for column in TARGET_COLUMNS:
        values = df[column].dropna()
        count = int(((values < 0.0) | (values > 100.0)).sum())
        if count:
            invalid[column] = count
    if invalid:
        details = ", ".join(
            f"{column}: {count}" for column, count in invalid.items()
        )
        raise ValueError(
            "Valeurs incompatibles avec les unités ou bornes physiques : "
            + details
        )


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        rows.append(
            {
                "Colonne": column,
                "Type": str(df[column].dtype),
                "Valeurs manquantes": int(df[column].isna().sum()),
                "Valeurs uniques": int(df[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def split_dataset(
    df: pd.DataFrame, random_state: int = RANDOM_STATE
) -> DataSplits:
    X = df[INPUT_COLUMNS].copy()
    y = df[TARGET_COLUMNS].copy()
    temperature_strata = pd.qcut(
        X["Temperature_C"],
        q=10,
        labels=False,
        duplicates="drop",
    )

    (
        X_train,
        X_holdout,
        y_train,
        y_holdout,
        _,
        holdout_strata,
    ) = train_test_split(
        X,
        y,
        temperature_strata,
        test_size=0.20,
        random_state=random_state,
        stratify=temperature_strata,
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.50,
        random_state=random_state,
        stratify=holdout_strata,
    )
    return DataSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )
