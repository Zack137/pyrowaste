"""Génère les preuves scientifiques du prototype multi-sorties Optuna.

Ce script ne réentraîne aucun modèle. Il consomme les résultats d'une
validation croisée imbriquée et répétée déjà calculée dans
``reports/nested_repeated_validation_budget_renforce``. Quand l'artefact final est disponible,
il ajoute des analyses d'inférence peu coûteuses : sensibilité un facteur à la
fois, échantillonnage de scénarios et front de Pareto exploratoire.

Sorties
-------
Les tableaux CSV et les figures PNG haute résolution sont écrits dans
``reports/optuna_scientific_evidence``. Les intervalles de confiance sont
calculés entre répétitions : les plis d'une même répétition sont d'abord
moyennés afin de ne pas les traiter comme des observations indépendantes.

Exemple
-------
    python experiments/generate_optuna_scientific_evidence.py

    python experiments/generate_optuna_scientific_evidence.py \
        --artifact artifacts/production_selected_bundle.joblib \
        --skip-learning-curve
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import unicodedata
import warnings

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import t as student_t
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    ARTIFACT_PATH,
    HALOGEN_COLUMNS,
    INPUT_COLUMNS,
    PRODUCTION_ARTIFACT_PATH,
    PRODUCTION_MODEL_NAME,
    RANDOM_STATE,
    TARGET_COLUMNS,
)
from src.data import load_dataset  # noqa: E402
from src.modeling import predict_primary  # noqa: E402


DEFAULT_NESTED_DIR = (
    ROOT / "reports" / "nested_repeated_validation_budget_renforce"
)
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "optuna_scientific_evidence"
REPEAT_METRIC_FILES = {
    "global": "repeat_global_metrics.csv",
    "output": "repeat_output_metrics.csv",
}
OUTER_FOLD_METRIC_FILES = {
    "global": "outer_fold_global_metrics.csv",
    "output": "outer_fold_output_metrics.csv",
}
PREDICTIONS_FILE = "outer_fold_predictions.csv"
LEARNING_CURVE_CANDIDATES = (
    "learning_curve_metrics.csv",
    "learning_curve.csv",
    "nested_learning_curve_metrics.csv",
)

DISPLAY_TARGETS = {
    "Solid_Yield_pct": "Rendement solide",
    "Liquid_Yield_pct": "Rendement liquide",
    "Gas_Yield_pct": "Rendement gaz",
    "HBr_pct": "HBr",
    "Br2_pct": "Br2",
    "HCl_pct": "HCl",
    "Cl2_pct": "Cl2",
    "HF_pct": "HF",
}
DISPLAY_INPUTS = {
    "Concentration_liquide_chimique_kg": "Concentration liquide (kg)",
    "Temperature_C": "Température (°C)",
    "Temps_reaction_min": "Temps de réaction (min)",
    "Heating_rate_C_min": "Vitesse de chauffage (°C/min)",
    "Taille_particules_mm": "Taille des particules (mm)",
}
SHORT_MODELS = {
    "Naive mean": "Naïf",
    "Naive moyenne": "Naïf",
    "Prédicteur naïf (moyenne)": "Naïf",
    "Ridge polynomial Optuna": "Ridge",
    "Ridge polynomial": "Ridge",
    "Extra Trees Optuna": "Extra Trees",
    "MLP Optuna": "MLP",
    "Hybride 2 branches MLP + XGBoost Optuna": "Hybride MLP+XGB",
    "Automatic selection": "Sélection auto.",
    PRODUCTION_MODEL_NAME: "Sélection auto.",
}

NAVY = "#17324D"
TEAL = "#007C83"
ORANGE = "#E58A1F"
RED = "#C44E52"
GREEN = "#2A9D6F"
PURPLE = "#7A5AA6"
BLUE = "#4C78A8"
GRAY = "#61717D"
LIGHT_GRAY = "#E3E8EB"
PALE = "#F7FAFA"
MODEL_COLORS = {
    "Naive mean": GRAY,
    "Naive moyenne": GRAY,
    "Prédicteur naïf (moyenne)": GRAY,
    "Ridge polynomial Optuna": BLUE,
    "Ridge polynomial": BLUE,
    "Extra Trees Optuna": GREEN,
    "MLP Optuna": ORANGE,
    "Hybride 2 branches MLP + XGBoost Optuna": PURPLE,
    "Automatic selection": TEAL,
    PRODUCTION_MODEL_NAME: TEAL,
}
MODEL_DISPLAY_ORDER = (
    "Prédicteur naïf (moyenne)",
    "Naive mean",
    "Naive moyenne",
    "Ridge polynomial Optuna",
    "Ridge polynomial",
    "Extra Trees Optuna",
    "MLP Optuna",
    "Hybride 2 branches MLP + XGBoost Optuna",
    PRODUCTION_MODEL_NAME,
    "Automatic selection",
)
INPUT_COLORS = {
    "Concentration_liquide_chimique_kg": BLUE,
    "Temperature_C": RED,
    "Temps_reaction_min": GREEN,
    "Heating_rate_C_min": ORANGE,
    "Taille_particules_mm": PURPLE,
}


class EvidenceInputError(RuntimeError):
    """Erreur d'entrée concise destinée à l'utilisateur du script."""


def _slug(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return "".join(character.lower() for character in text if character.isalnum())


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        raise EvidenceInputError(f"CSV illisible : {path}\nDétail : {exc}") from exc


def _rename_aliases(
    frame: pd.DataFrame,
    aliases: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    available = {_slug(column): column for column in frame.columns}
    renaming: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in (canonical, *candidates):
            source = available.get(_slug(candidate))
            if source is not None:
                renaming[source] = canonical
                break
    return frame.rename(columns=renaming)


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    source: Path,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise EvidenceInputError(
            f"Schéma incomplet dans {source} : colonnes absentes "
            + ", ".join(missing)
            + f". Colonnes trouvées : {', '.join(map(str, frame.columns))}"
        )


def _canonical_global(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    result = _rename_aliases(
        frame,
        {
            "model": ("modele", "modèle", "family", "model_name"),
            "repeat": ("repetition", "répétition", "outer_repeat"),
            "outer_fold": ("fold", "pli", "outerfold"),
            "mae_normalized": (
                "normalized_mae",
                "mae_norm",
                "outer_mae_normalized",
            ),
            "rmse_normalized": (
                "normalized_rmse",
                "rmse_norm",
                "outer_rmse_normalized",
            ),
            "r2_macro": ("macro_r2", "r2global", "outer_r2_macro"),
            "fit_seconds": (
                "training_seconds",
                "temps_entrainement_s",
                "elapsed_seconds",
                "optimization_and_fit_seconds",
            ),
        },
    ).copy()
    _require_columns(
        result,
        [
            "model",
            "repeat",
            "mae_normalized",
            "rmse_normalized",
            "r2_macro",
        ],
        source=source,
    )
    for column in (
        "repeat",
        "outer_fold",
        "mae_normalized",
        "rmse_normalized",
        "r2_macro",
        "fit_seconds",
    ):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result["model"] = result["model"].astype(str).str.strip()
    result["aggregation_unit"] = (
        "moyenne des plis externes par répétition"
        if "outer_fold" in result.columns
        else "répétition OOF complète"
    )
    result = result.dropna(
        subset=[
            "model",
            "repeat",
            "mae_normalized",
            "rmse_normalized",
            "r2_macro",
        ]
    )
    if result.empty:
        raise EvidenceInputError(f"Aucune métrique exploitable dans {source}.")
    return result


def _canonical_output(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    result = _rename_aliases(
        frame,
        {
            "model": ("modele", "modèle", "family", "model_name"),
            "repeat": ("repetition", "répétition", "outer_repeat"),
            "outer_fold": ("fold", "pli", "outerfold"),
            "target": ("sortie", "output", "cible"),
            "mae": ("mean_absolute_error",),
            "rmse": ("root_mean_squared_error",),
            "r2": ("r2_score", "coefficient_determination"),
        },
    ).copy()
    _require_columns(
        result,
        ["model", "repeat", "target", "mae", "rmse", "r2"],
        source=source,
    )
    for column in ("repeat", "outer_fold", "mae", "rmse", "r2"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result["model"] = result["model"].astype(str).str.strip()
    result["target"] = result["target"].astype(str).str.strip()
    result["aggregation_unit"] = (
        "moyenne des plis externes par répétition"
        if "outer_fold" in result.columns
        else "répétition OOF complète"
    )
    result = result.dropna(
        subset=["model", "repeat", "target", "mae", "rmse", "r2"]
    )
    unknown = sorted(set(result["target"]) - set(TARGET_COLUMNS))
    if unknown:
        warnings.warn(
            "Sorties inconnues ignorées dans les métriques : " + ", ".join(unknown),
            stacklevel=2,
        )
        result = result[result["target"].isin(TARGET_COLUMNS)]
    missing_targets = [target for target in TARGET_COLUMNS if target not in set(result["target"])]
    if missing_targets:
        raise EvidenceInputError(
            f"Les huit sorties ne sont pas présentes dans {source}. "
            "Sorties absentes : "
            + ", ".join(missing_targets)
        )
    return result


def _canonical_predictions(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    result = _rename_aliases(
        frame,
        {
            "model": ("modele", "modèle", "family", "model_name"),
            "repeat": ("repetition", "répétition", "outer_repeat"),
            "outer_fold": ("fold", "pli", "outerfold"),
            "row_index": (
                "sample_index",
                "observation_index",
                "record_index",
                "index",
            ),
            "target": ("sortie", "output", "cible"),
            "observed": ("y_true", "truth", "observe", "observé"),
            "predicted": ("y_pred", "prediction", "predit", "prédit"),
        },
    ).copy()

    # Le format long est le format officiel. Un format large Observed__/Predicted__
    # reste accepté afin de faciliter la migration des anciens diagnostics.
    if {"target", "observed", "predicted"}.issubset(result.columns):
        _require_columns(
            result,
            [
                "model",
                "repeat",
                "outer_fold",
                "row_index",
                "target",
                "observed",
                "predicted",
            ],
            source=source,
        )
    else:
        observed_columns = {
            column.split("__", 1)[1]: column
            for column in result.columns
            if str(column).startswith("Observed__")
        }
        predicted_columns = {
            column.split("__", 1)[1]: column
            for column in result.columns
            if str(column).startswith("Predicted__")
            and not str(column).startswith("Predicted_raw__")
        }
        targets = [
            target
            for target in TARGET_COLUMNS
            if target in observed_columns and target in predicted_columns
        ]
        if len(targets) != len(TARGET_COLUMNS):
            raise EvidenceInputError(
                f"Schéma des prédictions non reconnu dans {source}. "
                "Format attendu : model, repeat, outer_fold, row_index, target, "
                "observed, predicted."
            )
        id_columns = [
            column
            for column in (
                "model",
                "repeat",
                "outer_fold",
                "row_index",
                *INPUT_COLUMNS,
            )
            if column in result.columns
        ]
        if "row_index" not in id_columns:
            result["row_index"] = np.arange(len(result))
            id_columns.append("row_index")
        if "model" not in id_columns:
            result["model"] = "Automatic selection"
            id_columns.append("model")
        if "repeat" not in id_columns:
            result["repeat"] = 1
            id_columns.append("repeat")
        if "outer_fold" not in id_columns:
            result["outer_fold"] = 1
            id_columns.append("outer_fold")
        rows: list[pd.DataFrame] = []
        for target in targets:
            part = result[id_columns].copy()
            part["target"] = target
            part["observed"] = result[observed_columns[target]]
            part["predicted"] = result[predicted_columns[target]]
            rows.append(part)
        result = pd.concat(rows, ignore_index=True)

    for column in ("repeat", "outer_fold", "row_index", "observed", "predicted"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["model"] = result["model"].astype(str).str.strip()
    result["target"] = result["target"].astype(str).str.strip()
    result = result.dropna(
        subset=[
            "model",
            "repeat",
            "outer_fold",
            "row_index",
            "target",
            "observed",
            "predicted",
        ]
    )
    result = result[result["target"].isin(TARGET_COLUMNS)]
    missing_targets = [target for target in TARGET_COLUMNS if target not in set(result["target"])]
    if missing_targets:
        raise EvidenceInputError(
            f"Prédictions OOF incomplètes dans {source}. Sorties absentes : "
            + ", ".join(missing_targets)
        )
    result["row_index"] = result["row_index"].astype(int)
    return result


def load_nested_results(nested_dir: Path) -> dict[str, object]:
    if not nested_dir.exists() or not nested_dir.is_dir():
        raise EvidenceInputError(
            "Résultats de validation imbriquée répétée absents. "
            f"Dossier attendu : {nested_dir}\n"
            "Exécutez d'abord le protocole de validation imbriquée et répétée; "
            "ce générateur ne lance aucun entraînement."
        )

    predictions_path = nested_dir / PREDICTIONS_FILE
    global_path = next(
        (
            nested_dir / filename
            for filename in (
                REPEAT_METRIC_FILES["global"],
                OUTER_FOLD_METRIC_FILES["global"],
            )
            if (nested_dir / filename).exists()
        ),
        None,
    )
    output_path = next(
        (
            nested_dir / filename
            for filename in (
                REPEAT_METRIC_FILES["output"],
                OUTER_FOLD_METRIC_FILES["output"],
            )
            if (nested_dir / filename).exists()
        ),
        None,
    )
    missing_descriptions: list[str] = []
    if global_path is None:
        missing_descriptions.append(
            f"{REPEAT_METRIC_FILES['global']} ou "
            f"{OUTER_FOLD_METRIC_FILES['global']}"
        )
    if output_path is None:
        missing_descriptions.append(
            f"{REPEAT_METRIC_FILES['output']} ou "
            f"{OUTER_FOLD_METRIC_FILES['output']}"
        )
    if not predictions_path.exists():
        missing_descriptions.append(PREDICTIONS_FILE)
    if missing_descriptions:
        raise EvidenceInputError(
            "Résultats de validation imbriquée répétée incomplets. "
            "Fichiers obligatoires absents :\n- "
            + "\n- ".join(missing_descriptions)
            + "\nRelancez le protocole de validation avant de générer les preuves."
        )

    assert global_path is not None
    assert output_path is not None
    global_metrics = _canonical_global(_read_csv(global_path), global_path)
    output_metrics = _canonical_output(_read_csv(output_path), output_path)
    predictions = _canonical_predictions(_read_csv(predictions_path), predictions_path)

    repeat_count = int(global_metrics["repeat"].nunique())
    if repeat_count < 2:
        raise EvidenceInputError(
            "La validation trouvée n'est pas répétée : "
            f"{repeat_count} répétition détectée dans {global_path}. "
            "Au moins deux répétitions sont nécessaires pour estimer un IC95 % "
            "entre répétitions."
        )

    metadata: dict[str, object] = {}
    metadata_path = nested_dir / "run_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.warn(
                f"Métadonnées ignorées car illisibles ({metadata_path}) : {exc}",
                stacklevel=2,
            )

    learning: pd.DataFrame | None = None
    learning_path: Path | None = None
    for candidate in LEARNING_CURVE_CANDIDATES:
        path = nested_dir / candidate
        if path.exists():
            learning_path = path
            learning = _read_csv(path)
            break

    return {
        "global": global_metrics,
        "output": output_metrics,
        "predictions": predictions,
        "global_path": global_path,
        "output_path": output_path,
        "predictions_path": predictions_path,
        "metadata": metadata,
        "learning": learning,
        "learning_path": learning_path,
    }


def _ci95(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    count = len(numeric)
    if count == 0:
        return {
            "n_repeats": 0,
            "mean": np.nan,
            "std": np.nan,
            "se": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }
    mean = float(np.mean(numeric))
    if count == 1:
        return {
            "n_repeats": 1,
            "mean": mean,
            "std": np.nan,
            "se": np.nan,
            "ci95_low": mean,
            "ci95_high": mean,
        }
    std = float(np.std(numeric, ddof=1))
    standard_error = std / math.sqrt(count)
    half_width = float(student_t.ppf(0.975, count - 1) * standard_error)
    return {
        "n_repeats": count,
        "mean": mean,
        "std": std,
        "se": standard_error,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def _summary_aggregation_unit(frame: pd.DataFrame) -> str:
    if "aggregation_unit" not in frame.columns:
        return "moyenne des plis externes par répétition"
    units = frame["aggregation_unit"].dropna().astype(str).unique().tolist()
    if units == ["répétition OOF complète"]:
        return "répétition OOF complète"
    return "moyenne des plis externes par répétition"


def summarize_global(global_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in ("mae_normalized", "rmse_normalized", "r2_macro", "fit_seconds")
        if column in global_metrics.columns
    ]
    per_repeat = (
        global_metrics.groupby(["model", "repeat"], as_index=False)[metric_columns]
        .mean(numeric_only=True)
    )
    aggregation_unit = _summary_aggregation_unit(global_metrics)
    rows: list[dict[str, object]] = []
    for model, model_frame in per_repeat.groupby("model", sort=False):
        for metric in metric_columns:
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "aggregation_unit": aggregation_unit,
                    **_ci95(model_frame[metric]),
                }
            )
    return pd.DataFrame(rows)


def summarize_outputs(output_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["mae", "rmse", "r2"]
    per_repeat = (
        output_metrics.groupby(
            ["model", "target", "repeat"],
            as_index=False,
        )[metric_columns]
        .mean(numeric_only=True)
    )
    aggregation_unit = _summary_aggregation_unit(output_metrics)
    rows: list[dict[str, object]] = []
    for (model, target), group in per_repeat.groupby(
        ["model", "target"],
        sort=False,
    ):
        for metric in metric_columns:
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "metric": metric,
                    "aggregation_unit": aggregation_unit,
                    **_ci95(group[metric]),
                }
            )
    return pd.DataFrame(rows)


def _match_label(requested: str, available: list[str], label: str) -> str:
    if requested in available:
        return requested
    by_slug = {_slug(value): value for value in available}
    match = by_slug.get(_slug(requested))
    if match is not None:
        return match
    raise EvidenceInputError(
        f"{label} inconnu : {requested!r}. Valeurs disponibles : "
        + ", ".join(available)
    )


def select_diagnostic_model(
    predictions: pd.DataFrame,
    summary_global: pd.DataFrame,
    requested: str | None,
) -> str:
    available = list(dict.fromkeys(predictions["model"].astype(str).tolist()))
    if requested:
        return _match_label(requested, available, "Modèle diagnostique")

    automatic_slugs = {
        _slug("Automatic selection"),
        _slug(PRODUCTION_MODEL_NAME),
        _slug("Sélection automatique"),
    }
    for model in available:
        if _slug(model) in automatic_slugs:
            return model

    r2_rows = summary_global[
        (summary_global["metric"] == "r2_macro")
        & (summary_global["model"].isin(available))
    ].copy()
    non_naive = r2_rows[
        ~r2_rows["model"].map(lambda value: "naive" in _slug(value))
    ]
    candidates = non_naive if not non_naive.empty else r2_rows
    if candidates.empty:
        return available[0]
    return str(candidates.sort_values("mean", ascending=False).iloc[0]["model"])


def load_final_bundle(path: Path | None) -> tuple[object | None, Path | None]:
    candidates = [path] if path is not None else [
        PRODUCTION_ARTIFACT_PATH,
        ARTIFACT_PATH,
    ]
    existing = [candidate for candidate in candidates if candidate is not None and candidate.exists()]
    if not existing:
        warnings.warn(
            "Artefact final absent : les analyses de sensibilité, de scénarios "
            "et de Pareto seront omises. Chemins examinés : "
            + ", ".join(str(candidate) for candidate in candidates if candidate is not None),
            stacklevel=2,
        )
        return None, None
    artifact_path = existing[0]
    try:
        bundle = joblib.load(artifact_path)
    except Exception as exc:
        raise EvidenceInputError(
            f"Impossible de charger l'artefact final {artifact_path} : {exc}"
        ) from exc
    required_attributes = ("models", "reference_data", "input_bounds")
    missing = [name for name in required_attributes if not hasattr(bundle, name)]
    if missing:
        raise EvidenceInputError(
            f"L'artefact {artifact_path} n'est pas un ModelBundle final. "
            "Attributs absents : "
            + ", ".join(missing)
        )
    return bundle, artifact_path


def reference_data(bundle: object | None) -> pd.DataFrame:
    if bundle is not None:
        candidate = getattr(bundle, "reference_data", None)
        if isinstance(candidate, pd.DataFrame) and set(INPUT_COLUMNS).issubset(candidate.columns):
            return candidate.reset_index(drop=True).copy()
    return load_dataset().reset_index(drop=True)


def aggregate_diagnostic_predictions(
    predictions: pd.DataFrame,
    *,
    model: str,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    selected = predictions[predictions["model"] == model].copy()
    if selected.empty:
        raise EvidenceInputError(f"Aucune prédiction OOF pour le modèle {model!r}.")
    conflicts = (
        selected.groupby(["row_index", "target"])["observed"]
        .nunique(dropna=False)
        .gt(1)
    )
    if bool(conflicts.any()):
        raise EvidenceInputError(
            "Des valeurs observées incompatibles ont été trouvées pour un même "
            "couple row_index/target dans les prédictions OOF."
        )
    aggregated = (
        selected.groupby(["row_index", "target"], as_index=False)
        .agg(
            observed=("observed", "first"),
            predicted=("predicted", "mean"),
            prediction_std_between_repeats=("predicted", "std"),
            n_oof_predictions=("predicted", "size"),
        )
    )
    aggregated["residual"] = aggregated["observed"] - aggregated["predicted"]
    aggregated.insert(0, "model", model)

    reference_inputs = reference[INPUT_COLUMNS].copy().reset_index(drop=True)
    reference_inputs.insert(0, "row_index", np.arange(len(reference_inputs)))
    invalid_indices = sorted(
        set(aggregated["row_index"]) - set(reference_inputs["row_index"])
    )
    if invalid_indices:
        preview = ", ".join(map(str, invalid_indices[:8]))
        raise EvidenceInputError(
            "Impossible de relier les prédictions OOF aux entrées : row_index "
            f"hors de la base de référence ({preview})."
        )
    aggregated = aggregated.merge(
        reference_inputs,
        on="row_index",
        how="left",
        validate="many_to_one",
    )
    missing_targets = [
        target for target in TARGET_COLUMNS if target not in set(aggregated["target"])
    ]
    if missing_targets:
        raise EvidenceInputError(
            "Diagnostics OOF incomplets pour le modèle sélectionné. Sorties absentes : "
            + ", ".join(missing_targets)
        )
    return aggregated


def residual_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in TARGET_COLUMNS:
        values = diagnostics.loc[
            diagnostics["target"] == target,
            "residual",
        ].dropna()
        rows.append(
            {
                "model": diagnostics["model"].iloc[0],
                "target": target,
                "n": int(len(values)),
                "mean_residual": float(values.mean()),
                "std_residual": float(values.std(ddof=1)),
                "median_residual": float(values.median()),
                "q05_residual": float(values.quantile(0.05)),
                "q95_residual": float(values.quantile(0.95)),
                "mae_from_aggregated_oof": float(values.abs().mean()),
                "rmse_from_aggregated_oof": float(np.sqrt(np.mean(np.square(values)))),
            }
        )
    return pd.DataFrame(rows)


def residual_input_associations(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in TARGET_COLUMNS:
        part = diagnostics[diagnostics["target"] == target]
        for input_name in (
            "Temperature_C",
            "Concentration_liquide_chimique_kg",
        ):
            clean = part[[input_name, "residual"]].dropna()
            rows.append(
                {
                    "model": diagnostics["model"].iloc[0],
                    "target": target,
                    "input": input_name,
                    "n": len(clean),
                    "pearson_r": float(
                        clean[input_name].corr(clean["residual"], method="pearson")
                    ),
                    "spearman_rho": float(
                        clean[input_name].corr(clean["residual"], method="spearman")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _hex_rgb(value: str) -> tuple[int, int, int]:
    cleaned = value.lstrip("#")
    return tuple(int(cleaned[index : index + 2], 16) for index in (0, 2, 4))


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path(
            "C:/Windows/Fonts/arialbd.ttf"
            if bold
            else "C:/Windows/Fonts/arial.ttf"
        ),
        Path(
            "C:/Windows/Fonts/calibrib.ttf"
            if bold
            else "C:/Windows/Fonts/calibri.ttf"
        ),
        Path("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    return ImageFont.load_default()


def _save_png(image: Image.Image, path: Path, dpi: int) -> None:
    if image.mode != "RGB":
        background = Image.new("RGB", image.size, "white")
        if image.mode == "RGBA":
            background.paste(image, mask=image.getchannel("A"))
        else:
            background.paste(image)
        image = background
    image.save(path, format="PNG", dpi=(dpi, dpi), optimize=True)


def _fitted_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    initial_size: int,
    max_width: float,
    bold: bool = False,
    minimum_size: int = 14,
) -> ImageFont.ImageFont:
    size = initial_size
    font = _font(size, bold)
    while size > minimum_size and draw.textlength(text, font=font) > max_width:
        size -= 1
        font = _font(size, bold)
    return font


def _canvas(
    title: str,
    subtitle: str,
    *,
    size: tuple[int, int],
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image, "RGBA")
    left_margin = 76
    title_font = _fitted_font(
        draw,
        title,
        initial_size=42,
        max_width=size[0] - 2 * left_margin,
        bold=True,
        minimum_size=30,
    )
    subtitle_font = _fitted_font(
        draw,
        subtitle,
        initial_size=23,
        max_width=size[0] - 2 * left_margin,
        minimum_size=17,
    )
    draw.text((left_margin, 38), title, font=title_font, fill=NAVY)
    draw.text((left_margin, 96), subtitle, font=subtitle_font, fill=GRAY)
    draw.line(
        (left_margin, 140, size[0] - left_margin, 140),
        fill=LIGHT_GRAY,
        width=2,
    )
    return image, draw


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    return float(draw.textlength(text, font=font))


def _panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
) -> tuple[int, int, int, int]:
    draw.rounded_rectangle(
        box,
        radius=16,
        outline="#C8D3D8",
        width=2,
        fill=PALE,
    )
    header_bottom = box[1] + 58
    draw.rectangle(
        (box[0] + 3, box[1] + 3, box[2] - 3, header_bottom),
        fill="#EEF4F5",
    )
    draw.line(
        (box[0] + 3, header_bottom, box[2] - 3, header_bottom),
        fill="#C8D3D8",
        width=2,
    )
    title_font = _fitted_font(
        draw,
        title,
        initial_size=22,
        max_width=box[2] - box[0] - 40,
        bold=True,
        minimum_size=16,
    )
    draw.text((box[0] + 20, box[1] + 14), title, font=title_font, fill=NAVY)
    return (box[0] + 82, box[1] + 86, box[2] - 25, box[3] - 62)


def _target_panel_title(target: str) -> str:
    return f"{DISPLAY_TARGETS[target]} (%)"


def _draw_top_right_note(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    text: str,
    *,
    font_size: int = 14,
) -> None:
    font = _font(font_size, True)
    padding_x = 10
    padding_y = 6
    width = _text_width(draw, text, font) + 2 * padding_x
    height = font_size + 2 * padding_y + 3
    right = plot[2] - 8
    left = max(plot[0] + 8, right - width)
    top = plot[1] + 8
    draw.rounded_rectangle(
        (left, top, right, top + height),
        radius=7,
        fill=(255, 255, 255, 225),
        outline="#C8D3D8",
        width=1,
    )
    draw.text(
        (left + padding_x, top + padding_y - 1),
        text,
        font=font,
        fill=NAVY,
    )


def _finite_limits(
    values: list[float] | np.ndarray,
    *,
    include_zero: bool = False,
    padding: float = 0.08,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return (0.0, 1.0)
    low = float(np.min(array))
    high = float(np.max(array))
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    span = high - low
    if span <= 1e-12:
        span = max(abs(low), 1.0)
    return low - padding * span, high + padding * span


def _mapper(
    low: float,
    high: float,
    pixel_low: float,
    pixel_high: float,
) -> callable:
    span = max(high - low, 1e-12)
    return lambda value: pixel_low + (float(value) - low) / span * (
        pixel_high - pixel_low
    )


def _draw_xy_axes(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    xlabel: str,
    ylabel: str,
    x_ticks: int = 5,
    y_ticks: int = 5,
) -> tuple[callable, callable]:
    x0, y0, x1, y1 = plot
    draw.line((x0, y1, x1, y1), fill=NAVY, width=2)
    draw.line((x0, y0, x0, y1), fill=NAVY, width=2)
    x_map = _mapper(xlim[0], xlim[1], x0, x1)
    y_map = _mapper(ylim[0], ylim[1], y1, y0)
    tick_font = _font(13)
    for value in np.linspace(xlim[0], xlim[1], x_ticks):
        x = x_map(value)
        draw.line((x, y0, x, y1), fill=LIGHT_GRAY, width=1)
        label = f"{value:.3g}"
        draw.text(
            (x - _text_width(draw, label, tick_font) / 2, y1 + 8),
            label,
            font=tick_font,
            fill=GRAY,
        )
    for value in np.linspace(ylim[0], ylim[1], y_ticks):
        y = y_map(value)
        draw.line((x0, y, x1, y), fill=LIGHT_GRAY, width=1)
        label = f"{value:.3g}"
        draw.text(
            (x0 - _text_width(draw, label, tick_font) - 9, y - 8),
            label,
            font=tick_font,
            fill=GRAY,
        )
    axis_font = _font(15, True)
    draw.text(
        (
            (x0 + x1) / 2 - _text_width(draw, xlabel, axis_font) / 2,
            y1 + 32,
        ),
        xlabel,
        font=axis_font,
        fill=NAVY,
    )
    # Le libellé vertical est placé horizontalement juste au-dessus du tracé :
    # il reste lisible sans partager l'espace des annotations statistiques.
    if ylabel:
        draw.text((x0, y0 - 22), ylabel, font=_font(13, True), fill=NAVY)
    return x_map, y_map


def _grid_boxes(
    width: int,
    height: int,
    *,
    rows: int,
    columns: int,
    top: int = 165,
    side: int = 45,
    bottom: int = 45,
    gap_x: int = 24,
    gap_y: int = 24,
) -> list[tuple[int, int, int, int]]:
    panel_width = (width - 2 * side - (columns - 1) * gap_x) / columns
    panel_height = (
        height - top - bottom - (rows - 1) * gap_y
    ) / rows
    boxes = []
    for row in range(rows):
        for column in range(columns):
            x0 = int(side + column * (panel_width + gap_x))
            y0 = int(top + row * (panel_height + gap_y))
            boxes.append(
                (
                    x0,
                    y0,
                    int(x0 + panel_width),
                    int(y0 + panel_height),
                )
            )
    return boxes


def _ordered_models(values: list[str] | pd.Series) -> list[str]:
    unique = list(dict.fromkeys(map(str, values)))
    rank_by_slug: dict[str, int] = {}
    rank = 0
    for label in MODEL_DISPLAY_ORDER:
        key = _slug(label)
        if key not in rank_by_slug:
            rank_by_slug[key] = rank
            rank += 1
    source_order = {model: index for index, model in enumerate(unique)}
    return sorted(
        unique,
        key=lambda model: (
            rank_by_slug.get(_slug(model), len(rank_by_slug)),
            source_order[model],
        ),
    )


def _model_color(model: str, index: int = 0) -> str:
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    colors_by_slug = {_slug(label): color for label, color in MODEL_COLORS.items()}
    if _slug(model) in colors_by_slug:
        return colors_by_slug[_slug(model)]
    palette = (BLUE, GREEN, ORANGE, TEAL, PURPLE, RED, GRAY)
    return palette[index % len(palette)]


def _short_model(model: str) -> str:
    if model in SHORT_MODELS:
        return SHORT_MODELS[model]
    short_by_slug = {_slug(label): value for label, value in SHORT_MODELS.items()}
    if _slug(model) in short_by_slug:
        return short_by_slug[_slug(model)]
    compact = str(model).replace(" Optuna", "")
    return compact if len(compact) <= 22 else compact[:20] + "…"


def plot_model_comparison(
    summary: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> None:
    frame = summary[summary["metric"] == "r2_macro"].copy()
    if frame.empty:
        raise EvidenceInputError("Résumé R² macro vide : comparaison impossible.")
    models = _ordered_models(frame["model"])
    frame = frame.set_index("model").reindex(models).reset_index()
    oof_complete = (
        "aggregation_unit" in frame
        and set(frame["aggregation_unit"].dropna()) == {"répétition OOF complète"}
    )
    subtitle = (
        "R² macro sur OOF complets; barres d'erreur = IC95 % entre répétitions"
        if oof_complete
        else "R² macro moyen par répétition; barres d'erreur = IC95 %"
    )
    image, draw = _canvas(
        "Comparaison des modèles — validation imbriquée répétée",
        subtitle,
        size=(2200, 1350),
    )
    plot = (650, 210, 2100, 1170)
    limits = _finite_limits(
        frame[["ci95_low", "ci95_high"]].to_numpy().ravel(),
        include_zero=True,
        padding=0.06,
    )
    x_map, _ = _draw_xy_axes(
        draw,
        plot,
        limits,
        (0, len(frame)),
        xlabel="R² macro",
        ylabel="",
        x_ticks=7,
        y_ticks=1,
    )
    zero = x_map(0.0)
    draw.line((zero, plot[1], zero, plot[3]), fill=NAVY, width=3)
    slot = (plot[3] - plot[1]) / max(len(frame), 1)
    label_font = _font(24, True)
    value_font = _font(18, True)
    for index, row in frame.iterrows():
        center_y = plot[1] + (index + 0.5) * slot
        bar_height = min(62, slot * 0.56)
        value_x = x_map(row["mean"])
        left, right = sorted((zero, value_x))
        color = _model_color(str(row["model"]), index)
        draw.rounded_rectangle(
            (left, center_y - bar_height / 2, right, center_y + bar_height / 2),
            radius=10,
            fill=color,
        )
        low_x = x_map(row["ci95_low"])
        high_x = x_map(row["ci95_high"])
        draw.line((low_x, center_y, high_x, center_y), fill=NAVY, width=4)
        draw.line(
            (low_x, center_y - 14, low_x, center_y + 14),
            fill=NAVY,
            width=4,
        )
        draw.line(
            (high_x, center_y - 14, high_x, center_y + 14),
            fill=NAVY,
            width=4,
        )
        label = _short_model(str(row["model"]))
        draw.text(
            (
                plot[0] - _text_width(draw, label, label_font) - 24,
                center_y - 14,
            ),
            label,
            font=label_font,
            fill=NAVY,
        )
        value_label = f"{row['mean']:.3f}"
        value_label_x = value_x + 12 if row["mean"] >= 0 else value_x - 85
        draw.text(
            (value_label_x, center_y - 13),
            value_label,
            font=value_font,
            fill=NAVY,
        )
    _save_png(image, path, dpi)


def plot_metric_by_output(
    summary: pd.DataFrame,
    metric: str,
    path: Path,
    *,
    dpi: int,
) -> None:
    metric_labels = {
        "r2": "R²",
        "rmse": "RMSE (points de %)",
        "mae": "MAE (points de %)",
    }
    frame = summary[summary["metric"] == metric].copy()
    models = _ordered_models(frame["model"])
    title_label = {"r2": "R²", "rmse": "RMSE", "mae": "MAE"}[metric]
    oof_complete = (
        "aggregation_unit" in frame
        and set(frame["aggregation_unit"].dropna()) == {"répétition OOF complète"}
    )
    subtitle = (
        "Métrique sur OOF complets; traits horizontaux = IC95 % entre répétitions"
        if oof_complete
        else "Moyenne des plis par répétition; traits horizontaux = IC95 %"
    )
    image, draw = _canvas(
        f"{title_label} par sortie et par modèle",
        subtitle,
        size=(2800, 1750),
    )
    boxes = _grid_boxes(2800, 1750, rows=2, columns=4)
    for target, box in zip(TARGET_COLUMNS, boxes):
        target_frame = (
            frame[frame["target"] == target]
            .set_index("model")
            .reindex(models)
            .reset_index()
        )
        plot = _panel(draw, box, _target_panel_title(target))
        values = target_frame[["ci95_low", "ci95_high"]].to_numpy().ravel()
        limits = _finite_limits(
            values,
            include_zero=(metric in {"r2", "mae", "rmse"}),
            padding=0.08,
        )
        if metric in {"mae", "rmse"}:
            limits = (0.0, max(limits[1], 1e-9))
        x_map, _ = _draw_xy_axes(
            draw,
            plot,
            limits,
            (0, len(models)),
            xlabel=metric_labels[metric],
            ylabel="",
            x_ticks=4,
            y_ticks=1,
        )
        if limits[0] <= 0 <= limits[1]:
            zero = x_map(0)
            draw.line((zero, plot[1], zero, plot[3]), fill=NAVY, width=2)
        slot = (plot[3] - plot[1]) / max(len(models), 1)
        for index, row in target_frame.iterrows():
            if not np.isfinite(row["mean"]):
                continue
            center_y = plot[1] + (index + 0.5) * slot
            color = _model_color(str(row["model"]), index)
            left = x_map(max(0.0, limits[0])) if row["mean"] >= 0 else x_map(row["mean"])
            right = x_map(row["mean"]) if row["mean"] >= 0 else x_map(0)
            draw.rounded_rectangle(
                (
                    min(left, right),
                    center_y - min(22, slot * 0.24),
                    max(left, right),
                    center_y + min(22, slot * 0.24),
                ),
                radius=6,
                fill=color,
            )
            low_x = x_map(row["ci95_low"])
            high_x = x_map(row["ci95_high"])
            draw.line((low_x, center_y, high_x, center_y), fill=NAVY, width=3)
            draw.line(
                (low_x, center_y - 8, low_x, center_y + 8),
                fill=NAVY,
                width=3,
            )
            draw.line(
                (high_x, center_y - 8, high_x, center_y + 8),
                fill=NAVY,
                width=3,
            )
            label = _short_model(str(row["model"]))
            draw.text(
                (plot[0] + 8, center_y - 24),
                label,
                font=_font(12, True),
                fill=NAVY,
            )
    _save_png(image, path, dpi)


def plot_observed_predicted(
    diagnostics: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> None:
    model = str(diagnostics["model"].iloc[0])
    image, draw = _canvas(
        "Observé versus prédit — huit sorties",
        f"Prédictions hors pli moyennées par observation; modèle : {_short_model(model)}",
        size=(2800, 1750),
    )
    boxes = _grid_boxes(2800, 1750, rows=2, columns=4)
    for target, box in zip(TARGET_COLUMNS, boxes):
        part = diagnostics[diagnostics["target"] == target]
        plot = _panel(draw, box, _target_panel_title(target))
        observed = part["observed"].to_numpy(dtype=float)
        predicted = part["predicted"].to_numpy(dtype=float)
        limits = _finite_limits(
            np.concatenate([observed, predicted]),
            padding=0.05,
        )
        x_map, y_map = _draw_xy_axes(
            draw,
            plot,
            limits,
            limits,
            xlabel="Observé (%)",
            ylabel="Prédit (%)",
            x_ticks=4,
            y_ticks=4,
        )
        draw.line(
            (
                x_map(limits[0]),
                y_map(limits[0]),
                x_map(limits[1]),
                y_map(limits[1]),
            ),
            fill=ORANGE,
            width=3,
        )
        step = max(1, len(part) // 1600)
        for observed_value, predicted_value in zip(
            observed[::step],
            predicted[::step],
        ):
            x = x_map(observed_value)
            y = y_map(predicted_value)
            draw.ellipse(
                (x - 3, y - 3, x + 3, y + 3),
                fill=(*_hex_rgb(TEAL), 95),
            )
        r2 = float(r2_score(observed, predicted))
        _draw_top_right_note(
            draw,
            plot,
            f"R² OOF = {r2:.3f}",
            font_size=14,
        )
    _save_png(image, path, dpi)


def plot_residual_distributions(
    diagnostics: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> None:
    model = str(diagnostics["model"].iloc[0])
    image, draw = _canvas(
        "Distribution des résidus — huit sorties",
        f"Résidu = observé − prédit; prédictions OOF agrégées; modèle : {_short_model(model)}",
        size=(2800, 1750),
    )
    boxes = _grid_boxes(2800, 1750, rows=2, columns=4)
    for target, box in zip(TARGET_COLUMNS, boxes):
        values = diagnostics.loc[
            diagnostics["target"] == target,
            "residual",
        ].dropna().to_numpy(dtype=float)
        plot = _panel(draw, box, _target_panel_title(target))
        counts, edges = np.histogram(values, bins=30)
        xlim = _finite_limits(edges, include_zero=True, padding=0.02)
        ylim = (0.0, max(float(np.max(counts)) * 1.12, 1.0))
        x_map, y_map = _draw_xy_axes(
            draw,
            plot,
            xlim,
            ylim,
            xlabel="Résidu (points de %)",
            ylabel="Effectif",
            x_ticks=4,
            y_ticks=4,
        )
        for index, count in enumerate(counts):
            x0 = x_map(edges[index])
            x1 = x_map(edges[index + 1])
            draw.rectangle(
                (x0 + 1, y_map(count), x1 - 1, y_map(0)),
                fill=(*_hex_rgb(TEAL), 185),
            )
        if xlim[0] <= 0 <= xlim[1]:
            zero = x_map(0)
            draw.line((zero, plot[1], zero, plot[3]), fill=ORANGE, width=3)
        _draw_top_right_note(
            draw,
            plot,
            f"Moyenne = {np.mean(values):.3g}",
            font_size=14,
        )
    _save_png(image, path, dpi)


def _binned_residual_trend(
    x_values: np.ndarray,
    residuals: np.ndarray,
    bins: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x_values) & np.isfinite(residuals)
    x_values = x_values[finite]
    residuals = residuals[finite]
    if len(x_values) < 4:
        return np.array([]), np.array([])
    quantiles = np.unique(np.quantile(x_values, np.linspace(0.0, 1.0, bins + 1)))
    if len(quantiles) < 3:
        return np.array([]), np.array([])
    labels = np.digitize(x_values, quantiles[1:-1], right=True)
    xs, ys = [], []
    for label in range(len(quantiles) - 1):
        mask = labels == label
        if np.any(mask):
            xs.append(float(np.mean(x_values[mask])))
            ys.append(float(np.mean(residuals[mask])))
    return np.asarray(xs), np.asarray(ys)


def plot_residuals_vs_input(
    diagnostics: pd.DataFrame,
    input_name: str,
    path: Path,
    *,
    dpi: int,
) -> None:
    model = str(diagnostics["model"].iloc[0])
    image, draw = _canvas(
        f"Résidus selon {DISPLAY_INPUTS[input_name].lower()}",
        f"OOF agrégé; ligne orange = moyenne par quantiles; modèle : {_short_model(model)}",
        size=(2800, 1750),
    )
    boxes = _grid_boxes(2800, 1750, rows=2, columns=4)
    for target, box in zip(TARGET_COLUMNS, boxes):
        part = diagnostics[diagnostics["target"] == target].dropna(
            subset=[input_name, "residual"]
        )
        plot = _panel(draw, box, _target_panel_title(target))
        x_values = part[input_name].to_numpy(dtype=float)
        residuals = part["residual"].to_numpy(dtype=float)
        xlim = _finite_limits(x_values, padding=0.04)
        ylim = _finite_limits(residuals, include_zero=True, padding=0.07)
        x_map, y_map = _draw_xy_axes(
            draw,
            plot,
            xlim,
            ylim,
            xlabel=DISPLAY_INPUTS[input_name],
            ylabel="Résidu (points de %)",
            x_ticks=4,
            y_ticks=4,
        )
        if ylim[0] <= 0 <= ylim[1]:
            zero = y_map(0)
            draw.line((plot[0], zero, plot[2], zero), fill=GRAY, width=2)
        step = max(1, len(part) // 1600)
        for x_value, residual in zip(x_values[::step], residuals[::step]):
            x = x_map(x_value)
            y = y_map(residual)
            draw.ellipse(
                (x - 3, y - 3, x + 3, y + 3),
                fill=(*_hex_rgb(TEAL), 80),
            )
        trend_x, trend_y = _binned_residual_trend(x_values, residuals)
        points = [
            (x_map(x_value), y_map(y_value))
            for x_value, y_value in zip(trend_x, trend_y)
        ]
        if len(points) >= 2:
            draw.line(points, fill=ORANGE, width=4, joint="curve")
        correlation = pd.Series(x_values).corr(pd.Series(residuals))
        _draw_top_right_note(
            draw,
            plot,
            f"r Pearson = {correlation:.3f}",
            font_size=14,
        )
    _save_png(image, path, dpi)


def _predict(
    bundle: object,
    inputs: pd.DataFrame,
    model_name: str | None,
) -> pd.DataFrame:
    try:
        prediction = predict_primary(bundle, inputs, model_name=model_name)
    except Exception as exc:
        available = ", ".join(map(str, getattr(bundle, "models", {}).keys()))
        raise EvidenceInputError(
            "Échec de l'inférence avec l'artefact final. "
            f"Modèle demandé : {model_name or 'modèle principal'}. "
            f"Modèles disponibles : {available}. Détail : {exc}"
        ) from exc
    missing = [target for target in TARGET_COLUMNS if target not in prediction.columns]
    if missing:
        raise EvidenceInputError(
            "L'artefact ne produit pas les huit sorties. Sorties absentes : "
            + ", ".join(missing)
        )
    return prediction


def calculate_oaat_sensitivity(
    bundle: object,
    reference: pd.DataFrame,
    *,
    points: int,
    model_name: str | None,
) -> pd.DataFrame:
    if points < 5:
        raise EvidenceInputError("--sensitivity-points doit être au moins égal à 5.")
    baseline = reference[INPUT_COLUMNS].median(numeric_only=True)
    rows: list[pd.DataFrame] = []
    for input_name in INPUT_COLUMNS:
        values = pd.to_numeric(reference[input_name], errors="coerce").dropna()
        low = float(values.quantile(0.05))
        high = float(values.quantile(0.95))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(values.min())
            high = float(values.max())
        if high <= low:
            raise EvidenceInputError(
                f"Variation impossible pour {input_name} : domaine constant."
            )
        grid = np.linspace(low, high, points)
        inputs = pd.DataFrame(
            np.tile(baseline[INPUT_COLUMNS].to_numpy(dtype=float), (points, 1)),
            columns=INPUT_COLUMNS,
        )
        inputs[input_name] = grid
        prediction = _predict(bundle, inputs, model_name)
        result = prediction[TARGET_COLUMNS].reset_index(drop=True).copy()
        result.insert(0, "normalized_level", np.linspace(0.0, 1.0, points))
        result.insert(0, "input_value", grid)
        result.insert(0, "input", input_name)
        result["Toxicity_Index_pct"] = result[HALOGEN_COLUMNS].sum(axis=1)
        result["Valuable_Yield_pct"] = (
            result["Liquid_Yield_pct"] + result["Gas_Yield_pct"]
        )
        rows.append(result)
    sensitivity = pd.concat(rows, ignore_index=True)
    sensitivity.insert(
        0,
        "model",
        model_name or str(getattr(bundle, "best_model_name", "modèle principal")),
    )
    return sensitivity


def plot_oaat_sensitivity(
    sensitivity: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> None:
    model = str(sensitivity["model"].iloc[0])
    image, draw = _canvas(
        "Analyse de sensibilité un facteur à la fois",
        "Chaque entrée varie de son 5e au 95e percentile; les autres restent à leur médiane",
        size=(2800, 1820),
    )
    legend_y = 148
    legend_font = _font(15, True)
    cursor_x = 70
    for input_name in INPUT_COLUMNS:
        color = INPUT_COLORS[input_name]
        draw.line((cursor_x, legend_y + 10, cursor_x + 34, legend_y + 10), fill=color, width=5)
        label = DISPLAY_INPUTS[input_name]
        draw.text((cursor_x + 42, legend_y), label, font=legend_font, fill=NAVY)
        cursor_x += int(_text_width(draw, label, legend_font)) + 90
    boxes = _grid_boxes(2800, 1820, rows=2, columns=4, top=205)
    for target, box in zip(TARGET_COLUMNS, boxes):
        target_frame = sensitivity[["input", "normalized_level", target]]
        plot = _panel(draw, box, _target_panel_title(target))
        ylim = _finite_limits(target_frame[target].to_numpy(), padding=0.07)
        x_map, y_map = _draw_xy_axes(
            draw,
            plot,
            (0.0, 1.0),
            ylim,
            xlabel="Niveau dans le domaine observé (5e → 95e pct.)",
            ylabel="Prédiction (%)",
            x_ticks=5,
            y_ticks=4,
        )
        for input_name in INPUT_COLUMNS:
            part = target_frame[target_frame["input"] == input_name].sort_values(
                "normalized_level"
            )
            points = [
                (x_map(x_value), y_map(y_value))
                for x_value, y_value in zip(
                    part["normalized_level"],
                    part[target],
                )
            ]
            if len(points) >= 2:
                draw.line(
                    points,
                    fill=INPUT_COLORS[input_name],
                    width=4,
                    joint="curve",
                )
    draw.text(
        (60, 1780),
        f"Modèle d'inférence : {_short_model(model)}. Analyse locale exploratoire; "
        "elle ne démontre pas un effet causal.",
        font=_font(16),
        fill=GRAY,
    )
    _save_png(image, path, dpi)


def latin_hypercube_scenarios(
    reference: pd.DataFrame,
    *,
    count: int,
    seed: int,
) -> pd.DataFrame:
    if count < 100:
        raise EvidenceInputError("--scenario-count doit être au moins égal à 100.")
    generator = np.random.default_rng(seed)
    result = pd.DataFrame(index=np.arange(count))
    for input_name in INPUT_COLUMNS:
        values = pd.to_numeric(reference[input_name], errors="coerce").dropna().to_numpy()
        if values.size < 2:
            raise EvidenceInputError(
                f"Données insuffisantes pour échantillonner {input_name}."
            )
        strata = (np.arange(count) + generator.random(count)) / count
        generator.shuffle(strata)
        # Limitation au domaine central observé pour réduire les extrapolations.
        probabilities = 0.05 + 0.90 * strata
        result[input_name] = np.quantile(values, probabilities)
    result.insert(0, "scenario_id", np.arange(1, count + 1))
    return result


def pareto_mask(objectives: np.ndarray, block_size: int = 256) -> np.ndarray:
    """Retourne le masque des points non dominés pour des objectifs à minimiser."""
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2:
        raise ValueError("La matrice d'objectifs Pareto doit être bidimensionnelle.")
    if not np.isfinite(values).all():
        raise ValueError("Les objectifs Pareto doivent être finis.")
    count = len(values)
    efficient = np.ones(count, dtype=bool)
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        candidates = values[start:stop]
        less_or_equal = np.all(
            values[:, None, :] <= candidates[None, :, :],
            axis=2,
        )
        strictly_less = np.any(
            values[:, None, :] < candidates[None, :, :],
            axis=2,
        )
        efficient[start:stop] = ~np.any(
            less_or_equal & strictly_less,
            axis=0,
        )
    return efficient


def calculate_scenarios_and_pareto(
    bundle: object,
    reference: pd.DataFrame,
    *,
    count: int,
    seed: int,
    model_name: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = latin_hypercube_scenarios(reference, count=count, seed=seed)
    prediction = _predict(bundle, scenarios[INPUT_COLUMNS], model_name)
    result = pd.concat(
        [
            scenarios.reset_index(drop=True),
            prediction[TARGET_COLUMNS].reset_index(drop=True),
        ],
        axis=1,
    )
    result["Toxicity_Index_pct"] = result[HALOGEN_COLUMNS].sum(axis=1)
    result["Valuable_Yield_pct"] = (
        result["Liquid_Yield_pct"] + result["Gas_Yield_pct"]
    )
    result["model"] = model_name or str(
        getattr(bundle, "best_model_name", "modèle principal")
    )
    objectives = np.column_stack(
        [
            -result["Valuable_Yield_pct"].to_numpy(dtype=float),
            result["Toxicity_Index_pct"].to_numpy(dtype=float),
            result["Concentration_liquide_chimique_kg"].to_numpy(dtype=float),
        ]
    )
    mask = pareto_mask(objectives)
    result["pareto_nondominated"] = mask
    pareto = result[mask].copy().sort_values(
        ["Toxicity_Index_pct", "Valuable_Yield_pct"],
        ascending=[True, False],
    )
    return result, pareto


def scenario_summary(scenarios: pd.DataFrame) -> pd.DataFrame:
    variables = [
        *INPUT_COLUMNS,
        *TARGET_COLUMNS,
        "Toxicity_Index_pct",
        "Valuable_Yield_pct",
    ]
    rows: list[dict[str, object]] = []
    for variable in variables:
        values = pd.to_numeric(scenarios[variable], errors="coerce").dropna()
        rows.append(
            {
                "variable": variable,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "minimum": float(values.min()),
                "q05": float(values.quantile(0.05)),
                "median": float(values.median()),
                "q95": float(values.quantile(0.95)),
                "maximum": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _gradient_color(value: float, low: float, high: float) -> tuple[int, int, int, int]:
    ratio = 0.5 if high <= low else float(np.clip((value - low) / (high - low), 0, 1))
    left = np.asarray(_hex_rgb(BLUE), dtype=float)
    right = np.asarray(_hex_rgb(ORANGE), dtype=float)
    rgb = tuple(np.round(left * (1 - ratio) + right * ratio).astype(int))
    return (*rgb, 115)


def plot_scenarios_pareto(
    scenarios: pd.DataFrame,
    pareto: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> None:
    image, draw = _canvas(
        "Échantillonnage de scénarios et front de Pareto exploratoire",
        "Non-dominance sur 3 objectifs : maximiser liquide + gaz, minimiser toxicité "
        "et concentration chimique",
        size=(2900, 1150),
    )
    specs = [
        (
            "Rendement valorisable vs toxicité",
            "Toxicity_Index_pct",
            "Valuable_Yield_pct",
            "Concentration_liquide_chimique_kg",
            "Toxicité prédite (%)",
            "Liquide + gaz prédits (%)",
        ),
        (
            "Rendement valorisable vs concentration",
            "Concentration_liquide_chimique_kg",
            "Valuable_Yield_pct",
            "Toxicity_Index_pct",
            "Concentration liquide (kg)",
            "Liquide + gaz prédits (%)",
        ),
        (
            "Toxicité vs concentration",
            "Concentration_liquide_chimique_kg",
            "Toxicity_Index_pct",
            "Valuable_Yield_pct",
            "Concentration liquide (kg)",
            "Toxicité prédite (%)",
        ),
    ]
    color_labels = {
        "Concentration_liquide_chimique_kg": "concentration (kg)",
        "Toxicity_Index_pct": "toxicité prédite (%)",
        "Valuable_Yield_pct": "liquide + gaz prédits (%)",
    }
    boxes = _grid_boxes(2900, 1150, rows=1, columns=3, top=175, bottom=55)
    for box, spec in zip(boxes, specs):
        title, x_column, y_column, color_column, xlabel, ylabel = spec
        plot = _panel(draw, box, title)
        legend_y = box[1] + 68
        legend_font = _font(12, True)
        draw.ellipse(
            (box[0] + 22, legend_y, box[0] + 34, legend_y + 12),
            fill=(*_hex_rgb(BLUE), 150),
        )
        legend_text = (
            f"Couleur : {color_labels[color_column]} "
            "(bleu = faible; orange = élevé)"
        )
        draw.text(
            (box[0] + 42, legend_y - 2),
            legend_text,
            font=legend_font,
            fill=GRAY,
        )
        pareto_legend_x = box[2] - 180
        draw.ellipse(
            (
                pareto_legend_x,
                legend_y - 1,
                pareto_legend_x + 14,
                legend_y + 13,
            ),
            outline=NAVY,
            fill=(*_hex_rgb(RED), 210),
            width=2,
        )
        draw.text(
            (pareto_legend_x + 22, legend_y - 2),
            "Front de Pareto",
            font=legend_font,
            fill=GRAY,
        )
        plot = (plot[0], max(plot[1], box[1] + 116), plot[2], plot[3])
        xlim = _finite_limits(scenarios[x_column].to_numpy(), padding=0.04)
        ylim = _finite_limits(scenarios[y_column].to_numpy(), padding=0.06)
        x_map, y_map = _draw_xy_axes(
            draw,
            plot,
            xlim,
            ylim,
            xlabel=xlabel,
            ylabel=ylabel,
            x_ticks=4,
            y_ticks=4,
        )
        color_low = float(scenarios[color_column].min())
        color_high = float(scenarios[color_column].max())
        step = max(1, len(scenarios) // 4000)
        for _, row in scenarios.iloc[::step].iterrows():
            x = x_map(row[x_column])
            y = y_map(row[y_column])
            draw.ellipse(
                (x - 3, y - 3, x + 3, y + 3),
                fill=_gradient_color(row[color_column], color_low, color_high),
            )
        for _, row in pareto.iterrows():
            x = x_map(row[x_column])
            y = y_map(row[y_column])
            draw.ellipse(
                (x - 6, y - 6, x + 6, y + 6),
                outline=NAVY,
                fill=(*_hex_rgb(RED), 210),
                width=2,
            )
        _draw_top_right_note(
            draw,
            plot,
            f"{len(pareto)} scénarios non dominés / {len(scenarios)}",
            font_size=13,
        )
    _save_png(image, path, dpi)


def _canonical_learning(
    frame: pd.DataFrame,
    source: Path,
) -> pd.DataFrame:
    result = _rename_aliases(
        frame,
        {
            "model": ("modele", "modèle", "family"),
            "repeat": ("repetition", "répétition"),
            "outer_fold": ("fold", "pli"),
            "training_fraction": ("train_fraction", "fraction"),
            "training_size": ("train_size", "n_train"),
            "mae_normalized": ("normalized_mae", "mae_norm"),
            "rmse_normalized": ("normalized_rmse", "rmse_norm"),
            "r2_macro": ("macro_r2",),
        },
    ).copy()
    _require_columns(
        result,
        [
            "model",
            "repeat",
            "outer_fold",
            "training_fraction",
            "training_size",
            "mae_normalized",
            "rmse_normalized",
            "r2_macro",
        ],
        source=source,
    )
    numeric_columns = [
        "repeat",
        "outer_fold",
        "training_fraction",
        "training_size",
        "mae_normalized",
        "rmse_normalized",
        "r2_macro",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["model"] = result["model"].astype(str).str.strip()
    return result.dropna(subset=["model", *numeric_columns])


def summarize_learning_curve(
    frame: pd.DataFrame,
    *,
    source: Path,
    preferred_model: str,
) -> tuple[pd.DataFrame, str]:
    canonical = _canonical_learning(frame, source)
    available = list(dict.fromkeys(canonical["model"].tolist()))
    automatic = [
        model
        for model in available
        if _slug(model)
        in {
            _slug("Automatic selection"),
            _slug(PRODUCTION_MODEL_NAME),
            _slug("Sélection automatique"),
        }
    ]
    if automatic:
        selected_model = automatic[0]
    elif preferred_model in available:
        selected_model = preferred_model
    else:
        selected_model = available[0]
    selected = canonical[canonical["model"] == selected_model]
    per_repeat = (
        selected.groupby(
            ["model", "training_fraction", "training_size", "repeat"],
            as_index=False,
        )[["mae_normalized", "rmse_normalized", "r2_macro"]]
        .mean(numeric_only=True)
    )
    rows: list[dict[str, object]] = []
    for (model, fraction, size), group in per_repeat.groupby(
        ["model", "training_fraction", "training_size"],
        sort=True,
    ):
        for metric in ("mae_normalized", "rmse_normalized", "r2_macro"):
            rows.append(
                {
                    "model": model,
                    "training_fraction": fraction,
                    "training_size": size,
                    "metric": metric,
                    **_ci95(group[metric]),
                }
            )
    return pd.DataFrame(rows), selected_model


def plot_learning_curve(
    summary: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> None:
    model = str(summary["model"].iloc[0])
    image, draw = _canvas(
        "Courbe d'apprentissage en validation",
        f"Moyenne ± écart-type entre répétitions; modèle : {_short_model(model)}",
        size=(2700, 1050),
    )
    labels = {
        "mae_normalized": "MAE normalisée",
        "rmse_normalized": "RMSE normalisée",
        "r2_macro": "R² macro",
    }
    boxes = _grid_boxes(2700, 1050, rows=1, columns=3, top=175, bottom=55)
    for box, metric in zip(boxes, labels):
        part = summary[summary["metric"] == metric].sort_values("training_size")
        plot = _panel(draw, box, labels[metric])
        x_values = part["training_size"].to_numpy(dtype=float)
        means = part["mean"].to_numpy(dtype=float)
        std = part["std"].fillna(0).to_numpy(dtype=float)
        xlim = _finite_limits(x_values, padding=0.05)
        ylim = _finite_limits(
            np.concatenate([means - std, means + std]),
            include_zero=False,
            padding=0.08,
        )
        x_map, y_map = _draw_xy_axes(
            draw,
            plot,
            xlim,
            ylim,
            xlabel="Taille d'entraînement",
            ylabel=labels[metric],
            x_ticks=5,
            y_ticks=5,
        )
        points = [(x_map(x), y_map(y)) for x, y in zip(x_values, means)]
        if len(points) >= 2:
            draw.line(points, fill=TEAL, width=5, joint="curve")
        for x, mean, deviation in zip(x_values, means, std):
            center_x = x_map(x)
            low_y = y_map(mean - deviation)
            high_y = y_map(mean + deviation)
            draw.line((center_x, low_y, center_x, high_y), fill=NAVY, width=3)
            draw.line((center_x - 8, low_y, center_x + 8, low_y), fill=NAVY, width=3)
            draw.line((center_x - 8, high_y, center_x + 8, high_y), fill=NAVY, width=3)
            center_y = y_map(mean)
            draw.ellipse(
                (center_x - 6, center_y - 6, center_x + 6, center_y + 6),
                fill=TEAL,
                outline=NAVY,
                width=2,
            )
    _save_png(image, path, dpi)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def build_manifest(
    rows: list[dict[str, object]],
    *,
    nested_dir: Path,
    artifact_path: Path | None,
    diagnostic_model: str,
    prediction_model: str,
) -> pd.DataFrame:
    common = {
        "nested_results_directory": str(nested_dir),
        "artifact": str(artifact_path) if artifact_path else "",
        "diagnostic_model": diagnostic_model,
        "inference_model": prediction_model,
        "scope": "prototype exploratoire; aucune validation industrielle",
    }
    return pd.DataFrame([{**common, **row} for row in rows])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Génère des CSV et PNG scientifiques à partir d'une validation "
            "imbriquée répétée déjà calculée. Aucun modèle n'est réentraîné."
        )
    )
    parser.add_argument(
        "--nested-dir",
        type=Path,
        default=DEFAULT_NESTED_DIR,
        help=f"Dossier des résultats imbriqués (défaut : {DEFAULT_NESTED_DIR}).",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help=(
            "Artefact ModelBundle final. Par défaut, utilise d'abord "
            "artifacts/production_selected_bundle.joblib puis model_bundle.joblib."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Dossier de sortie (défaut : {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--diagnostic-model",
        default=None,
        help=(
            "Modèle des graphiques OOF. Défaut : sélection automatique si "
            "disponible, sinon meilleur R² macro."
        ),
    )
    parser.add_argument(
        "--prediction-model",
        default=None,
        help=(
            "Nom exact du modèle de l'artefact pour sensibilité/scénarios. "
            "Défaut : modèle principal de l'artefact."
        ),
    )
    parser.add_argument(
        "--scenario-count",
        type=int,
        default=2500,
        help="Nombre de scénarios Latin hypercube (défaut : 2500).",
    )
    parser.add_argument(
        "--sensitivity-points",
        type=int,
        default=31,
        help="Points par facteur pour l'analyse OAAT (défaut : 31).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help=f"Graine de l'échantillonnage (défaut : {RANDOM_STATE}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Métadonnée de résolution des PNG (défaut : 300 dpi).",
    )
    parser.add_argument(
        "--skip-learning-curve",
        action="store_true",
        help="N'essaie pas de produire la courbe d'apprentissage.",
    )
    parser.add_argument(
        "--skip-model-evidence",
        action="store_true",
        help=(
            "Omet sensibilité, scénarios et Pareto, même si l'artefact existe. "
            "Utile pour une génération strictement fondée sur les CSV."
        ),
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    nested_dir = args.nested_dir.resolve()
    output_dir = args.output_dir.resolve()
    nested = load_nested_results(nested_dir)
    global_metrics = nested["global"]
    output_metrics = nested["output"]
    predictions = nested["predictions"]

    summary_global = summarize_global(global_metrics)
    summary_output = summarize_outputs(output_metrics)
    diagnostic_model = select_diagnostic_model(
        predictions,
        summary_global,
        args.diagnostic_model,
    )

    bundle: object | None = None
    artifact_path: Path | None = None
    if not args.skip_model_evidence:
        requested_artifact = args.artifact.resolve() if args.artifact else None
        bundle, artifact_path = load_final_bundle(requested_artifact)
    reference = reference_data(bundle)
    diagnostics = aggregate_diagnostic_predictions(
        predictions,
        model=diagnostic_model,
        reference=reference,
    )

    # Toutes les entrées sont validées avant de créer le dossier de sortie.
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    global_csv = output_dir / "model_comparison_95ci.csv"
    output_csv = output_dir / "metrics_by_output_95ci.csv"
    diagnostic_csv = output_dir / "diagnostic_oof_predictions.csv"
    residual_csv = output_dir / "residual_summary.csv"
    association_csv = output_dir / "residual_input_associations.csv"
    _write_csv(summary_global, global_csv)
    _write_csv(summary_output, output_csv)
    _write_csv(diagnostics, diagnostic_csv)
    _write_csv(residual_summary(diagnostics), residual_csv)
    _write_csv(residual_input_associations(diagnostics), association_csv)
    for path, description in (
        (global_csv, "Métriques globales avec IC95 % entre répétitions"),
        (output_csv, "R², RMSE et MAE par sortie avec IC95 %"),
        (diagnostic_csv, "Prédictions OOF agrégées et résidus"),
        (residual_csv, "Résumé des distributions de résidus"),
        (association_csv, "Associations résidus-température/concentration"),
    ):
        manifest_rows.append(
            {
                "file": path.name,
                "type": "csv",
                "status": "generated",
                "description": description,
            }
        )

    plot_jobs = [
        (
            plot_model_comparison,
            (summary_global,),
            output_dir / "01_model_comparison_r2_ic95.png",
            "Comparaison des modèles, R² macro et IC95 %",
        ),
        (
            plot_metric_by_output,
            (summary_output, "r2"),
            output_dir / "02_r2_by_output_ic95.png",
            "R² par sortie et par modèle avec IC95 %",
        ),
        (
            plot_metric_by_output,
            (summary_output, "rmse"),
            output_dir / "03_rmse_by_output_ic95.png",
            "RMSE par sortie et par modèle avec IC95 %",
        ),
        (
            plot_metric_by_output,
            (summary_output, "mae"),
            output_dir / "04_mae_by_output_ic95.png",
            "MAE par sortie et par modèle avec IC95 %",
        ),
        (
            plot_observed_predicted,
            (diagnostics,),
            output_dir / "05_observed_vs_predicted_8_outputs.png",
            "Observé versus prédit pour les huit sorties",
        ),
        (
            plot_residual_distributions,
            (diagnostics,),
            output_dir / "06_residual_distributions_8_outputs.png",
            "Distribution des résidus pour les huit sorties",
        ),
        (
            plot_residuals_vs_input,
            (diagnostics, "Temperature_C"),
            output_dir / "07_residuals_vs_temperature.png",
            "Résidus selon la température",
        ),
        (
            plot_residuals_vs_input,
            (diagnostics, "Concentration_liquide_chimique_kg"),
            output_dir / "08_residuals_vs_concentration.png",
            "Résidus selon la concentration",
        ),
    ]
    for function, positional, path, description in plot_jobs:
        function(*positional, path, dpi=args.dpi)
        manifest_rows.append(
            {
                "file": path.name,
                "type": "png",
                "status": "generated",
                "description": description,
            }
        )

    inference_model = args.prediction_model or (
        str(getattr(bundle, "best_model_name", "")) if bundle is not None else ""
    )
    if bundle is not None:
        if args.prediction_model:
            available_models = list(map(str, getattr(bundle, "models", {}).keys()))
            inference_model = _match_label(
                args.prediction_model,
                available_models,
                "Modèle d'inférence",
            )
        sensitivity = calculate_oaat_sensitivity(
            bundle,
            reference,
            points=args.sensitivity_points,
            model_name=inference_model or None,
        )
        sensitivity_csv = output_dir / "oaat_sensitivity.csv"
        _write_csv(sensitivity, sensitivity_csv)
        sensitivity_png = output_dir / "09_oaat_sensitivity_8_outputs.png"
        plot_oaat_sensitivity(sensitivity, sensitivity_png, dpi=args.dpi)

        scenarios, pareto = calculate_scenarios_and_pareto(
            bundle,
            reference,
            count=args.scenario_count,
            seed=args.seed,
            model_name=inference_model or None,
        )
        scenarios_csv = output_dir / "scenario_sampling.csv"
        pareto_csv = output_dir / "pareto_front.csv"
        scenario_summary_csv = output_dir / "scenario_summary.csv"
        _write_csv(scenarios, scenarios_csv)
        _write_csv(pareto, pareto_csv)
        _write_csv(scenario_summary(scenarios), scenario_summary_csv)
        pareto_png = output_dir / "10_scenarios_and_pareto_front.png"
        plot_scenarios_pareto(scenarios, pareto, pareto_png, dpi=args.dpi)
        for path, file_type, description in (
            (
                sensitivity_csv,
                "csv",
                "Sensibilité un facteur à la fois, domaine 5e-95e percentile",
            ),
            (
                sensitivity_png,
                "png",
                "Courbes de sensibilité OAAT des huit sorties",
            ),
            (
                scenarios_csv,
                "csv",
                "Scénarios Latin hypercube et prédictions",
            ),
            (
                pareto_csv,
                "csv",
                "Scénarios non dominés sur trois objectifs",
            ),
            (
                scenario_summary_csv,
                "csv",
                "Statistiques descriptives de l'échantillonnage",
            ),
            (
                pareto_png,
                "png",
                "Projections du front de Pareto exploratoire",
            ),
        ):
            manifest_rows.append(
                {
                    "file": path.name,
                    "type": file_type,
                    "status": "generated",
                    "description": description,
                }
            )
    else:
        manifest_rows.append(
            {
                "file": "",
                "type": "model_evidence",
                "status": "omitted",
                "description": (
                    "Sensibilité, scénarios et Pareto omis : artefact final absent "
                    "ou --skip-model-evidence."
                ),
            }
        )

    if args.skip_learning_curve:
        manifest_rows.append(
            {
                "file": "",
                "type": "learning_curve",
                "status": "omitted",
                "description": "Courbe d'apprentissage omise via --skip-learning-curve.",
            }
        )
    elif nested["learning"] is None:
        manifest_rows.append(
            {
                "file": "",
                "type": "learning_curve",
                "status": "not_available",
                "description": (
                    "learning_curve_metrics.csv absent. Relancer la validation "
                    "avec son option de courbe d'apprentissage, puis régénérer."
                ),
            }
        )
        warnings.warn(
            "Courbe d'apprentissage non générée : learning_curve_metrics.csv absent. "
            "Aucune valeur de substitution n'a été inventée.",
            stacklevel=2,
        )
    else:
        learning_summary, _ = summarize_learning_curve(
            nested["learning"],
            source=nested["learning_path"],
            preferred_model=diagnostic_model,
        )
        learning_csv = output_dir / "learning_curve_summary.csv"
        learning_png = output_dir / "11_learning_curve.png"
        _write_csv(learning_summary, learning_csv)
        plot_learning_curve(learning_summary, learning_png, dpi=args.dpi)
        for path, file_type, description in (
            (
                learning_csv,
                "csv",
                "Courbe d'apprentissage agrégée entre répétitions",
            ),
            (
                learning_png,
                "png",
                "Courbe d'apprentissage moyenne ± écart-type",
            ),
        ):
            manifest_rows.append(
                {
                    "file": path.name,
                    "type": file_type,
                    "status": "generated",
                    "description": description,
                }
            )

    manifest = build_manifest(
        manifest_rows,
        nested_dir=nested_dir,
        artifact_path=artifact_path,
        diagnostic_model=diagnostic_model,
        prediction_model=inference_model,
    )
    _write_csv(manifest, output_dir / "manifest.csv")
    print(
        f"Preuves scientifiques générées dans {output_dir} "
        f"({len(manifest[manifest['status'] == 'generated'])} fichiers)."
    )
    print(
        "Portée : prototype exploratoire; les résultats ne constituent pas "
        "une validation industrielle ni une preuve causale."
    )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except EvidenceInputError as exc:
        print(f"ERREUR — preuves scientifiques non générées\n{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
