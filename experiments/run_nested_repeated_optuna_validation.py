"""Validation Optuna imbriquée et répétée des quatre familles actives.

Le protocole traite les huit cibles comme une régression supervisée
multi-sorties. À chaque pli externe, les hyperparamètres de Ridge polynomial,
Extra Trees, MLP et de l'hybride à deux branches MLP + XGBoost sont optimisés
séparément pour chacune des huit sorties sur les seuls exemples
d'entraînement. La famille retenue pour une sortie est celle qui minimise son
RMSE interne normalisé; le pli externe n'intervient jamais dans ce choix.

Le script évalue sur exactement les mêmes plis :

* un prédicteur naïf ajusté à la moyenne du pli d'entraînement ;
* Ridge polynomial Optuna ;
* Extra Trees Optuna ;
* MLP Optuna compact ;
* hybride deux branches MLP compact + XGBoost Optuna ;
* la sélection automatique d'une famille par sortie.

Les prédictions exportées sont brutes : aucune troncature à [0, 100] ni
correction du bilan massique n'est appliquée pendant l'évaluation. Cela évite
qu'un post-traitement ne masque les erreurs propres aux modèles.

Exécution scientifique complète (coûteuse) :

    python experiments/run_nested_repeated_optuna_validation.py

Par défaut, cette exécution utilise 5 plis externes répétés 5 fois, 3 plis
internes, 20 essais Optuna par famille/sortie/pli externe et 2 tâches
parallèles. Chaque pli externe terminé est conservé sous forme d'un checkpoint
atomique vérifié par les signatures du protocole, de l'implémentation et du
jeu de données afin de permettre une reprise automatique sûre.

Test fonctionnel court :

    python experiments/run_nested_repeated_optuna_validation.py --quick

Le mode rapide force 2 plis externes, 1 répétition, 2 plis internes, 1 essai
par famille/sortie et 1 tâche parallèle. Il vérifie le pipeline mais ne produit
pas d'intervalle de confiance interprétable.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import sys
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import optuna
import pandas as pd
import scipy
import sklearn
import xgboost
from scipy.stats import t as student_t
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    INPUT_COLUMNS,
    OPTUNA_HYBRID_MODEL_NAME,
    OPTUNA_MLP_MODEL_NAME,
    OPTUNA_MODEL_FAMILIES,
    PRODUCTION_MODEL_NAME,
    RANDOM_STATE,
    TARGET_COLUMNS,
)
from src.data import find_data_file, load_dataset  # noqa: E402
from src.features import MODEL_INPUT_COLUMNS  # noqa: E402
from src.optuna_training import (  # noqa: E402
    AutomaticTargetSelector,
    IndependentTargetRegressor,
    train_optuna_families,
)


DEFAULT_REPORT_DIR = ROOT / "reports" / "nested_repeated_validation"
NAIVE_MODEL_NAME = "Prédicteur naïf (moyenne)"
SELECTED_MODEL_NAME = PRODUCTION_MODEL_NAME
PRIMARY_METRICS_GLOBAL = [
    "mae_normalized",
    "rmse_normalized",
    "r2_macro",
    "mae_improvement_vs_naive_pct",
    "rmse_improvement_vs_naive_pct",
    "r2_delta_vs_naive",
]
PRIMARY_METRICS_OUTPUT = [
    "mae",
    "rmse",
    "r2",
    "mae_over_train_std",
    "rmse_over_train_std",
    "rmse_over_observed_mean_pct",
    "nrmse_observed_std",
    "nrmse_observed_range_pct",
    "mae_improvement_vs_naive_pct",
    "rmse_improvement_vs_naive_pct",
    "r2_delta_vs_naive",
]
LEARNING_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Protocol:
    outer_splits: int
    repeats: int
    trials: int
    inner_splits: int
    parallel_jobs: int
    quick: bool
    learning_curve: bool
    seed: int

    @property
    def total_outer_folds(self) -> int:
        return self.outer_splits * self.repeats


def _json_default(value: object) -> object:
    """Convertit les scalaires NumPy et les chemins pour les sidecars JSON."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime,)):
        return value.isoformat()
    raise TypeError(f"Objet non sérialisable en JSON : {type(value).__name__}")


def _json_dumps(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=_json_default,
    )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _indices_sha256(indices: np.ndarray) -> str:
    return _sha256_bytes(
        np.asarray(indices, dtype=np.int64).tobytes()
    )


def _protocol_signature(protocol: Protocol) -> str:
    """Signe le contrat complet qui conditionne la réutilisation d'un pli."""

    contract = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "protocol": {
            "outer_splits": protocol.outer_splits,
            "repeats": protocol.repeats,
            "trials": protocol.trials,
            "inner_splits": protocol.inner_splits,
            "parallel_jobs": protocol.parallel_jobs,
            "quick": protocol.quick,
            "learning_curve": protocol.learning_curve,
            "seed": protocol.seed,
        },
        "inputs": list(INPUT_COLUMNS),
        "engineered_inputs": list(MODEL_INPUT_COLUMNS),
        "targets": list(TARGET_COLUMNS),
        "families": list(OPTUNA_MODEL_FAMILIES),
        "selected_model": SELECTED_MODEL_NAME,
        "naive_model": NAIVE_MODEL_NAME,
        "raw_predictions_without_postprocessing": True,
    }
    return _sha256_bytes(_json_dumps(contract).encode("utf-8"))


def _implementation_signature() -> str:
    """Empêche de reprendre silencieusement des résultats d'un autre code."""

    sources = [Path(__file__).resolve()]
    training_source = inspect.getsourcefile(train_optuna_families)
    if training_source is not None:
        sources.append(Path(training_source).resolve())
    digest = hashlib.sha256()
    for source in sources:
        digest.update(str(source).encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _checkpoint_path(
    checkpoint_dir: Path,
    *,
    split_number: int,
    repeat: int,
    outer_fold: int,
) -> Path:
    return checkpoint_dir / (
        f"outer_{split_number:04d}_repeat_{repeat:03d}"
        f"_fold_{outer_fold:03d}.json"
    )


def _write_checkpoint_atomic(
    path: Path,
    *,
    metadata: dict[str, object],
    payload: dict[str, object],
) -> None:
    """Écrit un checkpoint complet, puis le publie par remplacement atomique."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload_sha256 = _sha256_bytes(
        _json_dumps(payload).encode("utf-8")
    )
    document = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "metadata": metadata,
        "payload_sha256": payload_sha256,
        "payload": payload,
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                document,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_checkpoint(
    path: Path,
    *,
    dataset_sha256: str,
    protocol_signature: str,
    implementation_signature: str,
    split_number: int,
    repeat: int,
    outer_fold: int,
    train_index_sha256: str,
    validation_index_sha256: str,
) -> tuple[dict[str, object] | None, str | None]:
    """Charge uniquement un checkpoint intègre et exactement compatible."""

    if not path.exists():
        return None, None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        metadata = document["metadata"]
        payload = document["payload"]
        if not isinstance(metadata, dict) or not isinstance(payload, dict):
            raise ValueError("structure metadata/payload invalide")
        expected_metadata = {
            "dataset_sha256": dataset_sha256,
            "protocol_signature": protocol_signature,
            "implementation_signature": implementation_signature,
            "split_number": split_number,
            "repeat": repeat,
            "outer_fold": outer_fold,
            "train_index_sha256": train_index_sha256,
            "validation_index_sha256": validation_index_sha256,
        }
        if document.get("checkpoint_format_version") != (
            CHECKPOINT_FORMAT_VERSION
        ):
            raise ValueError("version de checkpoint incompatible")
        mismatches = [
            key
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        ]
        if mismatches:
            raise ValueError(
                "signature incompatible: " + ", ".join(mismatches)
            )
        observed_payload_sha256 = _sha256_bytes(
            _json_dumps(payload).encode("utf-8")
        )
        if document.get("payload_sha256") != observed_payload_sha256:
            raise ValueError("empreinte du contenu invalide")
        required_lists = {
            "outer_global_rows",
            "outer_output_rows",
            "oof_rows",
            "selection_rows",
            "trial_rows",
            "split_manifest_rows",
            "assignment_rows",
            "inner_manifest_rows",
            "mlp_complexity_rows",
            "learning_rows",
        }
        invalid_payload = [
            key
            for key in required_lists
            if not isinstance(payload.get(key), list)
        ]
        if invalid_payload:
            raise ValueError(
                "sections absentes ou invalides: "
                + ", ".join(sorted(invalid_payload))
            )
        return payload, None
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, ValueError) as error:
        return None, str(error)


def _safe_float(value: object) -> float:
    result = float(value)
    return result if np.isfinite(result) else np.nan


def _safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return np.nan
    return float(numerator / denominator)


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(truth, prediction)))


def _r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or float(np.var(truth)) <= 1e-15:
        return np.nan
    return float(r2_score(truth, prediction))


def _validate_protocol(protocol: Protocol, n_rows: int) -> None:
    if len(TARGET_COLUMNS) != 8:
        raise ValueError(
            "Le protocole scientifique exige exactement 8 sorties; "
            f"{len(TARGET_COLUMNS)} ont été configurées."
        )
    if protocol.outer_splits < 2:
        raise ValueError("--outer-splits doit être supérieur ou égal à 2.")
    if protocol.repeats < 1:
        raise ValueError("--repeats doit être supérieur ou égal à 1.")
    if protocol.inner_splits < 2:
        raise ValueError("--inner-splits doit être supérieur ou égal à 2.")
    if protocol.trials < 1:
        raise ValueError("--trials doit être supérieur ou égal à 1.")
    if protocol.parallel_jobs < 1:
        raise ValueError(
            "--parallel-jobs doit être supérieur ou égal à 1."
        )
    if n_rows < protocol.outer_splits:
        raise ValueError(
            "Le nombre de plis externes dépasse le nombre d'observations."
        )
    smallest_outer_train = n_rows - int(np.ceil(n_rows / protocol.outer_splits))
    if smallest_outer_train < protocol.inner_splits:
        raise ValueError(
            "Pas assez d'observations dans un pli externe d'entraînement "
            "pour la validation interne demandée."
        )


def temperature_deciles(
    temperature: pd.Series,
    *,
    required_count_per_bin: int,
    preferred_bins: int = 10,
) -> tuple[pd.Series, int]:
    """Crée le maximum de quantiles stables, jusqu'aux déciles demandés.

    Les températures manquantes sont remplacées uniquement pour construire les
    partitions. L'imputation utilisée par les modèles reste, elle, ajustée dans
    chaque pli au sein des pipelines.
    """

    numeric = pd.to_numeric(temperature, errors="coerce")
    if numeric.notna().sum() == 0:
        raise ValueError(
            "La température est entièrement manquante; la stratification "
            "thermique est impossible."
        )
    filled = numeric.fillna(float(numeric.median()))
    maximum = min(int(preferred_bins), int(filled.nunique()))
    for requested_bins in range(maximum, 1, -1):
        labels = pd.qcut(
            filled,
            q=requested_bins,
            labels=False,
            duplicates="drop",
        )
        counts = labels.value_counts(dropna=False)
        if (
            labels.nunique(dropna=True) >= 2
            and not counts.empty
            and int(counts.min()) >= required_count_per_bin
        ):
            return labels.astype(int), int(labels.nunique())
    raise ValueError(
        "Impossible de construire au moins deux strates thermiques contenant "
        f"chacune {required_count_per_bin} observations."
    )


def training_scales(y_train: pd.DataFrame) -> dict[str, float]:
    scales: dict[str, float] = {}
    for target in TARGET_COLUMNS:
        scale = float(y_train[target].std(ddof=0))
        scales[target] = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
    return scales


def output_metric_row(
    *,
    model: str,
    repeat: int,
    outer_fold: int,
    target: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    train_truth: pd.Series,
    optimization_and_fit_seconds: float,
) -> dict[str, object]:
    mae = float(mean_absolute_error(truth, prediction))
    rmse = _rmse(truth, prediction)
    observed_mean = float(np.mean(truth))
    observed_std = float(np.std(truth, ddof=0))
    observed_range = float(np.max(truth) - np.min(truth))
    train_mean = float(train_truth.mean())
    train_std = float(train_truth.std(ddof=0))
    train_range = float(train_truth.max() - train_truth.min())
    safe_train_std = train_std if train_std > 1e-12 else 1.0
    return {
        "model": model,
        "repeat": repeat,
        "outer_fold": outer_fold,
        "target": target,
        "n_observations": int(len(truth)),
        "mae": mae,
        "rmse": rmse,
        "r2": _r2(truth, prediction),
        "observed_mean": observed_mean,
        "observed_std": observed_std,
        "observed_range": observed_range,
        "train_mean": train_mean,
        "train_std": train_std,
        "train_range": train_range,
        "mae_over_train_std": mae / safe_train_std,
        "rmse_over_train_std": rmse / safe_train_std,
        "rmse_over_observed_mean_pct": (
            100.0 * _safe_divide(rmse, abs(observed_mean))
        ),
        "nrmse_observed_std": _safe_divide(rmse, observed_std),
        "nrmse_observed_range_pct": (
            100.0 * _safe_divide(rmse, observed_range)
        ),
        "absolute_error_sum": float(np.abs(truth - prediction).sum()),
        "squared_error_sum": float(np.square(truth - prediction).sum()),
        "optimization_and_fit_seconds": optimization_and_fit_seconds,
    }


def global_metric_row(
    *,
    model: str,
    repeat: int,
    outer_fold: int,
    truth: pd.DataFrame,
    prediction: np.ndarray,
    scales: dict[str, float],
    optimization_and_fit_seconds: float,
) -> dict[str, object]:
    truth_array = truth[TARGET_COLUMNS].to_numpy(dtype=float)
    scale_array = np.asarray([scales[target] for target in TARGET_COLUMNS])
    normalized_residual = (truth_array - prediction) / scale_array
    r2_values = [
        _r2(truth_array[:, index], prediction[:, index])
        for index in range(len(TARGET_COLUMNS))
    ]
    return {
        "model": model,
        "repeat": repeat,
        "outer_fold": outer_fold,
        "n_observations": int(len(truth)),
        "n_targets": int(len(TARGET_COLUMNS)),
        "mae_normalized": float(np.mean(np.abs(normalized_residual))),
        "rmse_normalized": float(
            np.sqrt(np.mean(np.square(normalized_residual)))
        ),
        "r2_macro": float(np.nanmean(r2_values)),
        "optimization_and_fit_seconds": optimization_and_fit_seconds,
    }


def prediction_rows(
    *,
    model: str,
    repeat: int,
    outer_fold: int,
    validation_indices: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: pd.DataFrame,
    prediction: np.ndarray,
    scales: dict[str, float],
    selected_family_by_target: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    input_values = X_validation[INPUT_COLUMNS].reset_index(drop=True)
    truth_values = y_validation[TARGET_COLUMNS].reset_index(drop=True)
    for local_index, row_index in enumerate(validation_indices):
        common = {
            "repeat": repeat,
            "outer_fold": outer_fold,
            "row_index": int(row_index),
            **{
                input_name: input_values.iloc[local_index][input_name]
                for input_name in INPUT_COLUMNS
            },
        }
        for target_index, target in enumerate(TARGET_COLUMNS):
            observed = float(truth_values.iloc[local_index][target])
            predicted = float(prediction[local_index, target_index])
            residual = observed - predicted
            rows.append(
                {
                    **common,
                    "model": model,
                    "target": target,
                    "observed": observed,
                    "predicted": predicted,
                    "residual": residual,
                    "absolute_error": abs(residual),
                    "squared_error": residual**2,
                    "normalized_residual_train_std": (
                        residual / scales[target]
                    ),
                    "selected_family": (
                        selected_family_by_target[target]
                        if selected_family_by_target is not None
                        else model
                    ),
                }
            )
    return rows


def add_naive_comparisons(
    frame: pd.DataFrame,
    *,
    keys: Sequence[str],
) -> pd.DataFrame:
    """Ajoute les écarts appariés au prédicteur naïf sur les mêmes observations."""

    baseline_metrics = ["mae", "rmse", "r2"]
    if "target" not in frame.columns:
        baseline_metrics = ["mae_normalized", "rmse_normalized", "r2_macro"]
    baseline = frame.loc[
        frame["model"] == NAIVE_MODEL_NAME,
        [*keys, *baseline_metrics],
    ].rename(
        columns={
            metric: f"naive_{metric}"
            for metric in baseline_metrics
        }
    )
    result = frame.merge(
        baseline,
        on=list(keys),
        how="left",
        validate="many_to_one",
    )
    if "target" in frame.columns:
        result["mae_improvement_vs_naive_pct"] = 100.0 * (
            result["naive_mae"] - result["mae"]
        ) / result["naive_mae"].replace(0.0, np.nan)
        result["rmse_improvement_vs_naive_pct"] = 100.0 * (
            result["naive_rmse"] - result["rmse"]
        ) / result["naive_rmse"].replace(0.0, np.nan)
        result["r2_delta_vs_naive"] = result["r2"] - result["naive_r2"]
    else:
        result["mae_improvement_vs_naive_pct"] = 100.0 * (
            result["naive_mae_normalized"] - result["mae_normalized"]
        ) / result["naive_mae_normalized"].replace(0.0, np.nan)
        result["rmse_improvement_vs_naive_pct"] = 100.0 * (
            result["naive_rmse_normalized"] - result["rmse_normalized"]
        ) / result["naive_rmse_normalized"].replace(0.0, np.nan)
        result["r2_delta_vs_naive"] = (
            result["r2_macro"] - result["naive_r2_macro"]
        )
    return result


def repeat_metrics_from_oof(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regroupe les prédictions hors pli avant de calculer chaque répétition."""

    output_rows: list[dict[str, object]] = []
    for (model, repeat, target), group in predictions.groupby(
        ["model", "repeat", "target"],
        sort=False,
    ):
        truth = group["observed"].to_numpy(dtype=float)
        estimate = group["predicted"].to_numpy(dtype=float)
        residual = truth - estimate
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        observed_mean = float(np.mean(truth))
        observed_std = float(np.std(truth, ddof=0))
        observed_range = float(np.max(truth) - np.min(truth))
        normalized = group[
            "normalized_residual_train_std"
        ].to_numpy(dtype=float)
        output_rows.append(
            {
                "model": model,
                "repeat": int(repeat),
                "target": target,
                "n_observations": int(len(group)),
                "mae": mae,
                "rmse": rmse,
                "r2": _r2(truth, estimate),
                "observed_mean": observed_mean,
                "observed_std": observed_std,
                "observed_range": observed_range,
                "mae_over_train_std": float(
                    np.mean(np.abs(normalized))
                ),
                "rmse_over_train_std": float(
                    np.sqrt(np.mean(np.square(normalized)))
                ),
                "rmse_over_observed_mean_pct": (
                    100.0 * _safe_divide(rmse, abs(observed_mean))
                ),
                "nrmse_observed_std": _safe_divide(rmse, observed_std),
                "nrmse_observed_range_pct": (
                    100.0 * _safe_divide(rmse, observed_range)
                ),
            }
        )
    by_output = pd.DataFrame(output_rows)

    global_rows: list[dict[str, object]] = []
    for (model, repeat), group in predictions.groupby(
        ["model", "repeat"],
        sort=False,
    ):
        normalized = group[
            "normalized_residual_train_std"
        ].to_numpy(dtype=float)
        target_r2 = (
            by_output.loc[
                (by_output["model"] == model)
                & (by_output["repeat"] == repeat),
                "r2",
            ]
            .astype(float)
            .to_numpy()
        )
        global_rows.append(
            {
                "model": model,
                "repeat": int(repeat),
                "n_observations": int(
                    group[["row_index", "target"]].drop_duplicates().shape[0]
                    / len(TARGET_COLUMNS)
                ),
                "n_targets": int(len(TARGET_COLUMNS)),
                "mae_normalized": float(np.mean(np.abs(normalized))),
                "rmse_normalized": float(
                    np.sqrt(np.mean(np.square(normalized)))
                ),
                "r2_macro": float(np.nanmean(target_r2)),
            }
        )
    return pd.DataFrame(global_rows), by_output


def confidence_summary(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
    sampling_unit: str,
) -> pd.DataFrame:
    """Moyenne, écart-type et IC bilatéral de Student à 95 %."""

    rows: list[dict[str, object]] = []
    grouper: str | list[str]
    grouper = (
        group_columns[0]
        if len(group_columns) == 1
        else list(group_columns)
    )
    for keys, group in frame.groupby(grouper, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, object] = dict(zip(group_columns, key_values))
        row["sampling_unit"] = sampling_unit
        for metric in metric_columns:
            values = (
                pd.to_numeric(group[metric], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .to_numpy(dtype=float)
            )
            n = int(len(values))
            mean = float(np.mean(values)) if n else np.nan
            if n >= 2:
                std = float(np.std(values, ddof=1))
                sem = std / np.sqrt(n)
                margin = float(student_t.ppf(0.975, n - 1) * sem)
                low = mean - margin
                high = mean + margin
            else:
                std = np.nan
                sem = np.nan
                low = np.nan
                high = np.nan
            row[f"{metric}_n"] = n
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_sem"] = sem
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _inner_scores_and_selection(
    *,
    trained: Any,
    repeat: int,
    outer_fold: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    trials = trained.trials.copy()
    completed = trials.loc[
        trials["Etat"].eq("COMPLETE")
        & pd.to_numeric(
            trials["RMSE_CV_normalisee"], errors="coerce"
        ).notna()
    ].copy()
    if completed.empty:
        raise RuntimeError(
            "Aucun essai Optuna interne n'a été mené à terme."
        )

    rows: list[dict[str, object]] = []
    for target in TARGET_COLUMNS:
        target_candidates: list[dict[str, object]] = []
        for family_order, family in enumerate(OPTUNA_MODEL_FAMILIES):
            family_trials = completed.loc[
                completed["Famille"].eq(family)
                & completed["Sortie"].eq(target)
            ]
            if family_trials.empty:
                raise RuntimeError(
                    f"Aucun score interne pour {family} / {target}."
                )
            best_score = float(
                family_trials["RMSE_CV_normalisee"].astype(float).min()
            )
            target_candidates.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "target": target,
                    "family": family,
                    "family_order": family_order,
                    "inner_best_rmse_normalized": best_score,
                    "n_trials_complete": int(len(family_trials)),
                    "optimization_and_fit_seconds": float(
                        trained.training_seconds_by_target[family][target]
                    ),
                    "best_params": _json_dumps(
                        trained.best_params[family][target]
                    ),
                }
            )
        ordered = sorted(
            target_candidates,
            key=lambda row: (
                float(row["inner_best_rmse_normalized"]),
                int(row["family_order"]),
            ),
        )
        winner = str(ordered[0]["family"])
        for rank, row in enumerate(ordered, start=1):
            row["inner_rank"] = rank
            row["selected"] = bool(row["family"] == winner)
            rows.append(row)

    selection = pd.DataFrame(rows)
    best_family_by_target = {
        target: str(
            selection.loc[
                selection["target"].eq(target)
                & selection["selected"],
                "family",
            ].iloc[0]
        )
        for target in TARGET_COLUMNS
    }
    return selection, best_family_by_target


def _decorate_trials(
    trials: pd.DataFrame,
    *,
    repeat: int,
    outer_fold: int,
) -> pd.DataFrame:
    renamed = trials.rename(
        columns={
            "Famille": "family",
            "Sortie": "target",
            "Essai": "trial",
            "RMSE_CV_normalisee": "inner_rmse_normalized",
            "Etat": "state",
            "Hyperparametres": "params",
        }
    ).copy()
    renamed.insert(0, "outer_fold", outer_fold)
    renamed.insert(0, "repeat", repeat)
    group_min = renamed.groupby(
        ["family", "target"],
        sort=False,
    )["inner_rmse_normalized"].transform("min")
    renamed["best_trial"] = np.isclose(
        pd.to_numeric(renamed["inner_rmse_normalized"], errors="coerce"),
        pd.to_numeric(group_min, errors="coerce"),
        equal_nan=False,
    )
    return renamed


def _mlp_complexity_rows(
    *,
    trained: Any,
    repeat: int,
    outer_fold: int,
    train_size: int,
    inner_splits: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    input_width = len(MODEL_INPUT_COLUMNS)
    smallest_inner_train_size = train_size - int(
        np.ceil(train_size / inner_splits)
    )
    for family in (OPTUNA_MLP_MODEL_NAME, OPTUNA_HYBRID_MODEL_NAME):
        for target in TARGET_COLUMNS:
            params = trained.best_params[family][target]
            layers = tuple(
                int(params[f"units_layer_{index}"])
                for index in range(1, int(params["n_layers"]) + 1)
            )
            widths = (input_width, *layers, 1)
            parameter_count = int(
                sum(
                    widths[index] * widths[index + 1] + widths[index + 1]
                    for index in range(len(widths) - 1)
                )
            )
            outer_ratio = _safe_divide(
                float(train_size),
                float(parameter_count),
            )
            inner_ratio = _safe_divide(
                float(smallest_inner_train_size),
                float(parameter_count),
            )
            rows.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "family": family,
                    "mlp_role": (
                        "MLP autonome"
                        if family == OPTUNA_MLP_MODEL_NAME
                        else "Branche MLP de l'hybride"
                    ),
                    "target": target,
                    "architecture": "-".join(map(str, widths)),
                    "hidden_layers": "-".join(map(str, layers)),
                    "input_features": input_width,
                    "output_width": 1,
                    "parameter_count": parameter_count,
                    "outer_train_observations": train_size,
                    "outer_observations_per_parameter": outer_ratio,
                    "outer_rule_5x_satisfied": bool(outer_ratio >= 5.0),
                    "outer_rule_10x_satisfied": bool(outer_ratio >= 10.0),
                    "smallest_inner_train_observations": (
                        smallest_inner_train_size
                    ),
                    "inner_observations_per_parameter": inner_ratio,
                    "inner_rule_5x_satisfied": bool(inner_ratio >= 5.0),
                    "inner_rule_10x_satisfied": bool(inner_ratio >= 10.0),
                    "best_params": _json_dumps(params),
                }
            )
    return rows


def _model_predictions(
    *,
    trained: Any,
    X_validation: pd.DataFrame,
    y_train: pd.DataFrame,
    best_family_by_target: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    predictions: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    naive_started = perf_counter()
    naive_vector = y_train[TARGET_COLUMNS].mean(axis=0).to_numpy(dtype=float)
    predictions[NAIVE_MODEL_NAME] = np.tile(
        naive_vector,
        (len(X_validation), 1),
    )
    timings[NAIVE_MODEL_NAME] = perf_counter() - naive_started

    for family in OPTUNA_MODEL_FAMILIES:
        prediction = np.asarray(
            trained.family_models[family].predict(X_validation),
            dtype=float,
        )
        if prediction.shape != (len(X_validation), len(TARGET_COLUMNS)):
            raise RuntimeError(
                f"Forme de prédiction inattendue pour {family}: "
                f"{prediction.shape}."
            )
        predictions[family] = prediction
        timings[family] = float(trained.training_seconds[family])

    selector = AutomaticTargetSelector(
        family_models=trained.family_models,
        best_family_by_target=best_family_by_target,
    )
    predictions[SELECTED_MODEL_NAME] = np.asarray(
        selector.predict(X_validation),
        dtype=float,
    )
    timings[SELECTED_MODEL_NAME] = float(
        sum(trained.training_seconds.values())
    )
    return predictions, timings


def _stratified_subset(
    *,
    X_train: pd.DataFrame,
    fraction: float,
    seed: int,
) -> np.ndarray:
    if fraction >= 1.0:
        return np.arange(len(X_train), dtype=int)
    strata, _ = temperature_deciles(
        X_train["Temperature_C"],
        required_count_per_bin=2,
    )
    requested = max(
        int(np.ceil(fraction * len(X_train))),
        int(strata.nunique()) * 2,
    )
    requested = min(requested, len(X_train) - int(strata.nunique()))
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=requested,
        random_state=seed,
    )
    subset, _ = next(splitter.split(X_train, strata))
    return np.sort(np.asarray(subset, dtype=int))


def _learning_curve_rows(
    *,
    trained: Any,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_validation: pd.DataFrame,
    best_family_by_target: dict[str, str],
    full_predictions: dict[str, np.ndarray],
    repeat: int,
    outer_fold: int,
    seed: int,
) -> list[dict[str, object]]:
    """Courbe facultative avec hyperparamètres figés après la recherche interne."""

    rows: list[dict[str, object]] = []
    for fraction_index, fraction in enumerate(LEARNING_FRACTIONS):
        subset_index = _stratified_subset(
            X_train=X_train,
            fraction=fraction,
            seed=seed + fraction_index,
        )
        X_subset = X_train.iloc[subset_index]
        y_subset = y_train.iloc[subset_index]
        scales = training_scales(y_subset)
        predictions: dict[str, np.ndarray] = {}
        if fraction >= 1.0:
            predictions = full_predictions
        else:
            naive_vector = (
                y_subset[TARGET_COLUMNS].mean(axis=0).to_numpy(dtype=float)
            )
            predictions[NAIVE_MODEL_NAME] = np.tile(
                naive_vector,
                (len(X_validation), 1),
            )
            subset_family_models: dict[str, IndependentTargetRegressor] = {}
            for family in OPTUNA_MODEL_FAMILIES:
                estimators: dict[str, object] = {}
                for target in TARGET_COLUMNS:
                    estimator = clone(
                        trained.family_models[family].estimators[target]
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        estimator.fit(X_subset, y_subset[target])
                    estimators[target] = estimator
                subset_family_models[family] = IndependentTargetRegressor(
                    family_name=family,
                    estimators=estimators,
                )
                predictions[family] = np.asarray(
                    subset_family_models[family].predict(X_validation),
                    dtype=float,
                )
            selector = AutomaticTargetSelector(
                family_models=subset_family_models,
                best_family_by_target=best_family_by_target,
            )
            predictions[SELECTED_MODEL_NAME] = np.asarray(
                selector.predict(X_validation),
                dtype=float,
            )

        for model, prediction in predictions.items():
            metric = global_metric_row(
                model=model,
                repeat=repeat,
                outer_fold=outer_fold,
                truth=y_validation,
                prediction=prediction,
                scales=scales,
                optimization_and_fit_seconds=np.nan,
            )
            rows.append(
                {
                    "model": model,
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "training_fraction": fraction,
                    "training_size": int(len(subset_index)),
                    "validation_size": int(len(X_validation)),
                    "mae_normalized": metric["mae_normalized"],
                    "rmse_normalized": metric["rmse_normalized"],
                    "r2_macro": metric["r2_macro"],
                    "hyperparameters_retuned": False,
                }
            )
    return rows


def _selection_summaries(
    selection: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = selection.loc[selection["selected"]].copy()
    frequency = (
        selected.groupby(["target", "family"], as_index=False, sort=False)
        .agg(
            selected_count=("selected", "size"),
            mean_inner_rmse_normalized=(
                "inner_best_rmse_normalized",
                "mean",
            ),
            std_inner_rmse_normalized=(
                "inner_best_rmse_normalized",
                "std",
            ),
        )
    )
    totals = selected.groupby("target").size().rename("outer_folds")
    frequency = frequency.merge(totals, on="target", how="left")
    frequency["selection_rate"] = (
        frequency["selected_count"] / frequency["outer_folds"]
    )
    family_order = {
        family: index
        for index, family in enumerate(OPTUNA_MODEL_FAMILIES)
    }
    frequency["_family_order"] = frequency["family"].map(family_order)
    frequency = frequency.sort_values(
        [
            "target",
            "selected_count",
            "mean_inner_rmse_normalized",
            "_family_order",
        ],
        ascending=[True, False, True, True],
    ).drop(columns="_family_order")
    recommended = frequency.groupby(
        "target",
        as_index=False,
        sort=False,
    ).head(1).rename(
        columns={
            "family": "recommended_family",
            "selection_rate": "recommendation_stability",
        }
    )
    recommended["rule"] = (
        "mode des sélections internes; départage par RMSE interne moyen"
    )
    return frequency.reset_index(drop=True), recommended.reset_index(drop=True)


def _validate_oof_coverage(
    predictions: pd.DataFrame,
    *,
    n_rows: int,
    protocol: Protocol,
) -> None:
    expected_models = {
        NAIVE_MODEL_NAME,
        *OPTUNA_MODEL_FAMILIES,
        SELECTED_MODEL_NAME,
    }
    if set(predictions["model"].unique()) != expected_models:
        raise RuntimeError(
            "Les modèles présents dans les prédictions hors pli sont incomplets."
        )
    counts = (
        predictions.groupby(
            ["model", "repeat", "target"],
            sort=False,
        )["row_index"]
        .agg(["count", "nunique"])
        .reset_index()
    )
    invalid = counts.loc[
        counts["count"].ne(n_rows) | counts["nunique"].ne(n_rows)
    ]
    if not invalid.empty:
        raise RuntimeError(
            "Couverture OOF invalide : chaque observation doit apparaître "
            "exactement une fois par répétition, modèle et sortie."
        )
    if predictions["repeat"].nunique() != protocol.repeats:
        raise RuntimeError("Nombre de répétitions OOF inattendu.")


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        frame.to_csv(path, index=False, encoding="utf-8-sig")


def run(
    *,
    data_path: Path | None,
    output_dir: Path,
    protocol: Protocol,
) -> dict[str, Path]:
    """Exécute le protocole et renvoie les chemins effectivement écrits."""

    signature = inspect.signature(train_optuna_families)
    required_parameters = {"nested_mode", "parallel_jobs"}
    missing_parameters = sorted(
        required_parameters.difference(signature.parameters)
    )
    if missing_parameters:
        raise RuntimeError(
            "Extension requise dans src.optuna_training : "
            "`train_optuna_families` ne fournit pas "
            + ", ".join(missing_parameters)
            + "."
        )

    resolved_data_path = (
        Path(data_path).resolve()
        if data_path is not None
        else find_data_file().resolve()
    )
    dataset_sha256 = _sha256_bytes(resolved_data_path.read_bytes())
    frame = load_dataset(resolved_data_path)
    X = frame[INPUT_COLUMNS].copy().reset_index(drop=True)
    y = frame[TARGET_COLUMNS].copy().reset_index(drop=True)
    _validate_protocol(protocol, len(frame))

    strata, actual_temperature_bins = temperature_deciles(
        X["Temperature_C"],
        required_count_per_bin=protocol.outer_splits,
    )
    outer_splitter = RepeatedStratifiedKFold(
        n_splits=protocol.outer_splits,
        n_repeats=protocol.repeats,
        random_state=protocol.seed,
    )
    splits = list(outer_splitter.split(X, strata))

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    protocol_signature = _protocol_signature(protocol)
    implementation_signature = _implementation_signature()
    started_at = datetime.now(timezone.utc)
    total_started = perf_counter()
    outer_global_rows: list[dict[str, object]] = []
    outer_output_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    selection_frames: list[pd.DataFrame] = []
    trial_frames: list[pd.DataFrame] = []
    split_manifest_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []
    inner_manifest_rows: list[dict[str, object]] = []
    mlp_complexity_rows: list[dict[str, object]] = []
    learning_rows: list[dict[str, object]] = []
    checkpoints_reused = 0
    checkpoints_written = 0
    checkpoints_ignored = 0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for split_number, (train_index, validation_index) in enumerate(
        splits,
        start=1,
    ):
        repeat = (split_number - 1) // protocol.outer_splits + 1
        outer_fold = (split_number - 1) % protocol.outer_splits + 1
        train_index_sha256 = _indices_sha256(train_index)
        validation_index_sha256 = _indices_sha256(validation_index)
        checkpoint_path = _checkpoint_path(
            checkpoint_dir,
            split_number=split_number,
            repeat=repeat,
            outer_fold=outer_fold,
        )
        checkpoint_payload, checkpoint_error = _load_checkpoint(
            checkpoint_path,
            dataset_sha256=dataset_sha256,
            protocol_signature=protocol_signature,
            implementation_signature=implementation_signature,
            split_number=split_number,
            repeat=repeat,
            outer_fold=outer_fold,
            train_index_sha256=train_index_sha256,
            validation_index_sha256=validation_index_sha256,
        )
        if checkpoint_error is not None:
            checkpoints_ignored += 1
            print(
                f"[{split_number}/{protocol.total_outer_folds}] "
                f"checkpoint ignoré ({checkpoint_error}); recalcul du pli.",
                flush=True,
            )
        if checkpoint_payload is not None:
            outer_global_rows.extend(
                checkpoint_payload["outer_global_rows"]
            )
            outer_output_rows.extend(
                checkpoint_payload["outer_output_rows"]
            )
            oof_rows.extend(checkpoint_payload["oof_rows"])
            selection_frames.append(
                pd.DataFrame(checkpoint_payload["selection_rows"])
            )
            trial_frames.append(
                pd.DataFrame(checkpoint_payload["trial_rows"])
            )
            split_manifest_rows.extend(
                checkpoint_payload["split_manifest_rows"]
            )
            assignment_rows.extend(
                checkpoint_payload["assignment_rows"]
            )
            inner_manifest_rows.extend(
                checkpoint_payload["inner_manifest_rows"]
            )
            mlp_complexity_rows.extend(
                checkpoint_payload["mlp_complexity_rows"]
            )
            learning_rows.extend(checkpoint_payload["learning_rows"])
            checkpoints_reused += 1
            print(
                f"[{split_number}/{protocol.total_outer_folds}] "
                f"répétition={repeat}, pli_externe={outer_fold} "
                "repris depuis un checkpoint vérifié.",
                flush=True,
            )
            continue

        print(
            f"[{split_number}/{protocol.total_outer_folds}] "
            f"répétition={repeat}, pli_externe={outer_fold}",
            flush=True,
        )

        fold_starts = {
            "outer_global_rows": len(outer_global_rows),
            "outer_output_rows": len(outer_output_rows),
            "oof_rows": len(oof_rows),
            "selection_frames": len(selection_frames),
            "trial_frames": len(trial_frames),
            "split_manifest_rows": len(split_manifest_rows),
            "assignment_rows": len(assignment_rows),
            "inner_manifest_rows": len(inner_manifest_rows),
            "mlp_complexity_rows": len(mlp_complexity_rows),
            "learning_rows": len(learning_rows),
        }

        X_train = X.iloc[train_index].reset_index(drop=True)
        y_train = y.iloc[train_index].reset_index(drop=True)
        X_validation = X.iloc[validation_index].reset_index(drop=True)
        y_validation = y.iloc[validation_index].reset_index(drop=True)
        scales = training_scales(y_train)

        # Reproduction exacte des plis internes employés par
        # src.optuna_training.train_optuna_families.
        inner_splitter = KFold(
            n_splits=protocol.inner_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        for inner_fold, (
            inner_train_local,
            inner_validation_local,
        ) in enumerate(inner_splitter.split(X_train), start=1):
            inner_train_global = train_index[inner_train_local]
            inner_validation_global = train_index[inner_validation_local]
            overlap_outer = np.intersect1d(
                inner_validation_global,
                validation_index,
            )
            inner_manifest_rows.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "inner_train_n": int(len(inner_train_global)),
                    "inner_validation_n": int(
                        len(inner_validation_global)
                    ),
                    "outer_validation_overlap_n": int(len(overlap_outer)),
                    "inner_train_row_indices": _json_dumps(
                        inner_train_global
                    ),
                    "inner_validation_row_indices": _json_dumps(
                        inner_validation_global
                    ),
                }
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = train_optuna_families(
                X_train,
                y_train,
                n_trials=protocol.trials,
                inner_splits=protocol.inner_splits,
                parallel_jobs=protocol.parallel_jobs,
                fast=protocol.quick,
                nested_mode=True,
                progress=lambda message: print(
                    f"  {message}",
                    flush=True,
                ),
            )

        selection, best_family_by_target = _inner_scores_and_selection(
            trained=trained,
            repeat=repeat,
            outer_fold=outer_fold,
        )
        selection_frames.append(selection)
        trial_frames.append(
            _decorate_trials(
                trained.trials,
                repeat=repeat,
                outer_fold=outer_fold,
            )
        )
        mlp_complexity_rows.extend(
            _mlp_complexity_rows(
                trained=trained,
                repeat=repeat,
                outer_fold=outer_fold,
                train_size=len(X_train),
                inner_splits=protocol.inner_splits,
            )
        )

        predictions, timings = _model_predictions(
            trained=trained,
            X_validation=X_validation,
            y_train=y_train,
            best_family_by_target=best_family_by_target,
        )
        for model, prediction in predictions.items():
            outer_global_rows.append(
                global_metric_row(
                    model=model,
                    repeat=repeat,
                    outer_fold=outer_fold,
                    truth=y_validation,
                    prediction=prediction,
                    scales=scales,
                    optimization_and_fit_seconds=timings[model],
                )
            )
            for target_index, target in enumerate(TARGET_COLUMNS):
                target_seconds = (
                    float(
                        trained.training_seconds_by_target[model][target]
                    )
                    if model in OPTUNA_MODEL_FAMILIES
                    else (
                        float(
                            sum(
                                trained.training_seconds_by_target[
                                    family
                                ][target]
                                for family in OPTUNA_MODEL_FAMILIES
                            )
                        )
                        if model == SELECTED_MODEL_NAME
                        else timings[model]
                    )
                )
                outer_output_rows.append(
                    output_metric_row(
                        model=model,
                        repeat=repeat,
                        outer_fold=outer_fold,
                        target=target,
                        truth=y_validation[target].to_numpy(dtype=float),
                        prediction=prediction[:, target_index],
                        train_truth=y_train[target],
                        optimization_and_fit_seconds=target_seconds,
                    )
                )
            oof_rows.extend(
                prediction_rows(
                    model=model,
                    repeat=repeat,
                    outer_fold=outer_fold,
                    validation_indices=validation_index,
                    X_validation=X_validation,
                    y_validation=y_validation,
                    prediction=prediction,
                    scales=scales,
                    selected_family_by_target=(
                        best_family_by_target
                        if model == SELECTED_MODEL_NAME
                        else None
                    ),
                )
            )

        if protocol.learning_curve:
            learning_rows.extend(
                _learning_curve_rows(
                    trained=trained,
                    X_train=X_train,
                    y_train=y_train,
                    X_validation=X_validation,
                    y_validation=y_validation,
                    best_family_by_target=best_family_by_target,
                    full_predictions=predictions,
                    repeat=repeat,
                    outer_fold=outer_fold,
                    seed=protocol.seed + split_number * 1000,
                )
            )

        train_temperature = X_train["Temperature_C"]
        validation_temperature = X_validation["Temperature_C"]
        split_manifest_rows.append(
            {
                "repeat": repeat,
                "outer_fold": outer_fold,
                "outer_train_n": int(len(train_index)),
                "outer_validation_n": int(len(validation_index)),
                "outer_train_temperature_min": _safe_float(
                    train_temperature.min()
                ),
                "outer_train_temperature_max": _safe_float(
                    train_temperature.max()
                ),
                "outer_validation_temperature_min": _safe_float(
                    validation_temperature.min()
                ),
                "outer_validation_temperature_max": _safe_float(
                    validation_temperature.max()
                ),
                "outer_validation_row_indices": _json_dumps(
                    validation_index
                ),
                "outer_validation_index_sha256": hashlib.sha256(
                    np.asarray(validation_index, dtype=np.int64).tobytes()
                ).hexdigest(),
                "inner_outer_overlap_total": int(
                    sum(
                        row["outer_validation_overlap_n"]
                        for row in inner_manifest_rows
                        if row["repeat"] == repeat
                        and row["outer_fold"] == outer_fold
                    )
                ),
            }
        )
        for row_index in validation_index:
            assignment_rows.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "row_index": int(row_index),
                    "temperature_decile": int(strata.iloc[row_index]),
                }
            )

        fold_selection = pd.concat(
            selection_frames[fold_starts["selection_frames"]:],
            ignore_index=True,
        )
        fold_trials = pd.concat(
            trial_frames[fold_starts["trial_frames"]:],
            ignore_index=True,
        )
        checkpoint_payload = {
            "outer_global_rows": outer_global_rows[
                fold_starts["outer_global_rows"]:
            ],
            "outer_output_rows": outer_output_rows[
                fold_starts["outer_output_rows"]:
            ],
            "oof_rows": oof_rows[fold_starts["oof_rows"]:],
            "selection_rows": fold_selection.to_dict(orient="records"),
            "trial_rows": fold_trials.to_dict(orient="records"),
            "split_manifest_rows": split_manifest_rows[
                fold_starts["split_manifest_rows"]:
            ],
            "assignment_rows": assignment_rows[
                fold_starts["assignment_rows"]:
            ],
            "inner_manifest_rows": inner_manifest_rows[
                fold_starts["inner_manifest_rows"]:
            ],
            "mlp_complexity_rows": mlp_complexity_rows[
                fold_starts["mlp_complexity_rows"]:
            ],
            "learning_rows": learning_rows[
                fold_starts["learning_rows"]:
            ],
        }
        checkpoint_metadata = {
            "dataset_sha256": dataset_sha256,
            "protocol_signature": protocol_signature,
            "implementation_signature": implementation_signature,
            "split_number": split_number,
            "repeat": repeat,
            "outer_fold": outer_fold,
            "train_index_sha256": train_index_sha256,
            "validation_index_sha256": validation_index_sha256,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_checkpoint_atomic(
            checkpoint_path,
            metadata=checkpoint_metadata,
            payload=checkpoint_payload,
        )
        checkpoints_written += 1
        print(
            f"  checkpoint atomique conservé : {checkpoint_path.name}",
            flush=True,
        )

    outer_global = add_naive_comparisons(
        pd.DataFrame(outer_global_rows),
        keys=["repeat", "outer_fold"],
    )
    outer_output = add_naive_comparisons(
        pd.DataFrame(outer_output_rows),
        keys=["repeat", "outer_fold", "target"],
    )
    oof = pd.DataFrame(oof_rows)
    _validate_oof_coverage(
        oof,
        n_rows=len(frame),
        protocol=protocol,
    )
    repeat_global, repeat_output = repeat_metrics_from_oof(oof)
    repeat_global = add_naive_comparisons(
        repeat_global,
        keys=["repeat"],
    )
    repeat_output = add_naive_comparisons(
        repeat_output,
        keys=["repeat", "target"],
    )

    # Les IC principaux emploient la répétition complète comme unité
    # d'échantillonnage. Les résumés par pli sont secondaires et descriptifs,
    # car les ensembles d'entraînement se chevauchent.
    summary_global = confidence_summary(
        repeat_global,
        group_columns=["model"],
        metric_columns=PRIMARY_METRICS_GLOBAL,
        sampling_unit="répétition OOF complète",
    ).sort_values("rmse_normalized_mean")
    summary_output = confidence_summary(
        repeat_output,
        group_columns=["model", "target"],
        metric_columns=PRIMARY_METRICS_OUTPUT,
        sampling_unit="répétition OOF complète",
    )
    outer_summary_global = confidence_summary(
        outer_global,
        group_columns=["model"],
        metric_columns=PRIMARY_METRICS_GLOBAL,
        sampling_unit="pli externe (descriptif; plis corrélés)",
    ).sort_values("rmse_normalized_mean")
    outer_summary_output = confidence_summary(
        outer_output,
        group_columns=["model", "target"],
        metric_columns=PRIMARY_METRICS_OUTPUT,
        sampling_unit="pli externe (descriptif; plis corrélés)",
    )

    selection = pd.concat(selection_frames, ignore_index=True)
    trials = pd.concat(trial_frames, ignore_index=True)
    selection_frequency, recommended = _selection_summaries(selection)
    split_manifest = pd.DataFrame(split_manifest_rows)
    assignments = pd.DataFrame(assignment_rows)
    inner_manifest = pd.DataFrame(inner_manifest_rows)
    mlp_complexity = pd.DataFrame(mlp_complexity_rows)
    learning_curve = pd.DataFrame(learning_rows)

    outputs: dict[str, tuple[pd.DataFrame, str]] = {
        "outer_fold_global_metrics": (
            outer_global,
            "outer_fold_global_metrics.csv",
        ),
        "outer_fold_output_metrics": (
            outer_output,
            "outer_fold_output_metrics.csv",
        ),
        "outer_fold_predictions": (
            oof,
            "outer_fold_predictions.csv",
        ),
        "repeat_global_metrics": (
            repeat_global,
            "repeat_global_metrics.csv",
        ),
        "repeat_output_metrics": (
            repeat_output,
            "repeat_output_metrics.csv",
        ),
        "summary_global_95ci": (
            summary_global,
            "summary_global_95ci.csv",
        ),
        "summary_by_output_95ci": (
            summary_output,
            "summary_by_output_95ci.csv",
        ),
        "outer_fold_summary_global_95ci": (
            outer_summary_global,
            "outer_fold_summary_global_95ci.csv",
        ),
        "outer_fold_summary_by_output_95ci": (
            outer_summary_output,
            "outer_fold_summary_by_output_95ci.csv",
        ),
        "inner_selection_by_output": (
            selection,
            "inner_selection_by_output.csv",
        ),
        "best_family_frequency_by_output": (
            selection_frequency,
            "best_family_frequency_by_output.csv",
        ),
        "recommended_family_by_output": (
            recommended,
            "recommended_family_by_output.csv",
        ),
        "optuna_trials": (
            trials,
            "optuna_trials.csv",
        ),
        "split_manifest": (
            split_manifest,
            "split_manifest.csv",
        ),
        "outer_fold_assignments": (
            assignments,
            "outer_fold_assignments.csv",
        ),
        "inner_split_manifest": (
            inner_manifest,
            "inner_split_manifest.csv",
        ),
        "mlp_complexity": (
            mlp_complexity,
            "mlp_complexity.csv",
        ),
    }
    if protocol.learning_curve:
        outputs["learning_curve_metrics"] = (
            learning_curve,
            "learning_curve_metrics.csv",
        )

    written: dict[str, Path] = {}
    for key, (table, filename) in outputs.items():
        destination = output_dir / filename
        _write_csv(table, destination)
        written[key] = destination

    elapsed = perf_counter() - total_started
    completed_at = datetime.now(timezone.utc)
    metadata = {
        "status": "completed",
        "scientific_status": (
            "prototype exploratoire; aucune validation industrielle"
        ),
        "problem": {
            "type": "régression supervisée multi-sorties",
            "n_outputs": len(TARGET_COLUMNS),
            "targets": TARGET_COLUMNS,
            "raw_inputs": INPUT_COLUMNS,
            "engineered_inputs": MODEL_INPUT_COLUMNS,
            "independent_target_estimators": True,
            "interpretation": (
                "Chaque famille assemble huit régressions supervisées "
                "indépendantes dans une sortie multi-dimensionnelle."
            ),
        },
        "dataset": {
            "path": str(resolved_data_path),
            "sha256": dataset_sha256,
            "rows_after_cleaning": len(frame),
            "temperature_missing_count": int(
                X["Temperature_C"].isna().sum()
            ),
        },
        "models": {
            "naive": NAIVE_MODEL_NAME,
            "optuna_families": OPTUNA_MODEL_FAMILIES,
            "automatic_selection": SELECTED_MODEL_NAME,
            "mlp_reduced_search_space": True,
            "hybrid_two_branch_design": {
                "mlp_branch": "StandardScaler + MLP compact",
                "xgboost_branch": "XGBRegressor peu profond",
                "fusion": "poids convexe optimisé dans les plis internes",
                "standalone_xgboost_active": False,
            },
            "mlp_allowed_hidden_architectures": [
                [2],
                [4],
                [2, 2],
                [4, 2],
            ],
            "mlp_max_parameters_per_target": 77,
            "mlp_smallest_inner_train_observations": int(
                mlp_complexity[
                    "smallest_inner_train_observations"
                ].min()
            ),
            "mlp_worst_case_inner_observations_per_parameter": float(
                mlp_complexity[
                    "smallest_inner_train_observations"
                ].min()
                / 77.0
            ),
            "mlp_search_space_rule_5x_inner_satisfied": bool(
                mlp_complexity[
                    "smallest_inner_train_observations"
                ].min()
                / 77.0
                >= 5.0
            ),
            "mlp_rule_5x_outer_folds_satisfied": bool(
                mlp_complexity["outer_rule_5x_satisfied"].all()
            ),
            "mlp_rule_5x_smallest_inner_folds_satisfied": bool(
                mlp_complexity["inner_rule_5x_satisfied"].all()
            ),
            "mlp_complexity_export": "mlp_complexity.csv",
        },
        "validation": {
            "outer_method": (
                "RepeatedStratifiedKFold sur quantiles de température"
            ),
            "requested_temperature_bins": 10,
            "actual_temperature_bins": actual_temperature_bins,
            "outer_splits": protocol.outer_splits,
            "repeats": protocol.repeats,
            "total_outer_folds": protocol.total_outer_folds,
            "inner_method": (
                "KFold mélangé, identique entre familles et sorties"
            ),
            "inner_splits": protocol.inner_splits,
            "trials_per_family_target_outer_fold": protocol.trials,
            "parallel_jobs": protocol.parallel_jobs,
            "total_optuna_studies": (
                protocol.total_outer_folds
                * len(OPTUNA_MODEL_FAMILIES)
                * len(TARGET_COLUMNS)
            ),
            "total_optuna_trials_requested": (
                protocol.total_outer_folds
                * len(OPTUNA_MODEL_FAMILIES)
                * len(TARGET_COLUMNS)
                * protocol.trials
            ),
            "same_outer_folds_for_all_models": True,
            "outer_validation_used_for_tuning": False,
            "preprocessing_fitted_inside_inner_fold": True,
            "inner_outer_overlap_verified_zero": bool(
                inner_manifest["outer_validation_overlap_n"].eq(0).all()
            ),
            "selection_rule": (
                "RMSE CV interne normalisée minimale, séparément par sortie"
            ),
            "predictions": (
                "hors pli externes, brutes, sans post-traitement numérique de cohérence"
            ),
            "external_campaign_available": False,
        },
        "uncertainty": {
            "confidence_level": 0.95,
            "primary_sampling_unit": "répétition OOF complète",
            "method": (
                "intervalle bilatéral de Student sur les métriques "
                "agrégées de chaque répétition"
            ),
            "fold_level_intervals": (
                "exportés à titre descriptif seulement; les plis et leurs "
                "ensembles d'entraînement ne sont pas indépendants"
            ),
            "confidence_intervals_estimable": bool(
                protocol.repeats >= 2
            ),
            "quick_mode_ci_interpretable": bool(
                not protocol.quick and protocol.repeats >= 2
            ),
        },
        "naive_comparison": {
            "strategy": (
                "moyenne de chaque sortie calculée exclusivement sur le "
                "pli externe d'entraînement"
            ),
            "paired_same_folds": True,
        },
        "learning_curve": {
            "executed": protocol.learning_curve,
            "fractions": (
                list(LEARNING_FRACTIONS)
                if protocol.learning_curve
                else []
            ),
            "hyperparameters_retuned_at_each_fraction": False,
        },
        "limitations": [
            (
                "Les répétitions réutilisent la même base et ne remplacent "
                "pas une campagne expérimentale externe."
            ),
            (
                "La stratification thermique ne remplace pas un GroupKFold "
                "par étude, lot ou feedstock lorsque ces identifiants sont "
                "absents."
            ),
            (
                "Les intervalles mesurent la variabilité des partitions, pas "
                "l'incertitude instrumentale ni l'incertitude de domaine."
            ),
            (
                "La sélection automatique est exploratoire et ne doit pas "
                "piloter seule un procédé industriel."
            ),
        ],
        "implementation": {
            "shared_training_utility": (
                "src.optuna_training.train_optuna_families"
            ),
            "nested_mode": True,
            "quick": protocol.quick,
            "parallel_jobs": protocol.parallel_jobs,
            "seed": protocol.seed,
            "checkpointing": {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "directory": str(checkpoint_dir),
                "atomic_write": True,
                "automatic_resume": True,
                "kept_after_success": True,
                "dataset_sha256_validated": True,
                "protocol_signature": protocol_signature,
                "implementation_signature": implementation_signature,
                "reused": checkpoints_reused,
                "written": checkpoints_written,
                "ignored_as_incompatible_or_invalid": checkpoints_ignored,
            },
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "optuna": optuna.__version__,
            "xgboost": xgboost.__version__,
            "command": [sys.executable, *sys.argv],
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "elapsed_seconds": elapsed,
        },
        "exports": {
            key: path.name
            for key, path in written.items()
        },
        "xlsx": {
            "created_by_this_script": False,
            "reason": (
                "Le classeur final est construit séparément à partir des CSV "
                "afin de préserver le workflow de vérification du tableur."
            ),
        },
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    written["metadata"] = metadata_path

    print("\nSynthèse globale (IC 95 % sur les répétitions)")
    print(summary_global.to_string(index=False))
    print(f"\nExports : {output_dir}")
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation répétée et imbriquée des familles Optuna actives, "
            "avec prédicteur naïf et sélection automatique par sortie."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="CSV/XLSX d'entrée; la base active est utilisée par défaut.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=(
            "Répertoire des CSV/JSON "
            "(défaut: reports/nested_repeated_validation)."
        ),
    )
    parser.add_argument(
        "--outer-splits",
        type=int,
        default=5,
        help="Nombre de plis externes par répétition (défaut: 5).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Nombre de répétitions externes (défaut: 5).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help=(
            "Essais Optuna par famille, sortie et pli externe "
            "(défaut renforcé: 20)."
        ),
    )
    parser.add_argument(
        "--inner-splits",
        type=int,
        default=3,
        help="Nombre de plis internes Optuna (défaut: 3).",
    )
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=2,
        help=(
            "Nombre de tâches d'optimisation exécutées en parallèle "
            "(défaut: 2; forcé à 1 en mode --quick)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help=f"Graine des partitions externes (défaut: {RANDOM_STATE}).",
    )
    parser.add_argument(
        "--learning-curve",
        action="store_true",
        help=(
            "Ajoute une courbe d'apprentissage à hyperparamètres figés; "
            "augmente nettement le temps de calcul."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Test court: 2 plis × 1 répétition, 2 plis internes et "
            "1 essai; non destiné à l'interprétation scientifique."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.quick:
        protocol = Protocol(
            outer_splits=2,
            repeats=1,
            trials=1,
            inner_splits=2,
            parallel_jobs=1,
            quick=True,
            learning_curve=bool(args.learning_curve),
            seed=int(args.seed),
        )
    else:
        protocol = Protocol(
            outer_splits=int(args.outer_splits),
            repeats=int(args.repeats),
            trials=int(args.trials),
            inner_splits=int(args.inner_splits),
            parallel_jobs=int(args.parallel_jobs),
            quick=False,
            learning_curve=bool(args.learning_curve),
            seed=int(args.seed),
        )
    run(
        data_path=args.data,
        output_dir=Path(args.output_dir).resolve(),
        protocol=protocol,
    )


if __name__ == "__main__":
    main()
