from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    ARTIFACT_PATH,
    CHEMICAL_PRICE_PER_KG,
    DEFAULT_OPTUNA_INNER_SPLITS,
    DEFAULT_OPTUNA_TRIALS,
    FAST_OPTUNA_INNER_SPLITS,
    FAST_OPTUNA_TRIALS,
    HALOGEN_COLUMNS,
    INPUT_COLUMNS,
    MODEL_SCHEMA_VERSION,
    NUMERIC_INPUTS,
    OPTUNA_MODEL_FAMILIES,
    OPTUNA_REPORT_DIR,
    PRODUCTION_ARTIFACT_PATH,
    PRODUCTION_MODEL_NAME,
    TARGET_COLUMNS,
    YIELD_COLUMNS,
)
from .data import DataSplits, find_data_file, split_dataset
from .features import MODEL_INPUT_COLUMNS
from .optuna_training import (
    AutomaticTargetSelector,
    train_optuna_families,
)


BASELINE_MODEL_NAME = "Baseline moyenne (référence)"


@dataclass
class ModelBundle:
    models: dict[str, object]
    best_model_name: str
    best_model_by_target: dict[str, str]
    summary_metrics: pd.DataFrame
    target_metrics: pd.DataFrame
    splits: DataSplits
    target_scale: pd.Series
    input_bounds: dict[str, tuple[float, float]]
    reference_data: pd.DataFrame
    training_seconds: dict[str, float]
    training_seconds_by_target: dict[str, dict[str, float]]
    optuna_best_params: dict[str, dict[str, dict[str, object]]]
    optuna_trials: pd.DataFrame
    optuna_protocol: dict[str, object]
    trained_at: str
    data_file: str
    data_mtime: float
    model_schema_version: int
    data_sha256: str = ""


def dense_parameter_count(
    input_features: int,
    hidden_layers: tuple[int, ...],
    outputs: int,
) -> int:
    """Nombre de poids et biais d'un réseau dense entièrement connecté."""
    sizes = (input_features,) + tuple(hidden_layers) + (outputs,)
    return int(
        sum((left + 1) * right for left, right in zip(sizes[:-1], sizes[1:]))
    )


def make_preprocessor(
    feature_columns: list[str] | None = None,
) -> ColumnTransformer:
    """Préprocesseur numérique conservé pour les expériences archivées."""
    columns = feature_columns or MODEL_INPUT_COLUMNS
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, columns)],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def enforce_physical_constraints(prediction: np.ndarray) -> np.ndarray:
    """Applique les contraintes après assemblage des huit sorties."""
    values = np.asarray(prediction, dtype=float).copy()
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != len(TARGET_COLUMNS):
        raise ValueError(
            f"{len(TARGET_COLUMNS)} sorties attendues, reçu {values.shape[1]}."
        )
    values[:, : len(YIELD_COLUMNS)] = np.clip(
        values[:, : len(YIELD_COLUMNS)], 0.0, 100.0
    )
    yield_sum = values[:, : len(YIELD_COLUMNS)].sum(axis=1, keepdims=True)
    valid = yield_sum[:, 0] > 0
    values[valid, : len(YIELD_COLUMNS)] *= 100.0 / yield_sum[valid]
    if np.any(~valid):
        values[~valid, : len(YIELD_COLUMNS)] = 100.0 / len(YIELD_COLUMNS)
    values[:, len(YIELD_COLUMNS) :] = np.clip(
        values[:, len(YIELD_COLUMNS) :], 0.0, None
    )
    return values


def _prediction_frame(
    prediction: np.ndarray,
    index: pd.Index,
    *,
    apply_constraints: bool,
) -> pd.DataFrame:
    values = (
        enforce_physical_constraints(prediction)
        if apply_constraints
        else np.asarray(prediction, dtype=float)
    )
    return pd.DataFrame(
        values,
        columns=TARGET_COLUMNS,
        index=index,
    )


def _metric_rows(
    model_name: str,
    split_name: str,
    y_true: pd.DataFrame,
    prediction: np.ndarray,
    target_scale: pd.Series,
    training_seconds: float,
    target_training_seconds: dict[str, float] | None = None,
    apply_constraints: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    pred = _prediction_frame(
        prediction,
        y_true.index,
        apply_constraints=apply_constraints,
    )
    scale = target_scale.replace(0, 1.0).to_numpy()
    residual_normalized = (y_true[TARGET_COLUMNS].to_numpy() - pred.to_numpy()) / scale
    summary = {
        "Modèle": model_name,
        "Jeu": split_name,
        "MAE normalisée": float(np.mean(np.abs(residual_normalized))),
        "RMSE normalisé": float(np.sqrt(np.mean(residual_normalized**2))),
        "R² macro": float(
            r2_score(
                y_true[TARGET_COLUMNS],
                pred,
                multioutput="uniform_average",
            )
        ),
        "Temps entraînement (s)": float(training_seconds),
        "Post-traitement numérique": "Oui" if apply_constraints else "Non",
    }
    details: list[dict[str, object]] = []
    for target in TARGET_COLUMNS:
        details.append(
            {
                "Modèle": model_name,
                "Jeu": split_name,
                "Sortie": target,
                "MAE": float(mean_absolute_error(y_true[target], pred[target])),
                "RMSE": float(
                    np.sqrt(mean_squared_error(y_true[target], pred[target]))
                ),
                "R²": float(r2_score(y_true[target], pred[target])),
                "Temps entraînement (s)": float(
                    (target_training_seconds or {}).get(
                        target,
                        training_seconds,
                    )
                ),
                "Post-traitement numérique": (
                    "Oui" if apply_constraints else "Non"
                ),
            }
        )
    return summary, details


def _baseline_prediction(
    y_train: pd.DataFrame,
    rows: int,
) -> np.ndarray:
    means = y_train[TARGET_COLUMNS].mean(axis=0).to_numpy(dtype=float)
    return np.tile(means, (rows, 1))


def _select_best_family_by_target(
    validation_details: pd.DataFrame,
) -> dict[str, str]:
    candidates = validation_details[
        validation_details["Modèle"].isin(OPTUNA_MODEL_FAMILIES)
    ]
    selected: dict[str, str] = {}
    for target in TARGET_COLUMNS:
        rows = candidates[candidates["Sortie"] == target].sort_values(
            ["RMSE", "MAE", "Modèle"],
            kind="stable",
        )
        if rows.empty:
            raise ValueError(f"Aucune métrique de validation pour {target}.")
        selected[target] = str(rows.iloc[0]["Modèle"])
    return selected


def train_all_models(
    df: pd.DataFrame,
    *,
    fast: bool = False,
    data_path: str | Path | None = None,
    n_trials: int | None = None,
    inner_splits: int | None = None,
    parallel_jobs: int = 1,
    progress: Callable[[str], None] | None = None,
) -> ModelBundle:
    """Optimise quatre familles et sélectionne la meilleure pour chaque sortie.

    Optuna ne voit que les 80 % d'entraînement via une validation croisée
    interne. Les 10 % de validation choisissent la famille par cible. Les 10 %
    de test fournissent ensuite une estimation interne sur un découpage unique.
    """

    splits = split_dataset(df)
    trials_per_target = int(
        n_trials
        if n_trials is not None
        else (FAST_OPTUNA_TRIALS if fast else DEFAULT_OPTUNA_TRIALS)
    )
    cv_splits = int(
        inner_splits
        if inner_splits is not None
        else (
            FAST_OPTUNA_INNER_SPLITS
            if fast
            else DEFAULT_OPTUNA_INNER_SPLITS
        )
    )
    target_scale = splits.y_train.std(ddof=0).replace(0, 1.0)

    trained = train_optuna_families(
        splits.X_train,
        splits.y_train,
        n_trials=trials_per_target,
        inner_splits=cv_splits,
        fast=fast,
        parallel_jobs=parallel_jobs,
        progress=progress,
    )
    models: dict[str, object] = dict(trained.family_models)
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for split_name, X_part, y_part in (
        ("Validation", splits.X_validation, splits.y_validation),
        ("Test", splits.X_test, splits.y_test),
    ):
        for family in OPTUNA_MODEL_FAMILIES:
            prediction = models[family].predict(X_part)
            summary, details = _metric_rows(
                family,
                split_name,
                y_part,
                prediction,
                target_scale,
                trained.training_seconds[family],
                trained.training_seconds_by_target[family],
            )
            summary_rows.append(summary)
            detail_rows.extend(details)

        baseline = _baseline_prediction(splits.y_train, len(X_part))
        summary, details = _metric_rows(
            BASELINE_MODEL_NAME,
            split_name,
            y_part,
            baseline,
            target_scale,
            0.0,
        )
        summary_rows.append(summary)
        detail_rows.extend(details)

    interim_details = pd.DataFrame(detail_rows)
    best_by_target = _select_best_family_by_target(
        interim_details[interim_details["Jeu"] == "Validation"]
    )
    automatic = AutomaticTargetSelector(
        family_models=trained.family_models,
        best_family_by_target=best_by_target,
    )
    models[PRODUCTION_MODEL_NAME] = automatic
    automatic_training_seconds = float(sum(trained.training_seconds.values()))
    automatic_target_seconds = {
        target: trained.training_seconds_by_target[family][target]
        for target, family in best_by_target.items()
    }

    for split_name, X_part, y_part in (
        ("Validation", splits.X_validation, splits.y_validation),
        ("Test", splits.X_test, splits.y_test),
    ):
        summary, details = _metric_rows(
            PRODUCTION_MODEL_NAME,
            split_name,
            y_part,
            automatic.predict(X_part),
            target_scale,
            automatic_training_seconds,
            automatic_target_seconds,
            apply_constraints=True,
        )
        summary_rows.append(summary)
        detail_rows.extend(details)

    source = Path(data_path) if data_path else find_data_file()
    bounds = {
        column: (float(df[column].min()), float(df[column].max()))
        for column in NUMERIC_INPUTS
    }
    return ModelBundle(
        models=models,
        best_model_name=PRODUCTION_MODEL_NAME,
        best_model_by_target=best_by_target,
        summary_metrics=pd.DataFrame(summary_rows),
        target_metrics=pd.DataFrame(detail_rows),
        splits=splits,
        target_scale=target_scale,
        input_bounds=bounds,
        reference_data=df.copy(),
        training_seconds={
            **trained.training_seconds,
            PRODUCTION_MODEL_NAME: automatic_training_seconds,
        },
        training_seconds_by_target={
            **trained.training_seconds_by_target,
            PRODUCTION_MODEL_NAME: automatic_target_seconds,
        },
        optuna_best_params=trained.best_params,
        optuna_trials=trained.trials,
        optuna_protocol={
            "split": "80 % entraînement / 10 % validation / 10 % test",
            "model_families": list(OPTUNA_MODEL_FAMILIES),
            "inner_validation": (
                f"{cv_splits}-fold CV dans les 80 % d'entraînement"
            ),
            "inner_splits": cv_splits,
            "trials_per_family_and_target": trials_per_target,
            "parallel_jobs": int(parallel_jobs),
            "target_selection": (
                "RMSE minimale sur les 10 % de validation, séparément par sortie"
            ),
            "test_role": (
                "estimation interne uniquement; la preuve principale est la "
                "validation croisée imbriquée répétée"
            ),
        },
        trained_at=datetime.now(timezone.utc).isoformat(),
        data_file=source.name,
        data_mtime=source.stat().st_mtime,
        model_schema_version=MODEL_SCHEMA_VERSION,
        data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )


def predict_primary(
    bundle: ModelBundle,
    inputs: pd.DataFrame,
    model_name: str | None = None,
) -> pd.DataFrame:
    selected = model_name or (
        PRODUCTION_MODEL_NAME
        if PRODUCTION_MODEL_NAME in bundle.models
        else bundle.best_model_name
    )
    if selected not in bundle.models:
        raise ValueError(f"Modèle inconnu : {selected}")
    prediction = enforce_physical_constraints(
        bundle.models[selected].predict(inputs[INPUT_COLUMNS])
    )
    result = pd.DataFrame(
        prediction,
        columns=TARGET_COLUMNS,
        index=inputs.index,
    )
    result["Toxicity_Index_pct"] = result[HALOGEN_COLUMNS].sum(axis=1)
    result["Cout_liquide_USD"] = (
        inputs["Concentration_liquide_chimique_kg"].to_numpy()
        * CHEMICAL_PRICE_PER_KG
    )
    return result


def export_training_reports(
    bundle: ModelBundle,
    directory: str | Path = OPTUNA_REPORT_DIR,
) -> Path:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    bundle.summary_metrics.to_csv(
        destination / "evaluation_globale.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bundle.target_metrics.to_csv(
        destination / "evaluation_par_sortie.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bundle.optuna_trials.to_csv(
        destination / "essais_optuna.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selection = pd.DataFrame(
        [
            {"Sortie": target, "Meilleur modèle": family}
            for target, family in bundle.best_model_by_target.items()
        ]
    )
    selection.to_csv(
        destination / "selection_par_sortie.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (destination / "meilleurs_hyperparametres.json").write_text(
        json.dumps(
            bundle.optuna_best_params,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (destination / "protocole.json").write_text(
        json.dumps(bundle.optuna_protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination


def save_bundle(
    bundle: ModelBundle,
    path: str | Path = ARTIFACT_PATH,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination, compress=3)
    return destination


def _refresh_derived_reference_columns(bundle: ModelBundle) -> ModelBundle:
    """Recalcule les colonnes de gestion qui ne sont pas apprises par les modèles.

    Les artefacts peuvent rester statistiquement valides après la modification d'une
    hypothèse économique. Le recalcul au chargement évite qu'une ancienne valeur du
    prix du liquide soit réutilisée dans la normalisation du score interactif.
    """
    reference = getattr(bundle, "reference_data", None)
    if not isinstance(reference, pd.DataFrame):
        return bundle

    refreshed = reference.copy()
    concentration = "Concentration_liquide_chimique_kg"
    if concentration in refreshed.columns:
        refreshed["Cout_liquide_USD"] = (
            pd.to_numeric(refreshed[concentration], errors="coerce")
            * CHEMICAL_PRICE_PER_KG
        )
    if set(HALOGEN_COLUMNS).issubset(refreshed.columns):
        refreshed["Toxicity_Index_pct"] = refreshed[HALOGEN_COLUMNS].sum(
            axis=1,
            min_count=1,
        )
    try:
        return replace(bundle, reference_data=refreshed)
    except TypeError:
        # Compatibilité avec les anciens objets sérialisés qui ne seraient pas
        # des instances exactes de la dataclass actuelle.
        bundle.reference_data = refreshed
        return bundle


def load_bundle(path: str | Path = ARTIFACT_PATH) -> ModelBundle:
    bundle = joblib.load(Path(path))
    return _refresh_derived_reference_columns(bundle)


def _bundle_matches_data(bundle: ModelBundle, source: Path) -> bool:
    """Use file content across Git checkouts; support older local bundles."""
    if bundle.data_file != source.name:
        return False
    digest = getattr(bundle, "data_sha256", "")
    if digest:
        return digest == hashlib.sha256(source.read_bytes()).hexdigest()
    return abs(bundle.data_mtime - source.stat().st_mtime) < 1e-6


def artifact_is_current(path: str | Path = ARTIFACT_PATH) -> bool:
    artifact = Path(path)
    if not artifact.exists():
        return False
    try:
        bundle = load_bundle(artifact)
        source = find_data_file()
        return (
            _bundle_matches_data(bundle, source)
            and getattr(bundle, "model_schema_version", None)
            == MODEL_SCHEMA_VERSION
            and _has_serious_optuna_budget(bundle)
            and set(OPTUNA_MODEL_FAMILIES).issubset(bundle.models)
            and PRODUCTION_MODEL_NAME in bundle.models
        )
    except Exception:
        return False


def _has_serious_optuna_budget(bundle: ModelBundle) -> bool:
    """Refuse les artefacts produits avec l'ancien budget exploratoire."""
    protocol = getattr(bundle, "optuna_protocol", {})
    try:
        return (
            int(protocol.get("trials_per_family_and_target", 0))
            >= DEFAULT_OPTUNA_TRIALS
            and int(protocol.get("inner_splits", 0))
            >= DEFAULT_OPTUNA_INNER_SPLITS
        )
    except (AttributeError, TypeError, ValueError):
        return False


def as_production_bundle(bundle: ModelBundle) -> ModelBundle:
    """Conserve uniquement le sélecteur par sortie et ses modèles internes."""
    if PRODUCTION_MODEL_NAME not in bundle.models:
        raise ValueError(
            f"Le modèle {PRODUCTION_MODEL_NAME!r} est absent de l'artefact."
        )
    summary = bundle.summary_metrics[
        bundle.summary_metrics["Modèle"] == PRODUCTION_MODEL_NAME
    ].reset_index(drop=True)
    details = bundle.target_metrics[
        bundle.target_metrics["Modèle"] == PRODUCTION_MODEL_NAME
    ].reset_index(drop=True)
    return replace(
        bundle,
        models={PRODUCTION_MODEL_NAME: bundle.models[PRODUCTION_MODEL_NAME]},
        best_model_name=PRODUCTION_MODEL_NAME,
        summary_metrics=summary,
        target_metrics=details,
        training_seconds={
            PRODUCTION_MODEL_NAME: bundle.training_seconds[
                PRODUCTION_MODEL_NAME
            ]
        },
        training_seconds_by_target={
            PRODUCTION_MODEL_NAME: bundle.training_seconds_by_target[
                PRODUCTION_MODEL_NAME
            ]
        },
    )


def save_production_bundle(
    bundle: ModelBundle,
    path: str | Path = PRODUCTION_ARTIFACT_PATH,
) -> Path:
    return save_bundle(as_production_bundle(bundle), path)


def load_production_bundle(
    path: str | Path = PRODUCTION_ARTIFACT_PATH,
) -> ModelBundle:
    bundle = load_bundle(path)
    if set(bundle.models) != {PRODUCTION_MODEL_NAME}:
        raise ValueError(
            "L'artefact de production doit contenir uniquement le sélecteur."
        )
    return bundle


def production_artifact_is_current(
    path: str | Path = PRODUCTION_ARTIFACT_PATH,
) -> bool:
    artifact = Path(path)
    if not artifact.exists():
        return False
    try:
        bundle = load_production_bundle(artifact)
        source = find_data_file()
        return (
            _bundle_matches_data(bundle, source)
            and getattr(bundle, "model_schema_version", None)
            == MODEL_SCHEMA_VERSION
            and _has_serious_optuna_budget(bundle)
        )
    except Exception:
        return False
