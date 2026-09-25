from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import INPUT_COLUMNS, PRODUCTION_MODEL_NAME, TARGET_COLUMNS
from .modeling import ModelBundle


@dataclass
class ShapResult:
    target: str
    model_name: str
    importance: pd.DataFrame
    effects: pd.DataFrame


def _selected_target_estimator(
    bundle: ModelBundle,
    target: str,
) -> tuple[str, object]:
    family = bundle.best_model_by_target[target]
    if family in bundle.models:
        family_model = bundle.models[family]
    else:
        automatic = bundle.models[PRODUCTION_MODEL_NAME]
        family_model = automatic.family_models[family]
    return family, family_model.estimators[target]


def compute_shap(
    bundle: ModelBundle,
    target: str,
    *,
    max_samples: int = 60,
) -> ShapResult:
    """Explique la famille réellement sélectionnée pour la sortie demandée."""
    import shap

    if target not in TARGET_COLUMNS:
        raise ValueError(f"Sortie inconnue : {target}")
    family, estimator = _selected_target_estimator(bundle, target)
    sample = bundle.splits.X_validation.sample(
        n=min(max_samples, len(bundle.splits.X_validation)),
        random_state=42,
    )[INPUT_COLUMNS]
    background = bundle.splits.X_train.sample(
        n=min(40, len(bundle.splits.X_train)),
        random_state=42,
    )[INPUT_COLUMNS]

    def predict_target(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=INPUT_COLUMNS)
        return np.asarray(estimator.predict(frame), dtype=float).reshape(-1)

    explainer = shap.Explainer(
        predict_target,
        background.to_numpy(dtype=float),
        feature_names=INPUT_COLUMNS,
        algorithm="permutation",
    )
    explanation = explainer(
        sample.to_numpy(dtype=float),
        max_evals=2 * len(INPUT_COLUMNS) + 1,
        silent=True,
    )
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, 0]
    if values.ndim != 2:
        raise ValueError(f"Dimensions SHAP inattendues : {values.shape}")

    importance = pd.DataFrame(
        {
            "Variable": INPUT_COLUMNS,
            "Importance SHAP moyenne": np.abs(values).mean(axis=0),
        }
    ).sort_values("Importance SHAP moyenne", ascending=False)
    effects = sample.reset_index(drop=True).copy()
    for index, feature in enumerate(INPUT_COLUMNS):
        effects[f"SHAP__{feature}"] = values[:, index]
    return ShapResult(
        target=target,
        model_name=family,
        importance=importance.reset_index(drop=True),
        effects=effects,
    )
