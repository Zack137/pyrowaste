from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import (
    OPTUNA_EXTRA_TREES_MODEL_NAME,
    OPTUNA_HYBRID_MODEL_NAME,
    OPTUNA_MLP_MODEL_NAME,
    OPTUNA_MODEL_FAMILIES,
    OPTUNA_RIDGE_MODEL_NAME,
    PRODUCTION_MODEL_NAME,
)
from .features import engineer_input_features
from .modeling import ModelBundle, predict_primary


STRATEGY_AUTO = PRODUCTION_MODEL_NAME
STRATEGY_RIDGE_OPTUNA = OPTUNA_RIDGE_MODEL_NAME
STRATEGY_EXTRA_TREES_OPTUNA = OPTUNA_EXTRA_TREES_MODEL_NAME
STRATEGY_MLP_OPTUNA = OPTUNA_MLP_MODEL_NAME
STRATEGY_HYBRID_OPTUNA = OPTUNA_HYBRID_MODEL_NAME

MODEL_STRATEGIES = list(OPTUNA_MODEL_FAMILIES)
ALL_STRATEGIES = [STRATEGY_AUTO] + MODEL_STRATEGIES
RECOMMENDED_STRATEGY = STRATEGY_AUTO

STRATEGY_DESCRIPTIONS = {
    STRATEGY_AUTO: (
        "Assemble le meilleur Ridge polynomial, Extra Trees, MLP compact ou "
        "hybride MLP + XGBoost pour "
        "chacune des huit sorties. Le choix exploratoire est figé d'après le "
        "RMSE du jeu de validation interne."
    ),
    STRATEGY_RIDGE_OPTUNA: (
        "Huit régressions Ridge polynomiales indépendantes. Le degré, les "
        "interactions et la régularisation sont optimisés par Optuna."
    ),
    STRATEGY_EXTRA_TREES_OPTUNA: (
        "Huit modèles Extra Trees indépendants. Le nombre d'arbres, la "
        "profondeur et les paramètres de régularisation sont optimisés par Optuna."
    ),
    STRATEGY_MLP_OPTUNA: (
        "Huit réseaux MLP compacts indépendants, limités à 77 paramètres par "
        "cible. L'architecture, la régularisation, le taux d'apprentissage et "
        "la taille de lot sont optimisés par Optuna."
    ),
    STRATEGY_HYBRID_OPTUNA: (
        "Huit couples indépendants à deux branches : un MLP compact et un "
        "XGBoost peu profond. Optuna règle les deux branches et leur poids de "
        "combinaison à l'intérieur des plis d'entraînement."
    ),
}


@dataclass(frozen=True)
class StrategyAssets:
    """Compatibilité d'API : les modèles sont désormais dans ModelBundle."""

    source: str = "model_bundle"


def load_strategy_assets(
    directory: str | Path | None = None,
) -> StrategyAssets:
    del directory
    return StrategyAssets()


def engineer_yield_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """Alias conservé pour les anciens scripts d'analyse des variables."""
    return engineer_input_features(inputs)


def validate_strategies(bundle: ModelBundle) -> None:
    missing = sorted(set(ALL_STRATEGIES) - set(bundle.models))
    if missing:
        raise ValueError(
            "Modèles Optuna absents de l'artefact : " + ", ".join(missing)
        )


def predict_strategy(
    bundle: ModelBundle,
    assets: StrategyAssets,
    inputs: pd.DataFrame,
    strategy: str,
) -> pd.DataFrame:
    del assets
    if strategy not in ALL_STRATEGIES:
        raise ValueError(f"Stratégie inconnue : {strategy}")
    validate_strategies(bundle)
    return predict_primary(bundle, inputs, model_name=strategy)


def evaluate_strategies(
    bundle: ModelBundle,
    assets: StrategyAssets,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del assets
    validate_strategies(bundle)
    active_names = set(ALL_STRATEGIES) | {"Baseline moyenne (référence)"}
    summaries = bundle.summary_metrics[
        bundle.summary_metrics["Modèle"].isin(active_names)
    ].rename(columns={"Modèle": "Stratégie"})
    details = bundle.target_metrics[
        bundle.target_metrics["Modèle"].isin(active_names)
    ].rename(columns={"Modèle": "Stratégie"})
    return summaries.reset_index(drop=True), details.reset_index(drop=True)
