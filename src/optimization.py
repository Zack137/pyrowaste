from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    AMBIENT_TEMPERATURE_C,
    DEFAULT_ELECTRICITY_PRICE_USD_KWH,
    DEFAULT_REACTOR_POWER_KW,
    DEFAULT_WEIGHTS,
    HALOGEN_COLUMNS,
    INPUT_COLUMNS,
)
from .modeling import ModelBundle, predict_primary


@dataclass
class EnergyAssumptions:
    reactor_power_kw: float = DEFAULT_REACTOR_POWER_KW
    electricity_price_usd_kwh: float = DEFAULT_ELECTRICITY_PRICE_USD_KWH
    ambient_temperature_c: float = AMBIENT_TEMPERATURE_C


@dataclass
class OptimizationResult:
    recommendation: pd.DataFrame
    history: pd.DataFrame
    best_score: float


def estimate_energy_cost(
    inputs: pd.DataFrame,
    assumptions: EnergyAssumptions | None = None,
) -> pd.Series:
    config = assumptions or EnergyAssumptions()
    heating_rate = inputs["Heating_rate_C_min"].clip(lower=0.01)
    temperature_rise = (
        inputs["Temperature_C"] - config.ambient_temperature_c
    ).clip(lower=0.0)
    heating_minutes = temperature_rise / heating_rate
    operating_hours = (heating_minutes + inputs["Temps_reaction_min"]) / 60.0
    energy_kwh = config.reactor_power_kw * operating_hours
    return energy_kwh * config.electricity_price_usd_kwh


def _desirability(
    values: pd.Series,
    reference: pd.Series,
    *,
    maximize: bool,
) -> pd.Series:
    low = float(reference.min())
    high = float(reference.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return pd.Series(0.5, index=values.index, dtype=float)
    normalized = ((values - low) / (high - low)).clip(0.0, 1.0)
    return normalized if maximize else 1.0 - normalized


def score_components(
    inputs: pd.DataFrame,
    predictions: pd.DataFrame,
    reference_data: pd.DataFrame,
    assumptions: EnergyAssumptions | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    active_weights = dict(weights or DEFAULT_WEIGHTS)
    total_weight = sum(max(0.0, float(value)) for value in active_weights.values())
    if total_weight <= 0:
        raise ValueError("La somme des poids de la fonction objectif doit être positive.")
    active_weights = {
        key: max(0.0, float(value)) / total_weight
        for key, value in active_weights.items()
    }

    config = assumptions or EnergyAssumptions()
    current_energy = estimate_energy_cost(inputs, config)
    reference_energy = estimate_energy_cost(reference_data, config)
    current_valuable = predictions["Liquid_Yield_pct"] + predictions["Gas_Yield_pct"]
    reference_valuable = (
        reference_data["Liquid_Yield_pct"] + reference_data["Gas_Yield_pct"]
    )
    reference_toxicity = reference_data[HALOGEN_COLUMNS].sum(axis=1)

    components = pd.DataFrame(index=inputs.index)
    components["Liquid_Yield_pct"] = _desirability(
        predictions["Liquid_Yield_pct"], reference_data["Liquid_Yield_pct"], maximize=True
    )
    components["Gas_Yield_pct"] = _desirability(
        predictions["Gas_Yield_pct"], reference_data["Gas_Yield_pct"], maximize=True
    )
    components["Valuable_Yield_pct"] = _desirability(
        current_valuable, reference_valuable, maximize=True
    )
    components["Toxicity_Index_pct"] = _desirability(
        predictions["Toxicity_Index_pct"], reference_toxicity, maximize=False
    )
    components["Concentration_liquide_chimique_kg"] = _desirability(
        inputs["Concentration_liquide_chimique_kg"],
        reference_data["Concentration_liquide_chimique_kg"],
        maximize=False,
    )
    components["Cout_liquide_USD"] = _desirability(
        predictions["Cout_liquide_USD"], reference_data["Cout_liquide_USD"], maximize=False
    )
    components["Temps_reaction_min"] = _desirability(
        inputs["Temps_reaction_min"], reference_data["Temps_reaction_min"], maximize=False
    )
    components["Cout_energie_USD"] = _desirability(
        current_energy, reference_energy, maximize=False
    )

    weighted = pd.DataFrame(index=inputs.index)
    for key, weight in active_weights.items():
        weighted[key] = components[key] * weight
    weighted["Score_final"] = weighted.sum(axis=1) * 100.0
    weighted["Cout_energie_USD"] = current_energy
    return weighted


def optimize_recipe(
    bundle: ModelBundle,
    *,
    n_trials: int = 150,
    assumptions: EnergyAssumptions | None = None,
    weights: dict[str, float] | None = None,
    predictor: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> OptimizationResult:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    config = assumptions or EnergyAssumptions()

    def run_prediction(inputs: pd.DataFrame) -> pd.DataFrame:
        return predictor(inputs) if predictor is not None else predict_primary(bundle, inputs)

    def objective(trial: optuna.Trial) -> float:
        row: dict[str, float] = {}
        for column, (low, high) in bundle.input_bounds.items():
            row[column] = trial.suggest_float(column, low, high)
        inputs = pd.DataFrame([row], columns=INPUT_COLUMNS)
        prediction = run_prediction(inputs)
        score = score_components(
            inputs,
            prediction,
            bundle.reference_data,
            assumptions=config,
            weights=weights,
        )["Score_final"].iloc[0]
        return float(score)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=int(n_trials), n_jobs=1, show_progress_bar=False)

    parameters = dict(study.best_params)
    best_inputs = pd.DataFrame([parameters], columns=INPUT_COLUMNS)
    best_prediction = run_prediction(best_inputs)
    scored = score_components(
        best_inputs,
        best_prediction,
        bundle.reference_data,
        assumptions=config,
        weights=weights,
    )

    recommendation = pd.concat(
        [best_inputs.reset_index(drop=True), best_prediction.reset_index(drop=True)], axis=1
    )
    recommendation["Cout_energie_USD"] = estimate_energy_cost(
        best_inputs, config
    ).to_numpy()
    recommendation["Score_final"] = scored["Score_final"].to_numpy()

    history = pd.DataFrame(
        {
            "Essai": [trial.number + 1 for trial in study.trials],
            "Score": [trial.value for trial in study.trials],
        }
    )
    history["Meilleur score cumulé"] = history["Score"].cummax()
    return OptimizationResult(
        recommendation=recommendation,
        history=history,
        best_score=float(study.best_value),
    )
