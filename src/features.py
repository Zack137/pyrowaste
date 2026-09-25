from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import INPUT_COLUMNS


DERIVED_INPUT_COLUMNS = [
    "Thermal_Severity",
    "Heating_Exposure",
    "Temperature_Squared",
    "Temperature_Heating_Interaction",
    "Temperature_Particle_Ratio",
    "Log_Reaction_Time",
    "Concentration_Particle_Ratio",
    "Thermal_Material_Load",
    "Concentration_Time_Interaction",
    "Concentration_Heating_Interaction",
]
MODEL_INPUT_COLUMNS = INPUT_COLUMNS + DERIVED_INPUT_COLUMNS
EPSILON = 1e-8


def engineer_input_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """Calcule uniquement des variables disponibles avant l'expérience."""
    missing = sorted(set(INPUT_COLUMNS) - set(inputs.columns))
    if missing:
        raise ValueError(
            "Variables brutes absentes pour le feature engineering : "
            + ", ".join(missing)
        )

    result = inputs[INPUT_COLUMNS].copy()
    concentration = pd.to_numeric(
        result["Concentration_liquide_chimique_kg"], errors="coerce"
    )
    temperature = pd.to_numeric(result["Temperature_C"], errors="coerce")
    reaction_time = pd.to_numeric(result["Temps_reaction_min"], errors="coerce")
    heating_rate = pd.to_numeric(result["Heating_rate_C_min"], errors="coerce")
    particle_size = pd.to_numeric(
        result["Taille_particules_mm"], errors="coerce"
    )
    safe_particle_size = particle_size.clip(lower=EPSILON)

    result["Thermal_Severity"] = temperature * reaction_time
    result["Heating_Exposure"] = heating_rate * reaction_time
    result["Temperature_Squared"] = temperature**2
    result["Temperature_Heating_Interaction"] = temperature * heating_rate
    result["Temperature_Particle_Ratio"] = temperature / safe_particle_size
    result["Log_Reaction_Time"] = np.log1p(reaction_time.clip(lower=0.0))
    result["Concentration_Particle_Ratio"] = (
        concentration / safe_particle_size
    )
    result["Thermal_Material_Load"] = concentration * temperature
    result["Concentration_Time_Interaction"] = concentration * reaction_time
    result["Concentration_Heating_Interaction"] = concentration * heating_rate
    return result.replace([np.inf, -np.inf], np.nan)


class InputFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformateur scikit-learn réutilisé à l'entraînement et à l'inférence."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "InputFeatureEngineer":
        engineer_input_features(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return engineer_input_features(X)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(MODEL_INPUT_COLUMNS, dtype=object)
