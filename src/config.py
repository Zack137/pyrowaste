from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
ACTIVE_DATA_PATH = BASE_DIR / "e_waste_pyrolysis_merged_1000.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "model_bundle.joblib"
PRODUCTION_ARTIFACT_PATH = ARTIFACT_DIR / "production_selected_bundle.joblib"
IMPROVED_YIELD_ARTIFACT_PATH = ARTIFACT_DIR / "improved_yield" / "model.joblib"
OPTUNA_MLP_ALL_ARTIFACT_PATH = ARTIFACT_DIR / "optuna_mlp_8_outputs" / "model.joblib"
OPTUNA_REPORT_DIR = BASE_DIR / "reports" / "optuna_target_selection"

RANDOM_STATE = 42
CHEMICAL_PRICE_PER_KG = 2.0
MODEL_SCHEMA_VERSION = 7

# Budget central du protocole interactif. Les valeurs rapides restent
# explicitement séparées afin qu'un test léger ne puisse pas être confondu
# avec un artefact destiné à l'application.
DEFAULT_OPTUNA_TRIALS = 50
DEFAULT_OPTUNA_INNER_SPLITS = 5
FAST_OPTUNA_TRIALS = 1
FAST_OPTUNA_INNER_SPLITS = 2

OPTUNA_RIDGE_MODEL_NAME = "Ridge polynomial Optuna"
OPTUNA_EXTRA_TREES_MODEL_NAME = "Extra Trees Optuna"
OPTUNA_MLP_MODEL_NAME = "MLP Optuna"
OPTUNA_HYBRID_MODEL_NAME = "Hybride 2 branches MLP + XGBoost Optuna"
PRODUCTION_MODEL_NAME = "Sélection automatique par sortie"
OPTUNA_MODEL_FAMILIES = [
    OPTUNA_RIDGE_MODEL_NAME,
    OPTUNA_EXTRA_TREES_MODEL_NAME,
    OPTUNA_MLP_MODEL_NAME,
    OPTUNA_HYBRID_MODEL_NAME,
]

CATEGORICAL_INPUTS: list[str] = []
NUMERIC_INPUTS = [
    "Concentration_liquide_chimique_kg",
    "Temperature_C",
    "Temps_reaction_min",
    "Heating_rate_C_min",
    "Taille_particules_mm",
]
INPUT_COLUMNS = CATEGORICAL_INPUTS + NUMERIC_INPUTS

YIELD_COLUMNS = ["Solid_Yield_pct", "Liquid_Yield_pct", "Gas_Yield_pct"]
HALOGEN_COLUMNS = ["HBr_pct", "Br2_pct", "HCl_pct", "Cl2_pct", "HF_pct"]
TARGET_COLUMNS = YIELD_COLUMNS + HALOGEN_COLUMNS
DERIVED_COLUMNS = ["Toxicity_Index_pct", "Cout_liquide_USD"]
REQUIRED_COLUMNS = INPUT_COLUMNS + TARGET_COLUMNS

DISPLAY_NAMES = {
    "Concentration_liquide_chimique_kg": "Quantité de liquide chimique (kg)",
    "Temperature_C": "Température (°C)",
    "Temps_reaction_min": "Temps de réaction (min)",
    "Heating_rate_C_min": "Vitesse de chauffe (°C/min)",
    "Taille_particules_mm": "Taille des particules (mm)",
    "Thermal_Severity": "Interaction température–temps",
    "Heating_Exposure": "Interaction vitesse de chauffage–temps",
    "Temperature_Squared": "Température au carré",
    "Temperature_Heating_Interaction": "Interaction température–vitesse de chauffage",
    "Temperature_Particle_Ratio": "Rapport température–particules",
    "Log_Reaction_Time": "Logarithme du temps de réaction",
    "Concentration_Particle_Ratio": "Rapport quantité–taille des particules",
    "Thermal_Material_Load": "Interaction quantité–température",
    "Concentration_Time_Interaction": "Interaction quantité–temps",
    "Concentration_Heating_Interaction": "Interaction quantité–vitesse de chauffage",
    "Solid_Yield_pct": "Rendement solide (%)",
    "Liquid_Yield_pct": "Rendement liquide (%)",
    "Gas_Yield_pct": "Rendement gaz (%)",
    "HBr_pct": "HBr (%)",
    "Br2_pct": "Br₂ (%)",
    "HCl_pct": "HCl (%)",
    "Cl2_pct": "Cl₂ (%)",
    "HF_pct": "HF (%)",
    "Toxicity_Index_pct": "Indice de toxicité (%)",
    "Cout_liquide_USD": "Coût du liquide ($)",
    "Cout_energie_USD": "Coût énergétique estimé ($)",
    "Valuable_Yield_pct": "Rendement valorisable liquide + gaz (%)",
    "Score_final": "Score final (/100)",
}

# Les poids totalisent 1. Ils sont affichés dans l'interface afin que la
# fonction objectif ne soit jamais une « boîte noire ».
DEFAULT_WEIGHTS = {
    "Liquid_Yield_pct": 0.10,
    "Gas_Yield_pct": 0.07,
    "Valuable_Yield_pct": 0.05,
    "Toxicity_Index_pct": 0.25,
    "Concentration_liquide_chimique_kg": 0.33,
    "Cout_liquide_USD": 0.05,
    "Temps_reaction_min": 0.07,
    "Cout_energie_USD": 0.08,
}

DEFAULT_REACTOR_POWER_KW = 10.0
DEFAULT_ELECTRICITY_PRICE_USD_KWH = 0.15
AMBIENT_TEMPERATURE_C = 25.0
