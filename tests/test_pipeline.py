from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.modeling as modeling
import src.optuna_training as optuna_training
from src.config import (
    CHEMICAL_PRICE_PER_KG,
    DEFAULT_OPTUNA_INNER_SPLITS,
    DEFAULT_OPTUNA_TRIALS,
    FAST_OPTUNA_INNER_SPLITS,
    FAST_OPTUNA_TRIALS,
    HALOGEN_COLUMNS,
    INPUT_COLUMNS,
    OPTUNA_HYBRID_MODEL_NAME,
    OPTUNA_MODEL_FAMILIES,
    PRODUCTION_MODEL_NAME,
    RANDOM_STATE,
    TARGET_COLUMNS,
)
from src.data import load_dataset, split_dataset
from src.features import DERIVED_INPUT_COLUMNS, engineer_input_features
from src.modeling import (
    as_production_bundle,
    dense_parameter_count,
    predict_primary,
    train_all_models,
)
from src.optimization import EnergyAssumptions, estimate_energy_cost, score_components
from src.reporting import build_optimization_pdf
from src.strategies import (
    ALL_STRATEGIES,
    MODEL_STRATEGIES,
    load_strategy_assets,
    predict_strategy,
)


def test_dataset_and_business_identities():
    df = load_dataset()
    assert len(df) == 1000
    assert not df[INPUT_COLUMNS].isna().any().any()
    assert np.allclose(df["Toxicity_Index_pct"], df[HALOGEN_COLUMNS].sum(axis=1))
    assert np.allclose(
        df["Cout_liquide_USD"],
        df["Concentration_liquide_chimique_kg"] * CHEMICAL_PRICE_PER_KG,
    )


def test_loaded_artifact_refreshes_non_learned_business_columns(monkeypatch):
    stale = pd.DataFrame(
        {
            "Concentration_liquide_chimique_kg": [10.0, 25.0],
            "HBr_pct": [1.0, 2.0],
            "Br2_pct": [0.1, 0.2],
            "HCl_pct": [0.3, 0.4],
            "Cl2_pct": [0.05, 0.06],
            "HF_pct": [0.01, 0.02],
            "Cout_liquide_USD": [200.0, 500.0],
            "Toxicity_Index_pct": [999.0, 999.0],
        }
    )
    serialized = SimpleNamespace(reference_data=stale)
    monkeypatch.setattr(modeling.joblib, "load", lambda _path: serialized)

    loaded = modeling.load_bundle("ancien_artefact.joblib")

    assert np.allclose(
        loaded.reference_data["Cout_liquide_USD"],
        stale["Concentration_liquide_chimique_kg"] * CHEMICAL_PRICE_PER_KG,
    )
    assert np.allclose(
        loaded.reference_data["Toxicity_Index_pct"],
        stale[HALOGEN_COLUMNS].sum(axis=1),
    )


def test_split_is_exactly_80_10_10():
    splits = split_dataset(load_dataset())
    assert len(splits.X_train) == 800
    assert len(splits.X_validation) == 100
    assert len(splits.X_test) == 100


def test_feature_engineering_uses_inputs_only_and_is_finite():
    df = load_dataset()
    engineered = engineer_input_features(df.iloc[:20])
    assert len(engineered.columns) == 15
    assert set(DERIVED_INPUT_COLUMNS).issubset(engineered.columns)
    assert not any(column.endswith("_Yield_pct") for column in engineered.columns)
    assert np.isfinite(engineered.to_numpy(dtype=float)).all()
    assert np.allclose(
        engineered["Thermal_Severity"],
        df.iloc[:20]["Temperature_C"] * df.iloc[:20]["Temps_reaction_min"],
    )


def test_serious_optuna_budget_is_the_interactive_default(monkeypatch):
    captured: dict[str, object] = {}

    class TrainingIntercept(Exception):
        pass

    def intercept_training(*args, **kwargs):
        captured.update(kwargs)
        raise TrainingIntercept

    monkeypatch.setattr(
        modeling,
        "train_optuna_families",
        intercept_training,
    )
    with pytest.raises(TrainingIntercept):
        train_all_models(load_dataset())

    assert DEFAULT_OPTUNA_TRIALS == 50
    assert DEFAULT_OPTUNA_INNER_SPLITS == 5
    assert FAST_OPTUNA_TRIALS == 1
    assert FAST_OPTUNA_INNER_SPLITS == 2
    assert captured["n_trials"] == DEFAULT_OPTUNA_TRIALS
    assert captured["inner_splits"] == DEFAULT_OPTUNA_INNER_SPLITS
    assert captured["parallel_jobs"] == 1

    serious = SimpleNamespace(
        optuna_protocol={
            "trials_per_family_and_target": DEFAULT_OPTUNA_TRIALS,
            "inner_splits": DEFAULT_OPTUNA_INNER_SPLITS,
        }
    )
    under_budget = SimpleNamespace(
        optuna_protocol={
            "trials_per_family_and_target": DEFAULT_OPTUNA_TRIALS - 1,
            "inner_splits": DEFAULT_OPTUNA_INNER_SPLITS,
        }
    )
    assert modeling._has_serious_optuna_budget(serious)
    assert not modeling._has_serious_optuna_budget(under_budget)


def test_optuna_parallel_dispatch_is_process_based_and_deterministic(
    monkeypatch,
):
    parallel_configuration: dict[str, object] = {}
    calls: list[tuple[str, str, int]] = []

    class RecordingParallel:
        def __init__(self, **kwargs):
            parallel_configuration.update(kwargs)

        def __call__(self, tasks):
            materialized = list(tasks)
            parallel_configuration["task_count"] = len(materialized)
            return [
                function(*args, **kwargs)
                for function, args, kwargs in materialized
            ]

    def fake_optimize(
        family,
        target,
        X_train,
        y_train,
        *,
        n_trials,
        inner_splits,
        fast,
        nested_mode,
        seed,
    ):
        assert y_train.name == target
        assert n_trials == 7
        assert inner_splits == 3
        calls.append((family, target, seed))
        return object(), {"seed": seed}, [], 0.25

    monkeypatch.setattr(optuna_training, "Parallel", RecordingParallel)
    monkeypatch.setattr(
        optuna_training,
        "_optimize_one_target",
        fake_optimize,
    )
    data = load_dataset().iloc[:20]
    trained = optuna_training.train_optuna_families(
        data[INPUT_COLUMNS],
        data[TARGET_COLUMNS],
        n_trials=7,
        inner_splits=3,
        fast=False,
        parallel_jobs=3,
    )

    expected = [
        (
            family,
            target,
            RANDOM_STATE + family_index * 100 + target_index,
        )
        for family_index, family in enumerate(OPTUNA_MODEL_FAMILIES)
        for target_index, target in enumerate(TARGET_COLUMNS)
    ]
    assert calls == expected
    assert parallel_configuration == {
        "n_jobs": 3,
        "backend": "loky",
        "inner_max_num_threads": 1,
        "task_count": 32,
    }
    assert list(trained.family_models) == OPTUNA_MODEL_FAMILIES
    assert all(
        list(trained.family_models[family].estimators) == TARGET_COLUMNS
        for family in OPTUNA_MODEL_FAMILIES
    )

    with pytest.raises(ValueError, match="parallel_jobs"):
        optuna_training.train_optuna_families(
            data[INPUT_COLUMNS],
            data[TARGET_COLUMNS],
            n_trials=1,
            inner_splits=2,
            fast=True,
            parallel_jobs=0,
        )


def test_energy_formula_is_transparent():
    inputs = pd.DataFrame(
        [
            {
                "Concentration_liquide_chimique_kg": 50.0,
                "Temperature_C": 425.0,
                "Temps_reaction_min": 20.0,
                "Heating_rate_C_min": 20.0,
                "Taille_particules_mm": 2.0,
            }
        ]
    )
    # Chauffe : (425 - 25) / 20 = 20 min, réaction : 20 min.
    # Énergie : 10 kW * 40/60 h = 6.6667 kWh, à 0.15 $/kWh = 1 $.
    cost = estimate_energy_cost(inputs, EnergyAssumptions()).iloc[0]
    assert np.isclose(cost, 1.0)


def test_fast_training_prediction_and_score():
    df = load_dataset()
    bundle = train_all_models(df, fast=True)
    assert (
        bundle.optuna_protocol["trials_per_family_and_target"]
        == FAST_OPTUNA_TRIALS
    )
    assert bundle.optuna_protocol["inner_splits"] == FAST_OPTUNA_INNER_SPLITS
    assert bundle.optuna_protocol["parallel_jobs"] == 1
    assert len(OPTUNA_MODEL_FAMILIES) == 4
    assert MODEL_STRATEGIES == OPTUNA_MODEL_FAMILIES
    assert set(OPTUNA_MODEL_FAMILIES).issubset(bundle.models)
    assert not any(
        forbidden in bundle.models
        for forbidden in (
            "Random Forest",
            "XGBoost",
            "MLP compact",
            "MLP Deep Learning",
        )
    )
    assert set(bundle.best_model_by_target) == set(TARGET_COLUMNS)
    assert set(bundle.best_model_by_target.values()).issubset(
        set(OPTUNA_MODEL_FAMILIES)
    )
    for family in OPTUNA_MODEL_FAMILIES:
        assert set(bundle.models[family].estimators) == set(TARGET_COLUMNS)
        assert set(bundle.optuna_best_params[family]) == set(TARGET_COLUMNS)
    for params in bundle.optuna_best_params["MLP Optuna"].values():
        hidden_layers = tuple(
            int(params[f"units_layer_{index}"])
            for index in range(1, int(params["n_layers"]) + 1)
        )
        assert dense_parameter_count(15, hidden_layers, 1) <= 77
    for params in bundle.optuna_best_params[OPTUNA_HYBRID_MODEL_NAME].values():
        hidden_layers = tuple(
            int(params[f"units_layer_{index}"])
            for index in range(1, int(params["n_layers"]) + 1)
        )
        assert dense_parameter_count(15, hidden_layers, 1) <= 77
        assert 0.0 <= float(params["mlp_weight"]) <= 1.0

    production_bundle = as_production_bundle(bundle)
    assert list(production_bundle.models) == [PRODUCTION_MODEL_NAME]
    assert production_bundle.best_model_name == PRODUCTION_MODEL_NAME
    inputs = df[INPUT_COLUMNS].iloc[[0]].copy()
    prediction = predict_primary(production_bundle, inputs)
    assert prediction.shape == (1, 10)
    assert np.isclose(
        prediction[["Solid_Yield_pct", "Liquid_Yield_pct", "Gas_Yield_pct"]]
        .sum(axis=1)
        .iloc[0],
        100.0,
    )
    assert (prediction[HALOGEN_COLUMNS] >= 0).all().all()
    score = score_components(inputs, prediction, df)["Score_final"].iloc[0]
    assert 0.0 <= score <= 100.0

    strategy_assets = load_strategy_assets()
    for strategy in ALL_STRATEGIES:
        strategy_prediction = predict_strategy(
            bundle, strategy_assets, inputs, strategy
        )
        assert strategy_prediction.shape == (1, 10)
        assert np.isclose(
            strategy_prediction[
                ["Solid_Yield_pct", "Liquid_Yield_pct", "Gas_Yield_pct"]
            ].sum(axis=1).iloc[0],
            100.0,
        )
        assert (strategy_prediction[HALOGEN_COLUMNS] >= 0).all().all()

    validation_details = bundle.target_metrics[
        bundle.target_metrics["Jeu"] == "Validation"
    ]
    for target, selected_family in bundle.best_model_by_target.items():
        family_rows = validation_details[
            (validation_details["Sortie"] == target)
            & validation_details["Modèle"].isin(OPTUNA_MODEL_FAMILIES)
        ]
        expected = family_rows.sort_values(
            ["RMSE", "MAE", "Modèle"], kind="stable"
        ).iloc[0]["Modèle"]
        assert selected_family == expected


def test_automatic_pdf_report():
    before = {
        "Concentration_liquide_chimique_kg": 32.0,
        "Temperature_C": 440.0,
        "Temps_reaction_min": 15.0,
        "Heating_rate_C_min": 20.0,
        "Taille_particules_mm": 2.0,
        "Solid_Yield_pct": 25.0,
        "Liquid_Yield_pct": 55.0,
        "Gas_Yield_pct": 20.0,
        "HBr_pct": 1.5,
        "Br2_pct": 0.1,
        "HCl_pct": 0.2,
        "Cl2_pct": 0.05,
        "HF_pct": 0.01,
        "Toxicity_Index_pct": 1.86,
        "Cout_liquide_USD": 64.0,
        "Cout_energie_USD": 0.8,
        "Score_final": 60.0,
    }
    after = dict(before)
    after.update(
        {
            "Concentration_liquide_chimique_kg": 16.0,
            "Toxicity_Index_pct": 0.6,
            "Cout_liquide_USD": 32.0,
            "Cout_energie_USD": 0.6,
            "Score_final": 85.0,
        }
    )
    pdf = build_optimization_pdf(
        strategy=PRODUCTION_MODEL_NAME,
        before=before,
        after=after,
        trials=50,
    )
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 5_000
