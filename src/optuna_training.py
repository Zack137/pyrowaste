from __future__ import annotations

from dataclasses import dataclass
import json
from time import perf_counter
from typing import Callable
import warnings

from joblib import Parallel, delayed
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from xgboost import XGBRegressor

from .config import (
    OPTUNA_EXTRA_TREES_MODEL_NAME,
    OPTUNA_HYBRID_MODEL_NAME,
    OPTUNA_MLP_MODEL_NAME,
    OPTUNA_MODEL_FAMILIES,
    OPTUNA_RIDGE_MODEL_NAME,
    RANDOM_STATE,
    TARGET_COLUMNS,
)
from .features import InputFeatureEngineer


class TwoBranchMLPXGBoostRegressor(BaseEstimator, RegressorMixin):
    """Combinaison convexe d'un MLP compact et d'un XGBoost régularisé.

    La standardisation des entrées est propre à la branche MLP. La branche
    XGBoost reçoit les mêmes variables imputées sans standardisation. Le poids
    de combinaison est optimisé dans la validation croisée interne, comme les
    autres hyperparamètres : aucune observation du pli externe n'intervient.
    """

    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...] = (4,),
        activation: str = "relu",
        mlp_alpha: float = 0.001,
        batch_size: int = 32,
        learning_rate_init: float = 0.001,
        mlp_max_iter: int = 220,
        mlp_n_iter_no_change: int = 20,
        xgb_n_estimators: int = 80,
        xgb_max_depth: int = 3,
        xgb_learning_rate: float = 0.05,
        xgb_min_child_weight: float = 1.0,
        xgb_subsample: float = 0.9,
        xgb_colsample_bytree: float = 0.9,
        xgb_reg_alpha: float = 0.001,
        xgb_reg_lambda: float = 1.0,
        mlp_weight: float = 0.5,
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.mlp_alpha = mlp_alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.mlp_max_iter = mlp_max_iter
        self.mlp_n_iter_no_change = mlp_n_iter_no_change
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_min_child_weight = xgb_min_child_weight
        self.xgb_subsample = xgb_subsample
        self.xgb_colsample_bytree = xgb_colsample_bytree
        self.xgb_reg_alpha = xgb_reg_alpha
        self.xgb_reg_lambda = xgb_reg_lambda
        self.mlp_weight = mlp_weight
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "TwoBranchMLPXGBoostRegressor":
        inputs, target = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=float,
            y_numeric=True,
        )
        self.mlp_scaler_ = StandardScaler()
        mlp_inputs = self.mlp_scaler_.fit_transform(inputs)
        self.mlp_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver="adam",
            alpha=float(self.mlp_alpha),
            batch_size=int(self.batch_size),
            learning_rate_init=float(self.learning_rate_init),
            max_iter=int(self.mlp_max_iter),
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=int(self.mlp_n_iter_no_change),
            random_state=int(self.random_state),
        )
        self.xgboost_ = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(self.xgb_n_estimators),
            max_depth=int(self.xgb_max_depth),
            learning_rate=float(self.xgb_learning_rate),
            min_child_weight=float(self.xgb_min_child_weight),
            subsample=float(self.xgb_subsample),
            colsample_bytree=float(self.xgb_colsample_bytree),
            reg_alpha=float(self.xgb_reg_alpha),
            reg_lambda=float(self.xgb_reg_lambda),
            tree_method="hist",
            random_state=int(self.random_state),
            n_jobs=1,
            verbosity=0,
        )
        self.mlp_.fit(mlp_inputs, target)
        self.xgboost_.fit(inputs, target)
        self.n_features_in_ = int(inputs.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("mlp_scaler_", "mlp_", "xgboost_"))
        inputs = check_array(X, accept_sparse=False, dtype=float)
        mlp_prediction = np.asarray(
            self.mlp_.predict(self.mlp_scaler_.transform(inputs)),
            dtype=float,
        ).reshape(-1)
        xgb_prediction = np.asarray(
            self.xgboost_.predict(inputs),
            dtype=float,
        ).reshape(-1)
        weight = float(np.clip(self.mlp_weight, 0.0, 1.0))
        return weight * mlp_prediction + (1.0 - weight) * xgb_prediction


@dataclass
class IndependentTargetRegressor:
    """Huit estimateurs indépendants, rangés dans l'ordre des sorties."""

    family_name: str
    estimators: dict[str, object]

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        columns = [
            np.asarray(self.estimators[target].predict(inputs), dtype=float).reshape(-1)
            for target in TARGET_COLUMNS
        ]
        return np.column_stack(columns)


@dataclass
class AutomaticTargetSelector:
    """Assemble, pour chaque sortie, la famille choisie sur la validation."""

    family_models: dict[str, IndependentTargetRegressor]
    best_family_by_target: dict[str, str]

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        required_families = set(self.best_family_by_target.values())
        predictions = {
            family: self.family_models[family].predict(inputs)
            for family in required_families
        }
        selected = [
            predictions[self.best_family_by_target[target]][:, index]
            for index, target in enumerate(TARGET_COLUMNS)
        ]
        return np.column_stack(selected)


@dataclass
class OptunaTrainingResult:
    family_models: dict[str, IndependentTargetRegressor]
    best_params: dict[str, dict[str, dict[str, object]]]
    trials: pd.DataFrame
    training_seconds: dict[str, float]
    training_seconds_by_target: dict[str, dict[str, float]]


def _suggest_ridge_params(
    trial: optuna.Trial, *, fast: bool, nested_mode: bool
) -> dict[str, object]:
    if fast:
        return {
            "degree": trial.suggest_categorical("degree", [2]),
            "interaction_only": trial.suggest_categorical(
                "interaction_only", [False]
            ),
            "alpha": trial.suggest_categorical("alpha", [1.0]),
        }
    degree = trial.suggest_int("degree", 1, 2)
    return {
        "degree": degree,
        "interaction_only": (
            trial.suggest_categorical("interaction_only", [False, True])
            if degree == 2
            else False
        ),
        "alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True),
    }


def _suggest_extra_trees_params(
    trial: optuna.Trial, *, fast: bool, nested_mode: bool
) -> dict[str, object]:
    if fast:
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [40]),
            "max_depth": trial.suggest_categorical("max_depth", [12]),
            "min_samples_split": trial.suggest_categorical(
                "min_samples_split", [2]
            ),
            "min_samples_leaf": trial.suggest_categorical(
                "min_samples_leaf", [2]
            ),
            "max_features": trial.suggest_categorical("max_features", [0.8]),
            "bootstrap": trial.suggest_categorical("bootstrap", [False]),
        }
    if nested_mode:
        return {
            "n_estimators": trial.suggest_categorical(
                "n_estimators", [80, 120]
            ),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 6, 12]
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 8
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 4
            ),
            "max_features": trial.suggest_float(
                "max_features", 0.50, 1.0
            ),
            "bootstrap": trial.suggest_categorical(
                "bootstrap", [False, True]
            ),
        }
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators", 160, 520, step=40
        ),
        "max_depth": trial.suggest_categorical(
            "max_depth", [None, 6, 10, 16, 24]
        ),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        "max_features": trial.suggest_float("max_features", 0.45, 1.0),
        "bootstrap": trial.suggest_categorical("bootstrap", [False, True]),
    }


def _suggest_mlp_params(
    trial: optuna.Trial, *, fast: bool, nested_mode: bool
) -> dict[str, object]:
    if fast:
        return {
            "n_layers": trial.suggest_categorical("n_layers", [1]),
            "units_layer_1": trial.suggest_categorical("units_layer_1", [4]),
            "activation": trial.suggest_categorical("activation", ["relu"]),
            "alpha": trial.suggest_categorical("alpha", [0.001]),
            "learning_rate_init": trial.suggest_categorical(
                "learning_rate_init", [0.001]
            ),
            "batch_size": trial.suggest_categorical("batch_size", [32]),
        }

    # Réseau volontairement très compact : avec 15 variables après
    # feature engineering, l'architecture maximale (4, 2) ne compte que
    # 77 poids et biais par sortie. Elle reste donc compatible avec la règle
    # indicative de 5 observations par paramètre jusque dans les sous-plis
    # internes les plus petits de la validation imbriquée.
    layer_count = trial.suggest_int("n_layers", 1, 2)
    params: dict[str, object] = {
        "n_layers": layer_count,
        "activation": trial.suggest_categorical(
            "activation", ["relu", "tanh"]
        ),
        "alpha": trial.suggest_float(
            "alpha",
            1e-4 if nested_mode else 1e-5,
            1e-2 if nested_mode else 1e-1,
            log=True,
        ),
        "learning_rate_init": trial.suggest_float(
            "learning_rate_init",
            2e-4 if nested_mode else 1e-4,
            3e-3 if nested_mode else 5e-3,
            log=True,
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", [32, 64] if nested_mode else [16, 32, 64]
        ),
        "units_layer_1": trial.suggest_categorical(
            "units_layer_1", [2, 4]
        ),
    }
    if layer_count == 2:
        params["units_layer_2"] = trial.suggest_categorical(
            "units_layer_2", [2]
        )
    return params


def _suggest_hybrid_params(
    trial: optuna.Trial, *, fast: bool, nested_mode: bool
) -> dict[str, object]:
    """Espace parcimonieux pour l'hybride à deux branches.

    La branche MLP reprend exactement les limites de complexité du MLP seul.
    La branche XGBoost reste volontairement peu profonde afin de limiter la
    variance et le coût de la validation imbriquée.
    """

    params = _suggest_mlp_params(
        trial,
        fast=fast,
        nested_mode=nested_mode,
    )
    if fast:
        params.update(
            {
                "xgb_n_estimators": trial.suggest_categorical(
                    "xgb_n_estimators", [40]
                ),
                "xgb_max_depth": trial.suggest_categorical(
                    "xgb_max_depth", [2]
                ),
                "xgb_learning_rate": trial.suggest_categorical(
                    "xgb_learning_rate", [0.05]
                ),
                "xgb_min_child_weight": trial.suggest_categorical(
                    "xgb_min_child_weight", [2.0]
                ),
                "xgb_subsample": trial.suggest_categorical(
                    "xgb_subsample", [0.9]
                ),
                "xgb_colsample_bytree": trial.suggest_categorical(
                    "xgb_colsample_bytree", [0.9]
                ),
                "xgb_reg_alpha": trial.suggest_categorical(
                    "xgb_reg_alpha", [0.001]
                ),
                "xgb_reg_lambda": trial.suggest_categorical(
                    "xgb_reg_lambda", [1.0]
                ),
                "mlp_weight": trial.suggest_categorical(
                    "mlp_weight", [0.5]
                ),
            }
        )
        return params

    params.update(
        {
            "xgb_n_estimators": trial.suggest_categorical(
                "xgb_n_estimators",
                [60, 100] if nested_mode else [80, 120, 180],
            ),
            "xgb_max_depth": trial.suggest_int("xgb_max_depth", 2, 4),
            "xgb_learning_rate": trial.suggest_float(
                "xgb_learning_rate",
                0.02,
                0.12,
                log=True,
            ),
            "xgb_min_child_weight": trial.suggest_float(
                "xgb_min_child_weight",
                1.0,
                6.0,
            ),
            "xgb_subsample": trial.suggest_float(
                "xgb_subsample",
                0.70,
                1.0,
            ),
            "xgb_colsample_bytree": trial.suggest_float(
                "xgb_colsample_bytree",
                0.70,
                1.0,
            ),
            "xgb_reg_alpha": trial.suggest_float(
                "xgb_reg_alpha",
                1e-5,
                1.0,
                log=True,
            ),
            "xgb_reg_lambda": trial.suggest_float(
                "xgb_reg_lambda",
                0.1,
                10.0,
                log=True,
            ),
            "mlp_weight": trial.suggest_float(
                "mlp_weight",
                0.20,
                0.80,
            ),
        }
    )
    return params


def _hidden_layers(params: dict[str, object]) -> tuple[int, ...]:
    return tuple(
        int(params[f"units_layer_{index}"])
        for index in range(1, int(params["n_layers"]) + 1)
    )


def _build_estimator(
    family: str,
    params: dict[str, object],
    *,
    seed: int,
    fast: bool,
    nested_mode: bool,
) -> TransformedTargetRegressor:
    common_steps: list[tuple[str, object]] = [
        ("feature_engineering", InputFeatureEngineer()),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if family == OPTUNA_RIDGE_MODEL_NAME:
        regressor = Pipeline(
            common_steps
            + [
                ("scale_before_polynomial", StandardScaler()),
                (
                    "polynomial",
                    PolynomialFeatures(
                        degree=int(params["degree"]),
                        interaction_only=bool(params["interaction_only"]),
                        include_bias=False,
                    ),
                ),
                ("scale", StandardScaler()),
                (
                    "regressor",
                    Ridge(alpha=float(params["alpha"]), solver="lsqr"),
                ),
            ]
        )
    elif family == OPTUNA_EXTRA_TREES_MODEL_NAME:
        regressor = Pipeline(
            common_steps
            + [
                (
                    "regressor",
                    ExtraTreesRegressor(
                        n_estimators=int(params["n_estimators"]),
                        max_depth=params["max_depth"],
                        min_samples_split=int(params["min_samples_split"]),
                        min_samples_leaf=int(params["min_samples_leaf"]),
                        max_features=float(params["max_features"]),
                        bootstrap=bool(params["bootstrap"]),
                        random_state=seed,
                        n_jobs=1,
                    ),
                )
            ]
        )
    elif family == OPTUNA_MLP_MODEL_NAME:
        regressor = Pipeline(
            common_steps
            + [
                ("scale", StandardScaler()),
                (
                    "regressor",
                    MLPRegressor(
                        hidden_layer_sizes=_hidden_layers(params),
                        activation=str(params["activation"]),
                        solver="adam",
                        alpha=float(params["alpha"]),
                        batch_size=int(params["batch_size"]),
                        learning_rate_init=float(
                            params["learning_rate_init"]
                        ),
                        max_iter=(
                            100 if fast else 220 if nested_mode else 600
                        ),
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=(
                            15 if fast else 20 if nested_mode else 35
                        ),
                        random_state=seed,
                    ),
                ),
            ]
        )
    elif family == OPTUNA_HYBRID_MODEL_NAME:
        regressor = Pipeline(
            common_steps
            + [
                (
                    "regressor",
                    TwoBranchMLPXGBoostRegressor(
                        hidden_layer_sizes=_hidden_layers(params),
                        activation=str(params["activation"]),
                        mlp_alpha=float(params["alpha"]),
                        batch_size=int(params["batch_size"]),
                        learning_rate_init=float(
                            params["learning_rate_init"]
                        ),
                        mlp_max_iter=(
                            100 if fast else 220 if nested_mode else 600
                        ),
                        mlp_n_iter_no_change=(
                            15 if fast else 20 if nested_mode else 35
                        ),
                        xgb_n_estimators=int(params["xgb_n_estimators"]),
                        xgb_max_depth=int(params["xgb_max_depth"]),
                        xgb_learning_rate=float(
                            params["xgb_learning_rate"]
                        ),
                        xgb_min_child_weight=float(
                            params["xgb_min_child_weight"]
                        ),
                        xgb_subsample=float(params["xgb_subsample"]),
                        xgb_colsample_bytree=float(
                            params["xgb_colsample_bytree"]
                        ),
                        xgb_reg_alpha=float(params["xgb_reg_alpha"]),
                        xgb_reg_lambda=float(params["xgb_reg_lambda"]),
                        mlp_weight=float(params["mlp_weight"]),
                        random_state=seed,
                    ),
                )
            ]
        )
    else:  # pragma: no cover - garde défensif
        raise ValueError(f"Famille Optuna inconnue : {family}")

    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
        check_inverse=False,
    )


def _suggest_params(
    family: str,
    trial: optuna.Trial,
    *,
    fast: bool,
    nested_mode: bool,
) -> dict[str, object]:
    if family == OPTUNA_RIDGE_MODEL_NAME:
        return _suggest_ridge_params(
            trial, fast=fast, nested_mode=nested_mode
        )
    if family == OPTUNA_EXTRA_TREES_MODEL_NAME:
        return _suggest_extra_trees_params(
            trial, fast=fast, nested_mode=nested_mode
        )
    if family == OPTUNA_MLP_MODEL_NAME:
        return _suggest_mlp_params(
            trial, fast=fast, nested_mode=nested_mode
        )
    if family == OPTUNA_HYBRID_MODEL_NAME:
        return _suggest_hybrid_params(
            trial, fast=fast, nested_mode=nested_mode
        )
    raise ValueError(f"Famille Optuna inconnue : {family}")


def _optimize_one_target(
    family: str,
    target: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_trials: int,
    inner_splits: int,
    fast: bool,
    nested_mode: bool,
    seed: int,
) -> tuple[object, dict[str, object], list[dict[str, object]], float]:
    folds = list(
        KFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        ).split(X_train)
    )
    target_scale = max(float(y_train.std(ddof=0)), 1e-12)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(
            family,
            trial,
            fast=fast,
            nested_mode=nested_mode,
        )
        normalized_rmse: list[float] = []
        for train_index, validation_index in folds:
            estimator = _build_estimator(
                family,
                params,
                seed=seed,
                fast=fast,
                nested_mode=nested_mode,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(
                    X_train.iloc[train_index],
                    y_train.iloc[train_index],
                )
            prediction = np.asarray(
                estimator.predict(X_train.iloc[validation_index]),
                dtype=float,
            )
            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_train.iloc[validation_index],
                        prediction,
                    )
                )
            )
            normalized_rmse.append(rmse / target_scale)
        return float(np.mean(normalized_rmse))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"{family}__{target}",
    )
    started = perf_counter()
    study.optimize(
        objective,
        n_trials=max(1, int(n_trials)),
        n_jobs=1,
        show_progress_bar=False,
    )
    best_params = dict(study.best_params)
    if family == OPTUNA_RIDGE_MODEL_NAME:
        best_params.setdefault("interaction_only", False)
    estimator = _build_estimator(
        family,
        best_params,
        seed=seed,
        fast=fast,
        nested_mode=nested_mode,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(X_train, y_train)
    elapsed = perf_counter() - started

    trial_rows: list[dict[str, object]] = []
    for trial in study.trials:
        trial_rows.append(
            {
                "Famille": family,
                "Sortie": target,
                "Essai": int(trial.number + 1),
                "RMSE_CV_normalisee": (
                    float(trial.value)
                    if trial.value is not None
                    else np.nan
                ),
                "Etat": trial.state.name,
                "Hyperparametres": json.dumps(
                    trial.params,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return estimator, best_params, trial_rows, elapsed


def train_optuna_families(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    *,
    n_trials: int,
    inner_splits: int,
    fast: bool,
    nested_mode: bool = False,
    parallel_jobs: int = 1,
    progress: Callable[[str], None] | None = None,
) -> OptunaTrainingResult:
    """Optimise puis ajuste les quatre familles pour chacune des 8 sorties.

    Les 32 études famille × sortie sont indépendantes et peuvent être
    distribuées dans des processus ``loky``. Chaque étude conserve une graine
    déterministe propre et chaque estimateur de base reste mono-processus afin
    d'éviter le parallélisme imbriqué.
    """

    if inner_splits < 2:
        raise ValueError("inner_splits doit être supérieur ou égal à 2.")
    worker_count = int(parallel_jobs)
    if worker_count < 1:
        raise ValueError("parallel_jobs doit être supérieur ou égal à 1.")
    emit = progress or (lambda message: None)
    family_models: dict[str, IndependentTargetRegressor] = {}
    best_params: dict[str, dict[str, dict[str, object]]] = {
        family: {} for family in OPTUNA_MODEL_FAMILIES
    }
    all_trials: list[dict[str, object]] = []
    training_seconds: dict[str, float] = {
        family: 0.0 for family in OPTUNA_MODEL_FAMILIES
    }
    training_seconds_by_target: dict[str, dict[str, float]] = {
        family: {} for family in OPTUNA_MODEL_FAMILIES
    }
    estimators_by_family: dict[str, dict[str, object]] = {
        family: {} for family in OPTUNA_MODEL_FAMILIES
    }

    task_specs = [
        (
            family_index,
            family,
            target_index,
            target,
            RANDOM_STATE + family_index * 100 + target_index,
        )
        for family_index, family in enumerate(OPTUNA_MODEL_FAMILIES)
        for target_index, target in enumerate(TARGET_COLUMNS)
    ]

    def run_task(
        spec: tuple[int, str, int, str, int],
    ) -> tuple[object, dict[str, object], list[dict[str, object]], float]:
        _, family, _, target, seed = spec
        return _optimize_one_target(
            family,
            target,
            X_train,
            y_train[target],
            n_trials=n_trials,
            inner_splits=inner_splits,
            fast=fast,
            nested_mode=nested_mode,
            seed=seed,
        )

    if worker_count == 1:
        task_results = []
        for spec in task_specs:
            _, family, target_index, target, _ = spec
            emit(
                f"{family} — optimisation de {target} "
                f"({target_index + 1}/{len(TARGET_COLUMNS)})"
            )
            task_results.append(run_task(spec))
    else:
        emit(
            f"Lancement de {len(task_specs)} études Optuna indépendantes "
            f"sur {worker_count} processus."
        )
        task_results = Parallel(
            n_jobs=worker_count,
            backend="loky",
            inner_max_num_threads=1,
        )(delayed(run_task)(spec) for spec in task_specs)

    for spec, result in zip(task_specs, task_results):
        _, family, target_index, target, _ = spec
        estimator, params, trial_rows, elapsed = result
        estimators_by_family[family][target] = estimator
        best_params[family][target] = params
        training_seconds_by_target[family][target] = elapsed
        all_trials.extend(trial_rows)
        training_seconds[family] += elapsed
        if worker_count > 1:
            emit(
                f"{family} — optimisation terminée pour {target} "
                f"({target_index + 1}/{len(TARGET_COLUMNS)})"
            )

    for family in OPTUNA_MODEL_FAMILIES:
        family_models[family] = IndependentTargetRegressor(
            family_name=family,
            estimators=estimators_by_family[family],
        )

    return OptunaTrainingResult(
        family_models=family_models,
        best_params=best_params,
        trials=pd.DataFrame(all_trials),
        training_seconds=training_seconds,
        training_seconds_by_target=training_seconds_by_target,
    )
