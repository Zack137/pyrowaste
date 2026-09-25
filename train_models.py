from __future__ import annotations

import argparse

from src.data import find_data_file, load_dataset
from src.modeling import (
    export_training_reports,
    save_bundle,
    save_production_bundle,
    train_all_models,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîner les modèles PyroWaste")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Configuration allégée destinée aux vérifications rapides",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help=(
            "Nombre d'essais Optuna pour chaque famille et chaque sortie "
            "(50 par défaut; 1 avec --fast)"
        ),
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=None,
        help="Nombre de plis internes Optuna (5 par défaut; 2 avec --fast)",
    )
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=2,
        help=(
            "Nombre d'études famille × sortie exécutées en parallèle "
            "(2 par défaut)"
        ),
    )
    args = parser.parse_args()

    source = find_data_file()
    data = load_dataset(source)
    print(f"Base : {source.name} — {len(data)} expériences")
    bundle = train_all_models(
        data,
        fast=args.fast,
        data_path=source,
        n_trials=args.trials,
        inner_splits=args.inner_folds,
        parallel_jobs=args.parallel_jobs,
        progress=lambda message: print(message, flush=True),
    )
    destination = save_bundle(bundle)
    production_destination = save_production_bundle(bundle)
    report_destination = export_training_reports(bundle)

    print("\nComparaison :")
    print(bundle.summary_metrics.to_string(index=False))
    print("\nSélection automatique par sortie :")
    for target, model in bundle.best_model_by_target.items():
        print(f"- {target}: {model}")
    print(f"Artefact : {destination}")
    print(f"Artefact de production : {production_destination}")
    print(f"Rapports Optuna : {report_destination}")


if __name__ == "__main__":
    main()
