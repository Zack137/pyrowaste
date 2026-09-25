# PyroWaste

## Démarrage de cette copie GitHub

Utiliser Python 3.11, puis :

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Les modèles déjà entraînés sont inclus dans `artifacts/` : il n'est pas
nécessaire de lancer `train_models.py` pour utiliser l'application. Leur
compatibilité avec les données est vérifiée par SHA-256, indépendamment
des dates de fichiers modifiées par Git. Les versions des dépendances
correspondent à l'environnement local vérifié.

Pour un futur déploiement Streamlit, sélectionner la branche `main`,
le fichier `app.py` et Python 3.11. Le dépôt GitHub seul ne publie pas
l'application sur Internet.

Les résultats utilisés par l'interface sont inclus. Les documents Word/PDF
personnels, les journaux et les checkpoints intermédiaires restent locaux.

## Présentation

PyroWaste est un **prototype scientifique exploratoire** de régression
supervisée multi-sorties pour la pyrolyse des déchets électroniques. Il aide à
comparer des modèles et à préparer de nouvelles expériences ; il ne constitue
pas un système de décision industrielle autonome.

## Architecture active

L’application contient quatre familles entraînées et optimisées
séparément :

1. **Ridge polynomial Optuna** ;
2. **Extra Trees Optuna** ;
3. **MLP Optuna compact** ;
4. **Hybride 2 branches MLP + XGBoost Optuna**.

Chaque famille entraîne huit régressions indépendantes, une pour chaque cible
continue. La **sélection automatique par sortie** choisit ensuite, dans chaque
pli d’entraînement, la famille ayant la plus faible RMSE interne pour la cible
concernée.

Dans la quatrième famille, une branche MLP compacte et une branche XGBoost
produisent deux estimations complémentaires, combinées par une pondération
réglée par Optuna. Il ne s’agit pas du retour de l’ancien modèle XGBoost
autonome.

**Random Forest, l’ancien XGBoost autonome, le MLP sans Optuna et les anciens
artefacts hybrides restent retirés de l’application active. XGBoost n’est
autorisé que comme branche interne du nouvel hybride optimisé par Optuna.**

## Pipeline

```text
Base de données pyrolyse
        │
        ▼
1. Prétraitement
   doublons · valeurs manquantes · unités · entrées/sorties
        │
        ▼
2. Feature engineering sans cible
        │
        ▼
3. Construction de l’artefact : 80 % train / 10 % validation / 10 % test
        │
        ├──────── Ridge polynomial + Optuna ──────────────┐
        ├──────── Extra Trees + Optuna ───────────────────┤
        ├──────── MLP compact + Optuna ───────────────────┤
        └──────── Hybride MLP + XGBoost + Optuna ─────────┘
                                                          │
                                                          ▼
4. Évaluation séparée : R² · RMSE · MAE · temps · erreur et IC95 par sortie
                                                          │
                                                          ▼
5. Meilleure famille pour chaque cible
                                                          │
                                                          ▼
6. Post-traitement : valeurs ≥ 0 · rendements 0–100 % · bilan à 100 %
                                                          │
                                                          ▼
                                                  Prédictions finales
```

Le découpage 80/10/10 sert uniquement à construire l’artefact interactif.
L’évaluation scientifique principale utilise une validation croisée répétée et
imbriquée distincte.

La construction canonique de l’artefact applique **50 essais Optuna et 5 plis
internes** à chacune des 32 études famille × cible. Deux études indépendantes
sont exécutées en parallèle ; cette parallélisation ne mélange ni les cibles ni
les plis.

## Données et sorties

La base active est `e_waste_pyrolysis_merged_1000.csv` :

- 1 000 observations ;
- 5 entrées opératoires ;
- 8 sorties apprises ;
- aucune cible utilisée comme entrée.

Entrées :

- `Concentration_liquide_chimique_kg` ;
- `Temperature_C` ;
- `Temps_reaction_min` ;
- `Heating_rate_C_min` ;
- `Taille_particules_mm`.

Sorties :

- `Solid_Yield_pct`, `Liquid_Yield_pct`, `Gas_Yield_pct` ;
- `HBr_pct`, `Br2_pct`, `HCl_pct`, `Cl2_pct`, `HF_pct`.

## Feature engineering

Les cinq entrées produisent dix variables déterministes :

| Variable | Calcul |
|---|---|
| `Thermal_Severity` | température × temps |
| `Heating_Exposure` | interaction vitesse de chauffage–temps (`v × t`) |
| `Temperature_Squared` | température² |
| `Temperature_Heating_Interaction` | température × vitesse de chauffage |
| `Temperature_Particle_Ratio` | température / (taille + ε) |
| `Log_Reaction_Time` | ln(1 + temps) |
| `Concentration_Particle_Ratio` | concentration / (taille + ε) |
| `Thermal_Material_Load` | concentration × température |
| `Concentration_Time_Interaction` | concentration × temps |
| `Concentration_Heating_Interaction` | concentration × vitesse de chauffage |

Avec ε = 10⁻⁸, le modèle reçoit 15 caractéristiques. Les ratios ou indicateurs
calculés à partir des rendements et des gaz sont réservés au post-traitement et
à l’analyse : ils ne peuvent pas entrer dans les modèles qui prédisent ces mêmes
cibles.

## MLP réduit et hybride à deux branches

Optuna ne peut choisir que les architectures `(2)`, `(4)`, `(2, 2)` et
`(4, 2)`. Avec 15 caractéristiques et une sortie par réseau, la configuration
maximale possède 77 poids et biais. Cette réduction rend la règle indicative
de cinq observations par paramètre compatible avec les plis internes du
protocole exécuté.

Cette contrainte s’applique au MLP autonome et à la branche neuronale de
l’hybride. La seconde branche de l’hybride est un XGBoost régularisé. Optuna
ajuste les deux branches et leur pondération uniquement dans les plis internes ;
aucune prédiction du pli externe n’intervient dans ce choix.

## Validation scientifique obligatoire

Le script `experiments/run_nested_repeated_optuna_validation.py` applique :

- les mêmes plis externes aux quatre familles, à la sélection automatique et au
  prédicteur naïf ;
- 5 plis externes × 5 répétitions, soit 25 évaluations externes ;
- 3 plis internes réservés à Optuna ;
- 20 essais par famille, par sortie et par pli externe, soit 800 études et
  16 000 essais Optuna au total ;
- 2 études indépendantes exécutées en parallèle ;
- un checkpoint atomique après chaque pli externe terminé et une reprise
  automatique uniquement lorsque la base, le protocole et l’implémentation
  concordent ;
- une moyenne naïve calculée uniquement sur le train externe ;
- des prédictions hors pli brutes, avant correction physique ;
- MAE, RMSE et R² globaux et par sortie ;
- moyenne, écart-type et intervalle t nominal à 95 %, avec la répétition OOF
  complète comme unité d’échantillonnage ;
- fréquence de sélection de chaque famille pour chaque cible ;
- contrôle de la complexité des branches MLP et des recouvrements
  train/validation ;

Les cinq répétitions rendent l’estimation de la variabilité des partitions plus
informative, sans rendre ces répétitions indépendantes de la base. Les
intervalles t sont donc **nominaux et conditionnels** aux partitions observées.
Ils ne remplacent ni un `GroupKFold` par étude ou lot, ni une campagne externe
indépendante.

## Nouvelles preuves

`experiments/generate_optuna_scientific_evidence.py` produit :

- comparaison globale et par sortie avec IC95 ;
- graphiques observé–prédit pour les huit sorties ;
- distributions des résidus ;
- résidus selon température et concentration ;
- analyse de sensibilité un facteur à la fois ;
- scénarios Latin hypercube et front de Pareto exploratoire ;
- CSV sources et manifeste de traçabilité.

La sensibilité et le Pareto décrivent le comportement du modèle dans le domaine
observé. Ils ne démontrent pas une relation causale.

## Post-traitement

Après l’assemblage des huit prédictions :

- les valeurs négatives sont ramenées à zéro ;
- les rendements sont limités entre 0 et 100 % ;
- `Solid + Liquid + Gas` est renormalisé à 100 %.

Les métriques scientifiques restent calculées sur les sorties brutes afin que
ces corrections ne masquent pas l’erreur du modèle.

## Exécution

```powershell
python -m pip install -r requirements.txt
python train_models.py --trials 50 --inner-folds 5 --parallel-jobs 4
python experiments/run_nested_repeated_optuna_validation.py --outer-splits 5 --repeats 5 --trials 20 --inner-splits 3 --parallel-jobs 4 --output-dir reports/nested_repeated_validation_budget_renforce
python experiments/generate_optuna_scientific_evidence.py --skip-learning-curve
python -m streamlit run app.py --server.address 127.0.0.1
```

La seconde commande écrit un checkpoint après chaque pli externe. Relancer
exactement la même commande reprend automatiquement les plis compatibles déjà
terminés.

Vérification rapide :

```powershell
python train_models.py --fast
python experiments/run_nested_repeated_optuna_validation.py --quick --output-dir reports/nested_repeated_validation_quick
python -m pytest -q
```

## Résultats canoniques

- `reports/nested_repeated_validation_budget_renforce/` — métriques et intervalles t nominaux du protocole actif ;
- `reports/optuna_scientific_evidence/` — nouveaux calculs et figures ;
- `reports/RAPPORT_FINAL_REMARQUES_PROFESSEUR.docx` — rapport scientifique final ;
- `reports/optuna_target_selection/` — contrôle interne 80/10/10 ;
- `artifacts/production_selected_bundle.joblib` — artefact interactif compact.

Les anciens fichiers éventuellement conservés dans les sous-dossiers
d’archives ne sont ni chargés par l’application ni utilisés pour la conclusion
scientifique actuelle.
