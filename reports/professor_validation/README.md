# Résultats scientifiques canoniques

Les fichiers à utiliser dans le rapport et dans l'application sont :

- `statistical_*` : références statistiques, 5 plis × 5 répétitions ;
- `heavy_*` : Random Forest, XGBoost, MLP profond/Optuna et hybrides,
  5 plis × 5 répétitions ;
- `ridge_oof_predictions.csv` : prédictions hors pli de Ridge ;
- `ridge_learning_curve_*` : courbe d'apprentissage ;
- `ridge_feature_ablation_*` : comparaison 5 entrées contre 5 + 10 ;
- `nested_optuna_*` : Optuna imbriqué, 5 plis externes × 3 internes ;
- `composition_yields_*` : régression compositionnelle des rendements ;
- `augmentation_*` : expérience exploratoire sur un découpage fixe, à ne pas
  présenter comme une validation croisée répétée.

Les dossiers `archive_smoke_2x1` et
`archive_legacy_5fold_single_repeat` conservent seulement des essais
techniques antérieurs. Ils ne doivent pas être utilisés pour conclure sur les
performances.

Les deux métadonnées canoniques contiennent le SHA-256 de la base active :
`55bbcc208062f6b8c7f327e225a0435eefd698a3a0c6b968efcfd593599d6776`.
