# Provenance scientifique de la fusion des données gazeuses

## Objet

Ce document accompagne le fichier
`e_waste_pyrolysis_merged_1000_provenance.csv`. Il restaure séparément les
métadonnées retirées de la base active afin de préserver la traçabilité sans
introduire de fuite d'information dans les modèles.

Les cinq colonnes du fichier sidecar sont :

- `Record_ID` : identifiant stable permettant une jointure un-à-un ;
- `Mass_balance_pct` : bilan massique des trois rendements ;
- `Merge_domain_status` : statut interpolation ou extrapolation ;
- `Data_quality_flag` : qualification explicite du caractère synthétique ;
- `Merge_method` : méthode ayant servi à produire les données gazeuses
  fusionnées.

Ces colonnes sont des métadonnées d'audit. Elles ne sont utilisées ni comme
variables d'entrée, ni comme cibles, ni dans le calcul du score final ou
l'optimisation de recette.

## Source et vérifications

- Fichier original fourni :
  `C:\Users\LENOVO\Downloads\e_waste_pyrolysis_merged_1000.csv`
- SHA-256 de l'original :
  `D7644FB1ADB7C0BD5F6344BA1090F65BE707C8CDF79B3E265430F29C5FD56037`
- Base active nettoyée :
  `e_waste_pyrolysis_merged_1000.csv`
- SHA-256 de la base active :
  `55BBCC208062F6B8C7F327E225A0435EEFD698A3A0C6B968EFCFD593599D6776`
- Sidecar de provenance :
  `data_enriched/e_waste_pyrolysis_merged_1000_provenance.csv`
- SHA-256 du sidecar :
  `2B2E318DFFC4854B36870D18B7FB21F2EB3F2B5D7DB7185FAC1340E4EBF21605`
- Date de vérification : 23 juillet 2026.

Les contrôles ont confirmé :

- 1 000 lignes dans le fichier original, la base active et le sidecar ;
- 1 000 valeurs `Record_ID` uniques ;
- aucune différence ligne par ligne sur les cinq entrées et les huit cibles
  fondamentales entre l'original et la base active ;
- un bilan massique compris entre 99,99 % et 100,01 %, de moyenne
  100,00009 %.

## Répartition interpolation/extrapolation

| Statut de domaine | Drapeau de qualité | Nombre | Proportion |
|---|---|---:|---:|
| `interpolation_400_500` | `prototype_synthetic_interpolated` | 543 | 54,3 % |
| `extrapolation_outside_400_500` | `prototype_synthetic_extrapolated` | 457 | 45,7 % |
| **Total** |  | **1 000** | **100,0 %** |

La valeur de `Merge_method` est identique pour les 1 000 lignes :

> Temperature-based linear model from second database; not a row-index join

Ainsi, 543 lignes sont situées dans le domaine thermique d'interpolation
400–500 °C, tandis que 457 lignes sont à l'extérieur de ce domaine et reposent
sur une extrapolation. Ces mentions concernent l'enrichissement gazeux issu de
la seconde base. Elles ne transforment pas les lignes en nouvelles expériences
indépendantes.

## Interprétation scientifique

Les valeurs marquées `prototype_synthetic_interpolated` et
`prototype_synthetic_extrapolated` sont des estimations produites par un modèle
linéaire dépendant de la température. Elles ne doivent pas être décrites comme
des mesures instrumentales directes. Les extrapolations présentent en outre un
risque plus élevé lorsque la relation réelle n'est pas linéaire hors de
400–500 °C.

La base active conserve ces colonnes gazeuses à des fins d'analyse
exploratoire, mais le modèle prédictif principal n'utilise comme entrées que les
cinq conditions opératoires. Les huit cibles apprises restent les trois
rendements et les cinq gaz halogénés. Les colonnes de provenance demeurent
séparées pour éviter qu'un statut de fusion ou un indicateur de qualité ne
devienne artificiellement prédictif.

## Utilisation correcte du sidecar

La base active ne contient plus `Record_ID`. L'alignement initial du sidecar
repose donc sur l'ordre des 1 000 lignes, vérifié sans aucune différence sur les
cinq entrées et les huit cibles. Avant tout tri, filtrage ou rééchantillonnage,
il faudra soit réintroduire `Record_ID` comme clé technique explicitement
exclue des features, soit produire une table de correspondance stable. L'ordre
des lignes ne doit pas devenir une clé implicite dans une chaîne scientifique.

Pour une validation externe défendable, il faudra disposer d'identifiants de
publication, de campagne, de lot ou de source expérimentale. Le présent
sidecar documente la fusion et sa qualité, mais ne fournit pas à lui seul les
groupes nécessaires à une validation `GroupKFold` ou « leave-one-source-out ».
