# Classification d'étoiles APOGEE DR17 : version 370 000 étoiles

Cette version reprend [apogee-stars-full](https://github.com/imranakki/apogee-stars-full) (100 000 étoiles) sur 370 000 étoiles : 246 572 bons spectres et 123 428 mauvais. 85 des 370 085 étoiles demandées n'ont pas de spectre disponible. Même catalogue `allStar`, mêmes étiquettes, mêmes modèles.

Le volume impose quatre changements de code, décrits dans `docs/memoire.md` :

1. Les spectres prétraités sont stockés en float16 dans `features_X.npy` : 5,6 Go au lieu de 11 Go.
2. Les lots d'entraînement sont lus ligne par ligne dans ce tableau : le jeu d'entraînement n'est jamais copié.
3. La PCA s'ajuste sur 50 000 spectres tirés au hasard, puis projette tout l'échantillon par morceaux.
4. HDBSCAN tourne sur un sous-échantillon de 50 000 étoiles.

`03_download_spectra.py --seed-from` copie les spectres déjà téléchargés par une autre version au lieu de les retélécharger. Si `data.sdss.org` ne répond plus, `STARS_SAS=https://dr17.sdss.org` bascule sur le miroir officiel.

## Résultats

Sur `test_good` (37 034 bons spectres), entraînement sur les bons spectres :

| Mesure | 30 781 | 99 773 | 370 000 |
|---|---|---|---|
| Macro F1, CNN | 0,818 | 0,834 | 0,848 |
| Macro F1, forêt | 0,696 | 0,751 | 0,776 |
| Erreur Teff, CNN (K) | 44,6 | 32,9 | 24,7 |
| Erreur log g, CNN (dex) | 0,083 | 0,069 | 0,050 |
| Rappel du red clump, CNN | 56 % | 60 % | 58 % |
| ARI, GMM, k = 3 | 0,42 | 0,42 | 0,41 |
| ARI, GMM, k = 4 | 0,09 | 0,44 | 0,10 |

- Plus de données améliore surtout la régression.
- Le red clump reste confondu avec la RGB à 42 %.
- Le GMM à 3 groupes tient. Le score de 0,44 à 4 groupes de la version 100 000 ne se reproduit pas.
- Entraîner sur tous les spectres améliore le macro F1 du CNN (0,857) mais double l'erreur sur Teff (45,1 K).
- L'expérience complète prend 2 heures sur une RTX 3060 portable.

### Choix des paramètres

Chaque valeur du projet a une raison, détaillée dans `docs/choix_parametres.md`. Des tests de sensibilité changent un paramètre à la fois. Les meilleurs réglages sont choisis sur la validation, puis mesurés une seule fois sur le test, avec 3 graines par CNN :

| Modèle | Référence | Meilleurs réglages |
|---|---|---|
| Macro F1, CNN | 0,852 ± 0,010 | 0,865 ± 0,000 |
| Macro F1, CNN, SNR < 70 | 0,642 ± 0,023 | 0,738 ± 0,022 |
| Macro F1, forêt | 0,776 | 0,803 |
| Erreur log g, forêt | 0,106 | 0,085 |

- La forêt fait mieux avec 10 composantes PCA qu'avec 50.
- Ajouter 50 % des mauvais spectres aide le CNN à classer. En régression, les mauvais spectres nuisent aux bons spectres.
- Deux runs du même CNN diffèrent d'environ 0,01 de macro F1. Un seul run ne suffit pas à comparer deux réglages.
- En régression, 30 époques sont trop courtes : la perte de validation baisse encore à 50.

Tableaux dans `results/` (tests dans `results/sensitivity/` et `results/best/`), rapport dans `report/rapport.pdf`.

Les résultats des versions 30 000 et 100 000 étoiles restent dans leurs dépôts.

Projet de cours. Le but est de regrouper des étoiles à partir de leur seul spectre infrarouge APOGEE, puis de vérifier si les groupes correspondent à des classes physiques : naines, sous-géantes, géantes rouges (RGB), red clump (RC) et populations du disque, du halo et du bulbe.

Le projet suit les trois demandes de l'équipe du cours :

1. Réduction de dimension (PCA, t-SNE, UMAP) pour voir si les classes se séparent.
2. Apprentissage supervisé avec un CNN 1-D, comparé à une forêt aléatoire.
3. Impact de la qualité des données : entraînement sur les bons spectres seulement, puis sur tous.

## Installation

```bash
pip install -r requirements.txt
```

Les données prennent environ 45 Go pour 370 000 étoiles. Choisissez un disque avec assez de place :

```bash
export STARS_DATA=/media/imranakki/Seagate/01_Projects/apogee-stars-370k-data
```

Sans `STARS_DATA`, les scripts écrivent dans `data/` à la racine du projet.

## Pipeline

```bash
python scripts/01_get_catalogs.py        # catalogues APOGEE, red clump, OCCAM (4,2 Go)
python scripts/02_select_stars.py        # échantillon, qualité, étiquettes, split
python scripts/03_download_spectra.py    # spectres, reprise automatique (--seed-from pour réutiliser)
python scripts/04_explore.py             # PCA, t-SNE, UMAP
python scripts/05_baseline.py            # forêt aléatoire
python scripts/06_train.py --model cnn   # CNN 1-D, reprise automatique
python scripts/07_cluster.py             # K-means, GMM, HDBSCAN
python scripts/08_report.py              # tableaux de résultats
python scripts/09_quality_by_snr.py      # performance par tranche de SNR
python scripts/10_label_source.py        # erreurs par origine d'étiquette
python scripts/11_sensitivity.py cluster_dims cluster_k rf_trees rf_pca size epochs snr badfrac lr   # environ 8 h
python scripts/12_best.py                # meilleurs réglages, 3 graines, environ 4 h
python report/make_sensitivity_figures.py
```

Pour lancer toute l'expérience qualité d'un coup :

```bash
bash scripts/run_quality_experiment.sh
```

Les étapes 01, 03 et 06 reprennent où elles se sont arrêtées. Relancez la même commande après une interruption.

## Organisation

```
stars/      code partagé (configuration, catalogue, spectres, modèles, entraînement, figures)
scripts/    une étape du pipeline par script, numérotés dans l'ordre d'exécution
docs/       explication de chaque script et des choix de données
results/    résultats de l'expérience complète : tableaux et figures
report/     rapport LaTeX (rapport.pdf), présentation beamer (presentation.pdf) et script des figures
```

Commencez par `docs/00_vue_ensemble.md`.
