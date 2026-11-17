# Classification d'étoiles APOGEE DR17 : version 370 000 étoiles

Cette version reprend [apogee-stars-full](https://github.com/imranakki/apogee-stars-full) (100 000 étoiles) sur 370 085 étoiles : 246 572 bons spectres et 123 513 mauvais. Même catalogue `allStar`, mêmes étiquettes, mêmes modèles.

Le volume impose quatre changements de code, décrits dans `docs/memoire.md` :

1. Les spectres prétraités sont stockés en float16 dans `features_X.npy` : 5,6 Go au lieu de 11 Go.
2. Les lots d'entraînement sont lus ligne par ligne dans ce tableau : le jeu d'entraînement n'est jamais copié.
3. La PCA s'ajuste sur 50 000 spectres tirés au hasard, puis projette tout l'échantillon par morceaux.
4. HDBSCAN tourne sur un sous-échantillon de 50 000 étoiles.

`03_download_spectra.py --seed-from` copie les spectres déjà téléchargés par une autre version au lieu de les retélécharger.

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

Les données prennent environ 40 Go pour 370 000 étoiles. Choisissez un disque avec assez de place :

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
