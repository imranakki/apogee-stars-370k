# Vue d'ensemble

## Question

Un algorithme regroupe-t-il des étoiles en classes physiques à partir de leur seul spectre, sans étiquettes ? Les paramètres du catalogue (Teff, log g, [M/H], [α/M]) servent uniquement à juger le résultat, jamais à l'entraînement non supervisé.

## Ordre des scripts

| Script | Rôle | Produit | Doc |
|---|---|---|---|
| `01_get_catalogs.py` | télécharge les catalogues | 3 fichiers FITS | [01](01_get_catalogs.md) |
| `02_select_stars.py` | choisit les étoiles, attribue qualité et étiquettes | `catalog.csv` | [02](02_select_stars.md) |
| `03_download_spectra.py` | télécharge les spectres | `flux.npy`, `err.npy`, `status.npy` | [03](03_download_spectra.md) |
| `04_explore.py` | PCA, t-SNE, UMAP | `pca.npz`, figures | [04](04_explore.md) |
| `05_baseline.py` | forêt aléatoire supervisée | `runs/rf_*/metrics.json` | [05](05_baseline.md) |
| `06_train.py` | CNN 1-D ou autoencodeur | `runs/<nom>/` avec checkpoints | [06](06_train.md) |
| `07_cluster.py` | K-means, GMM, HDBSCAN | `runs/cluster_*/`, figures | [07](07_cluster.md) |
| `08_report.py` | tableaux récapitulatifs | `summary_*.csv` | [08](08_report.md) |
| `09_quality_by_snr.py` | performance des CNN par tranche de SNR | `quality_by_snr.csv`, figures | [09](09_quality_by_snr.md) |
| `10_label_source.py` | erreurs par origine d'étiquette | `label_source.csv` | [10](10_label_source.md) |
| `11_sensitivity.py` | tests de sensibilité, un paramètre à la fois | `sensitivity/*.csv` | [choix des paramètres](choix_parametres.md) |
| `12_best.py` | meilleurs réglages sur la validation, 3 graines par CNN | `best/selection.json`, `best/final.csv` | [choix des paramètres](choix_parametres.md) |
| `run_quality_experiment.sh` | lance l'expérience qualité complète | tous les runs | [qualité](experience_qualite.md) |

Les définitions de qualité, d'étiquettes et de split sont dans [donnees.md](donnees.md). Le code partagé est décrit dans [package_stars.md](package_stars.md). Les choix qui font tenir 370 000 spectres dans 15 Go de RAM sont dans [memoire.md](memoire.md).

## Flux des données

```
allStar + RC + OCCAM ──────> 02 ──> catalog.csv ──> 03 ──> flux.npy / err.npy
                                                              │
                                     features_X.npy (prétraité, créé au premier usage)
                                                              │
                         ┌────────────┬──────────────┬────────┴─────────┐
                         04 (PCA)     05 (RF)        06 (CNN, AE)        
                         │                            │ latent.npy
                         └────── pca.npz ──> 07 <─────┘
                                              │
                                   08 rassemble tous les metrics.json
```

## Où vont les fichiers

Tout ce que les scripts écrivent va dans `STARS_DATA` (par défaut `data/`) :

```
STARS_DATA/
  allStar-dr17-synspec_rev1.fits       catalogue principal (4,0 Go)
  apogee-rc-DR17.fits                  catalogue red clump
  occam_member-DR17.fits               membres d'amas ouverts
  catalog.csv                          échantillon choisi, une ligne par étoile
  flux.npy, err.npy                    spectres bruts, même ordre que catalog.csv
  status.npy                           état du téléchargement de chaque étoile
  features_X.npy                       spectres prétraités, float16 (5,6 Go)
  features.npz                         lignes gardées, pixels gardés, fraction de pixels douteux
  pca.npz                              projection PCA
  runs/                                un dossier par modèle ou clustering
  figures/                             toutes les figures
  summary_*.csv                        tableaux finaux
```

## Reprise après interruption

| Étape | Mécanisme |
|---|---|
| 01 | fichier `.part`, la suite est demandée au serveur avec un en-tête Range |
| 03 | `status.npy` garde l'état de chaque étoile, sauvé toutes les 500 étoiles et à l'arrêt |
| 06 | `last.pt` écrit à chaque époque, avec modèle, optimiseur et historique |

Dans les trois cas : relancez la même commande.
