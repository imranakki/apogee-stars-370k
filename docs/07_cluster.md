# 07_cluster.py

## Rôle

Cœur non supervisé du projet. Le script regroupe les étoiles sans étiquettes, puis compare les groupes aux étiquettes du catalogue.

## Entrées possibles

| `--features` | Source | Dimensions |
|---|---|---|
| `pca` | `pca.npz` de l'étape 04 | 10 premières composantes (`--n-dims`) |
| `latent` | `runs/<run>/latent.npy` de l'étape 06 | taille du latent (16 par défaut) |

Les entrées sont standardisées avant le clustering.

## Étapes

1. Balayage de k de 2 à 12 avec K-means. Pour chaque k :
   - silhouette (sur 5000 étoiles) : séparation des groupes ;
   - stabilité : K-means lancé avec 10 graines, ARI moyen entre toutes les paires de résultats ;
   - BIC d'un mélange gaussien (GMM).
2. Choix de k : meilleure silhouette, ou valeur donnée par `--k`.
3. Clustering final avec K-means, GMM et HDBSCAN.
4. Validation a posteriori de chaque méthode.

La stabilité répond à la limite de l'étude de référence : K-means sur APOGEE dépendait de l'initialisation. Une stabilité proche de 1 signifie que toutes les graines donnent les mêmes groupes.

HDBSCAN choisit lui-même le nombre de groupes et marque les points isolés comme bruit (étiquette -1).

## Métriques de validation

| Métrique | Sens |
|---|---|
| `ari_class`, `nmi_class` | accord avec les classes dwarf / subgiant / rgb / rc |
| `ari_population`, `nmi_population` | accord avec les populations galactiques |
| `occam_purity` | pour chaque amas ouvert d'au moins 5 étoiles, part des membres dans le même groupe, moyennée |
| `noise_frac` | part des étoiles classées bruit (HDBSCAN) |

ARI vaut 0 pour un clustering au hasard et 1 pour un accord parfait. NMI va de 0 à 1.

## Commande

```bash
python scripts/07_cluster.py
python scripts/07_cluster.py --features pca --subset all --k 6
python scripts/07_cluster.py --features latent --ae-run autoencoder_none_good
```

| Option | Défaut | Effet |
|---|---|---|
| `--features` | pca | `pca` ou `latent` |
| `--ae-run` | autoencoder_none_good | run d'autoencodeur pour `latent` |
| `--subset` | good | bons spectres seulement, ou tous |
| `--n-dims` | 10 | composantes PCA gardées |
| `--k-min`, `--k-max` | 2, 12 | plage de k balayée |
| `--k` | auto | k final |
| `--seeds` | 10 | graines K-means pour la stabilité |
| `--min-cluster-size` | 100 | taille minimale d'un groupe HDBSCAN |

## Sorties

`runs/cluster_<features>_<subset>/` (suffixe `_k<k>` si `--k` est donné) :

- `kmeans_scan.csv` : inertie, silhouette, stabilité, BIC par k ;
- `metrics.json` : scores de validation des trois méthodes ;
- `crosstab_<méthode>_class.csv` : tableau groupe × classe, pour nommer chaque groupe.

`figures/cluster/<features>_<subset>/` :

- `kmeans_scan.png` : silhouette et stabilité en fonction de k ;
- `kiel_<méthode>.png` : diagramme Teff / log g coloré par groupe.

Le diagramme de Kiel est la figure la plus parlante : naines, sous-géantes, RGB et RC occupent des zones connues. Des groupes qui suivent ces zones montrent que le clustering retrouve la physique.
