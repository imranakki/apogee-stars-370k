# 02_select_stars.py

## Rôle

Construit l'échantillon de travail et écrit `catalog.csv`. Pour chaque étoile, le script ajoute :

- `quality` : `good` ou `bad` ;
- `class` : `dwarf`, `subgiant`, `rgb`, `rc`, ou vide si log g manque ;
- `population` : `thin_disk`, `thick_disk`, `halo`, `bulge`, ou vide ;
- `occam_cluster` : nom de l'amas ouvert, ou vide ;
- `split` : `train`, `val` ou `test` ;
- `label_source` : `aspcap` (valeurs publiées), `fparam` (ajustement brut, pour les étoiles sans valeur publiée) ou vide ;
- `R_GC` : distance au centre galactique en kpc, à partir de la distance Gaia EDR3.

Les règles sont détaillées dans [donnees.md](donnees.md).

## Commande

```bash
python scripts/02_select_stars.py
python scripts/02_select_stars.py --n-good 30000 --n-bad 15000
```

| Option | Défaut | Effet |
|---|---|---|
| `--n-good` | 66000 | étoiles de bonne qualité tirées au hasard |
| `--n-bad` | 33000 | étoiles de mauvaise qualité tirées au hasard |
| `--no-occam` | non | ne pas ajouter d'office les membres d'amas ouverts |
| `--force` | non | écraser un `catalog.csv` existant |

## Sortie

`catalog.csv`, une ligne par étoile. La ligne i correspond à la ligne i de `flux.npy`. Le script affiche aussi un tableau classe × qualité et le nombre de membres OCCAM.

## Attention

Les spectres de l'étape 03 sont rangés dans l'ordre de `catalog.csv`. Le script refuse donc d'écraser le catalogue sans `--force`. Après `--force`, supprimez `flux.npy`, `err.npy`, `status.npy` et `features.npz`, puis relancez l'étape 03.

Le tirage utilise la graine 42 (`config.SEED`). La même commande donne le même échantillon sur toutes les machines de l'équipe.
