# Le package `stars/`

Code partagé par les scripts. Un script contient l'enchaînement d'une étape, le package contient la logique réutilisée.

| Module | Contenu |
|---|---|
| `config.py` | chemins, adresses SDSS, seuils de qualité, listes de classes, graine |
| `catalog.py` | lecture des FITS, coupes de qualité, classes, populations, split, lecture de `catalog.csv` |
| `download.py` | téléchargement avec reprise, lecture d'un spectre aspcapStar en mémoire |
| `spectra.py` | prétraitement, cache `features.npz`, `load_dataset()`, grille de longueurs d'onde |
| `tasks.py` | lignes et cibles des tâches supervisées, masques train/val/test, métriques |
| `models.py` | `SpectrumCNN` et `Autoencoder` (PyTorch) |
| `training.py` | `Checkpointer` (sauvegarde et reprise), boucle d'époque, prédiction par lots |
| `plots.py` | projections 2D colorées, diagramme de Kiel, courbes |

## Fonctions à connaître

`spectra.load_dataset()` renvoie `(X, cat)` :

- `X` : tableau (N, ~7500) des spectres prétraités, float32 ;
- `cat` : DataFrame du catalogue, ligne i = spectre i, avec la colonne `bad_pixel_frac` en plus.

Utilisez cette fonction dans un notebook pour vos propres analyses :

```python
import sys; sys.path.insert(0, "/chemin/vers/apogee-stars")
from stars import spectra
X, cat = spectra.load_dataset()
```

Pensez à définir `STARS_DATA` avant de lancer Python.

`tasks.split_masks(cat, task, train_on)` renvoie les masques `train`, `val`, `test_good`, `test_all`. Tous les modèles supervisés passent par là, donc ils voient exactement les mêmes étoiles.

## Changer un réglage

| Pour changer | Modifier |
|---|---|
| seuil SNR des bons spectres | `config.GOOD_SNR` |
| règles de qualité | `catalog.is_good` |
| frontières des classes | `catalog.evolutionary_class` |
| coupe thin / thick disk | `catalog.population` |
| prétraitement | `spectra.MAX_ERR`, `spectra.FLUX_CLIP` |
| architecture du CNN | `models.SpectrumCNN` |

Après un changement dans `catalog.py`, relancez à partir de l'étape 02 avec `--force`. Après un changement dans `spectra.py`, supprimez `features.npz`.
