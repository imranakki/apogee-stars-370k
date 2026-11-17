# 05_baseline.py

## Rôle

Modèle supervisé simple qui sert de référence. Si le CNN de l'étape 06 ne fait pas mieux, il n'apporte rien.

Pipeline : PCA (50 composantes) puis forêt aléatoire (300 arbres). La PCA est ajustée sur le train seulement.

## Tâches

| `--task` | Cible | Métriques |
|---|---|---|
| `class` | dwarf, subgiant, rgb, rc | accuracy, macro F1 |
| `regress` | Teff, log g, [M/H] | erreur absolue moyenne par cible |

Le macro F1 donne le même poids à chaque classe. Il compte plus que l'accuracy ici, car les géantes RGB dominent l'échantillon.

## Commande

```bash
python scripts/05_baseline.py --task class --train-on good
python scripts/05_baseline.py --task regress --train-on all
```

| Option | Défaut | Effet |
|---|---|---|
| `--task` | class | classification ou régression |
| `--train-on` | good | entraîner sur les bons spectres ou sur tous |
| `--n-components` | 50 | composantes PCA |
| `--n-trees` | 300 | arbres de la forêt |

## Sortie

`runs/rf_<task>_<train_on>/metrics.json` avec les scores sur `test_good` et `test_all`.

Pas de reprise ici : l'entraînement dure quelques minutes.
