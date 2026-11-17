# 08_report.py

## Rôle

Lit tous les `runs/*/metrics.json` et produit deux tableaux :

| Fichier | Contenu |
|---|---|
| `summary_supervised.csv` | forêts aléatoires et CNN, scores sur `test_good` et `test_all` |
| `summary_clustering.csv` | runs de clustering, scores de chaque méthode |

Le script affiche aussi les colonnes principales dans le terminal.

## Commande

```bash
python scripts/08_report.py
```

## Lire le tableau supervisé

Chaque ligne est un run. Pour l'expérience qualité, comparez les paires `*_good` et `*_all` du même modèle :

- colonne `test_good.*` : effet des mauvais spectres à l'entraînement sur une évaluation propre ;
- colonne `test_all.*` : performance sur des données réelles, bruit compris.

Un exemple de lecture : si `cnn_class_all` a un meilleur macro F1 que `cnn_class_good` sur `test_all`, les spectres bruités à l'entraînement aident le modèle à généraliser au bruit.
