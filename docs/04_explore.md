# 04_explore.py

## Rôle

Première réponse à la question du superviseur : les classes se séparent-elles dans les spectres ? Le script projette les spectres en 2D sans étiquettes, puis colore les points avec les étiquettes du catalogue.

Au premier lancement, il prétraite les spectres et crée `features_X.npy` et `features.npz` (voir [donnees.md](donnees.md)).

## Étapes

1. PCA à 50 composantes sur tous les spectres téléchargés. Résultat gardé dans `pca.npz` pour l'étape 07.
2. t-SNE sur 10 000 étoiles tirées au hasard, à partir des 50 composantes PCA.
3. UMAP sur le même sous-échantillon, si `umap-learn` est installé.

t-SNE part des composantes PCA et non des 7500 pixels : le calcul va plus vite et le bruit pixel par pixel pèse moins.

## Commande

```bash
python scripts/04_explore.py
python scripts/04_explore.py --tsne-n 20000 --perplexity 50
```

| Option | Défaut | Effet |
|---|---|---|
| `--n-components` | 50 | composantes PCA gardées |
| `--tsne-n` | 10000 | taille du sous-échantillon t-SNE et UMAP |
| `--perplexity` | 30 | perplexité t-SNE |

## Figures (`figures/explore/`)

| Fichier | Contenu |
|---|---|
| `pca_variance.png` | variance expliquée cumulée |
| `pca_components.png` | PC1 à PC3 en fonction de la longueur d'onde |
| `mean_spectra.png` | spectre moyen de chaque classe |
| `kiel_labels.png` | diagramme Teff / log g coloré par classe, pour vérifier les étiquettes |
| `pca_2d.png` | PC1 contre PC2, colorés par classe, log g, Teff, [M/H] et qualité |
| `tsne.png` | même chose pour t-SNE |
| `umap.png` | même chose pour UMAP |

## Lecture

- Des couleurs groupées dans `tsne.png` signifient que le spectre contient l'information de classe.
- Le panneau `quality` montre si les mauvais spectres forment leur propre groupe. Si oui, le bruit domine sur la physique pour ces étoiles.
- `pca_components.png` montre quelles raies portent la variance. Comparez avec les raies connues (Fe, Mg, CO, OH).

t-SNE préserve les voisins proches, pas les distances globales. Ne concluez rien sur la distance entre deux groupes éloignés.
