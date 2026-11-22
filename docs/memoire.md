# Tenir 370 000 spectres dans 15 Go de RAM

## Le problème

Un spectre prétraité compte 7 514 pixels. En float32, 370 000 spectres occupent 11,1 Go. La machine de référence a 15 Go de RAM, dont 10 Go libres. Un code direct chargerait tout en float32, copierait le jeu d'entraînement pour PyTorch et ajusterait la PCA sur la matrice entière. Chacune de ces étapes dépasse la RAM à 370 000 étoiles.

## Les quatre changements

| Changement | Fichier | Mémoire |
|---|---|---|
| Spectres prétraités en float16 dans `features_X.npy` | `stars/spectra.py`, `build_features` | 5,6 Go au lieu de 11,1 Go |
| Lots lus ligne par ligne, convertis en float32 au vol | `stars/training.py`, `Batches` | un lot à la fois au lieu d'une copie de 3,9 Go |
| PCA ajustée sur 50 000 spectres, projection par morceaux de 20 000 | `stars/spectra.py`, `fit_pca` et `project` | 1,5 Go au lieu de 11 Go |
| HDBSCAN sur 50 000 étoiles tirées au hasard | `scripts/07_cluster.py`, `--hdbscan-n` | temps de calcul raisonnable |

## Pourquoi float16 suffit

Le flux normalisé vaut entre 0 et 1,5. Près du continuum (1,0), float16 a un pas de 0,0005. Le bruit d'un bon spectre APOGEE vaut environ 0,01 par pixel, vingt fois plus. La perte de précision reste invisible pour les modèles.

## Profondeur des raies

Les réseaux reçoivent 1 − flux. Faire ce calcul une fois sur tout le tableau demanderait une copie en float32. `stars.training.depth` le fait sur chaque lot, au moment de la conversion en float32. `predict` et `Batches` l'appliquent tous les deux : les scripts passent X tel quel.

## PCA sur un sous-échantillon

La PCA s'ajuste sur 50 000 spectres tirés au hasard. Avec 50 000 spectres pour 50 composantes, l'estimation des axes reste stable.
