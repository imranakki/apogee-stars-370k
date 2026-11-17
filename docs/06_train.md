# 06_train.py

## Rôle

Entraîne un réseau de neurones sur les spectres, avec reprise après interruption. Deux modèles :

| `--model` | Usage | Sortie principale |
|---|---|---|
| `cnn` | supervisé : classe ou Teff, log g, [M/H] | `metrics.json` |
| `autoencoder` | non supervisé : compresse chaque spectre en 16 nombres | `latent.npy` pour l'étape 07 |

Les modèles sont définis dans `stars/models.py`.

## Entrée des réseaux

Le réseau reçoit 1 - flux, la profondeur des raies. Le continuum vaut alors 0 et les raies sont positives. Les données centrées près de zéro s'apprennent plus vite.

## CNN 1-D

Trois blocs convolution (noyau 7) + normalisation + ReLU + max-pooling (÷4), puis une couche dense de 128 et la sortie. Environ 150 000 paramètres. Sur une RTX 3060, une époque sur 20 000 spectres prend quelques secondes.

- Classification : perte d'entropie croisée.
- Régression : perte MSE sur des cibles standardisées avec la moyenne et l'écart-type du train. Les prédictions sont remises en unités physiques (K, dex) avant le calcul des métriques.

## Autoencodeur

Réseau dense : pixels → 512 → 128 → 16 → 128 → 512 → pixels, perte MSE. Le vecteur de 16 nombres au centre sert d'entrée au clustering. À la fin, le script encode toutes les étoiles et écrit `latent.npy`, dans le même ordre que `features.npz`.

## Commande

```bash
python scripts/06_train.py --model cnn --task class --train-on good
python scripts/06_train.py --model cnn --task regress --train-on all --epochs 50
python scripts/06_train.py --model autoencoder --train-on good --latent-dim 8
```

| Option | Défaut | Effet |
|---|---|---|
| `--model` | cnn | `cnn` ou `autoencoder` |
| `--task` | class | `class` ou `regress`, ignoré pour l'autoencodeur |
| `--train-on` | good | bons spectres seulement, ou tous |
| `--epochs` | 30 | nombre total d'époques, reprise comprise |
| `--batch-size` | 128 | taille de lot |
| `--lr` | 0.001 | taux d'apprentissage Adam |
| `--latent-dim` | 16 | taille du vecteur latent de l'autoencodeur |
| `--run-name` | déduit | nom du dossier dans `runs/` |
| `--fresh` | non | ignorer le checkpoint et repartir de zéro |

Nom par défaut du run : `<model>_<task>_<train_on>`, par exemple `cnn_class_good` ou `autoencoder_none_all`.

## Reprise

À chaque époque, `runs/<nom>/last.pt` reçoit le modèle, l'optimiseur, le numéro d'époque, la meilleure perte de validation et l'historique. Relancer la même commande recharge ce fichier et continue.

- Arrêt à l'époque 12 sur 30 : relancez, l'entraînement repart à l'époque 13.
- 30 époques finies, et vous en voulez 50 : relancez avec `--epochs 50`.
- Pour repartir de zéro : `--fresh`.

`last.pt` est d'abord écrit dans un fichier temporaire puis renommé. Une coupure pendant la sauvegarde laisse donc l'ancien checkpoint intact.

## Fichiers du run (`runs/<nom>/`)

| Fichier | Contenu |
|---|---|
| `last.pt` | dernier état, pour la reprise |
| `best.pt` | état à la meilleure perte de validation |
| `history.csv` | pertes train et val par époque |
| `config.json` | options de la commande |
| `metrics.json` | scores sur `test_good` et `test_all` (CNN) |
| `latent.npy` | vecteurs latents (autoencodeur) |

Les métriques finales utilisent `best.pt`, pas le dernier état.
