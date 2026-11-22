# 03_download_spectra.py

## Rôle

Télécharge le spectre aspcapStar de chaque étoile de `catalog.csv` et le range dans deux grands tableaux :

| Fichier | Forme | Contenu |
|---|---|---|
| `flux.npy` | (N, 8575) float32 | flux normalisé au continuum |
| `err.npy` | (N, 8575) float32 | erreur par pixel |
| `status.npy` | (N,) int8 | 0 à faire, 1 téléchargé, 2 absent du serveur |

Vitesse mesurée : environ 15 spectres par seconde avec 24 workers sur `data.sdss.org`, soit 4 h 30 pour 240 000 étoiles. Le miroir `dr17.sdss.org` donne environ 6 spectres par seconde avec 8 workers.

Les fichiers FITS ne sont pas gardés sur disque : le script lit le flux et l'erreur en mémoire, puis les écrit à la bonne ligne. Pour 370 000 étoiles, comptez 12,7 Go par tableau. Sur un disque exFAT, la création des deux tableaux écrit d'abord 25 Go de zéros : exFAT ne gère pas les fichiers creux.

## Commande

```bash
python scripts/03_download_spectra.py
python scripts/03_download_spectra.py --workers 32
python scripts/03_download_spectra.py --retry-missing
python scripts/03_download_spectra.py --seed-from /chemin/vers/un/autre/dossier-data
```

| Option | Défaut | Effet |
|---|---|---|
| `--workers` | 16 | téléchargements en parallèle |
| `--seed-from` | aucun | copier d'abord les spectres déjà téléchargés dans un autre dossier de données |
| `--retry-missing` | non | retenter les étoiles marquées absentes |

## Réutiliser des spectres déjà téléchargés

`--seed-from` lit `catalog.csv`, `status.npy`, `flux.npy` et `err.npy` d'un autre dossier. Chaque étoile reconnue par le triplet (APOGEE_ID, TELESCOPE, FIELD) et marquée OK là-bas est copiée ici, puis marquée OK. Copier 100 000 spectres prend quelques minutes, contre 1 h 30 de téléchargement. Relancer avec la même option ne recopie rien : seules les lignes encore à l'état 0 sont visées.

## Changer de serveur

La variable `STARS_SAS` choisit le serveur. Par défaut, le script lit `https://data.sdss.org`. Pendant notre téléchargement, ce serveur a cessé de répondre après 240 000 spectres. Le miroir officiel a servi les 30 315 derniers :

```bash
STARS_SAS=https://dr17.sdss.org python scripts/03_download_spectra.py --workers 8
```

## Reprise

`status.npy` est sauvé toutes les 500 étoiles, à la fin, et après Ctrl-C. Au lancement suivant, le script ne télécharge que les étoiles à l'état 0. Une erreur réseau laisse l'étoile à 0 : elle sera retentée au prochain lancement.

## Étoiles absentes

Certaines étoiles du catalogue n'ont pas de fichier aspcapStar, surtout à faible SNR. Le serveur répond 404, l'étoile passe à l'état 2 et les étapes suivantes l'ignorent. Le bilan final affiche le nombre d'étoiles OK, absentes et restantes.

## Adresse d'un spectre

```
https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/<TELESCOPE>/<FIELD>/aspcapStar-dr17-<APOGEE_ID>.fits
```
