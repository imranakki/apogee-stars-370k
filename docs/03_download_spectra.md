# 03_download_spectra.py

## Rôle

Télécharge le spectre aspcapStar de chaque étoile de `catalog.csv` et le range dans deux grands tableaux :

| Fichier | Forme | Contenu |
|---|---|---|
| `flux.npy` | (N, 8575) float32 | flux normalisé au continuum |
| `err.npy` | (N, 8575) float32 | erreur par pixel |
| `status.npy` | (N,) int8 | 0 à faire, 1 téléchargé, 2 absent du serveur |

Vitesse mesurée : environ 19 spectres par seconde avec 24 workers, soit 1 h 30 pour 100 000 étoiles.

Les fichiers FITS ne sont pas gardés sur disque : le script lit le flux et l'erreur en mémoire, puis les écrit à la bonne ligne. Pour 370 000 étoiles, comptez 12,7 Go par tableau. Sur un disque exFAT, la création des deux tableaux écrit d'abord 25 Go de zéros : exFAT ne gère pas les fichiers creux.

## Commande

```bash
python scripts/03_download_spectra.py
python scripts/03_download_spectra.py --workers 32
python scripts/03_download_spectra.py --retry-missing
python scripts/03_download_spectra.py --seed-from /chemin/vers/apogee-stars-full-data
```

| Option | Défaut | Effet |
|---|---|---|
| `--workers` | 16 | téléchargements en parallèle |
| `--seed-from` | aucun | copier d'abord les spectres déjà téléchargés dans un autre dossier de données |
| `--retry-missing` | non | retenter les étoiles marquées absentes |

## Réutiliser une autre version

`--seed-from` lit `catalog.csv`, `status.npy`, `flux.npy` et `err.npy` d'un autre dossier. Chaque étoile reconnue par le triplet (APOGEE_ID, TELESCOPE, FIELD) et marquée OK là-bas est copiée ici, puis marquée OK. La version 370 000 étoiles récupère ainsi environ 80 000 spectres de la version 100 000 en 7 minutes, au lieu de 70 minutes de téléchargement. Relancer avec la même option ne recopie rien : seules les lignes encore à l'état 0 sont visées.

## Reprise

`status.npy` est sauvé toutes les 500 étoiles, à la fin, et après Ctrl-C. Au lancement suivant, le script ne télécharge que les étoiles à l'état 0. Une erreur réseau laisse l'étoile à 0 : elle sera retentée au prochain lancement.

## Étoiles absentes

Certaines étoiles du catalogue n'ont pas de fichier aspcapStar, surtout à faible SNR. Le serveur répond 404, l'étoile passe à l'état 2 et les étapes suivantes l'ignorent. Le bilan final affiche le nombre d'étoiles OK, absentes et restantes.

## Adresse d'un spectre

```
https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/<TELESCOPE>/<FIELD>/aspcapStar-dr17-<APOGEE_ID>.fits
```
