# 01_get_catalogs.py

## Rôle

Télécharge les trois catalogues dont dépend la suite :

| Fichier | Taille |
|---|---|
| `allStar-dr17-synspec_rev1.fits` | 4,0 Go |
| `apogee-rc-DR17.fits` | 266 Mo |
| `occam_member-DR17.fits` | 4 Mo |

Les adresses sont dans `stars/config.py` (`URLS`).

## Commande

```bash
python scripts/01_get_catalogs.py
```

Aucune option.

## Reprise

Un téléchargement en cours s'écrit dans `<fichier>.part`. Après une coupure, le script demande au serveur la suite du fichier (en-tête HTTP Range) au lieu de tout recommencer. Un fichier déjà complet est ignoré.

## Pourquoi allStar et pas allStarLite

`allStarLite` (1,8 Go) suffit pour Teff, log g, [M/H] et les drapeaux. `allStar` (4 Go) ajoute trois colonnes utiles :

| Colonne | Usage |
|---|---|
| `FPARAM` | ajustement brut d'ASPCAP, publié même pour les étoiles STAR_BAD : étiquettes des mauvais spectres |
| `FE_H`, `MG_FE` | abondances pour séparer disque mince et disque épais |
| `GAIAEDR3_R_MED_GEO` | distance Gaia EDR3, pour placer le bulbe par sa distance au centre galactique |
