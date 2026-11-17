# Données, qualité et étiquettes

## Sources

| Fichier | Contenu | Usage |
|---|---|---|
| `allStar-dr17-synspec_rev1.fits` | une ligne par étoile : position, SNR, drapeaux, paramètres ASPCAP, FPARAM, abondances, distance Gaia | sélection et étiquettes |
| `aspcapStar-dr17-<ID>.fits` | spectre normalisé au continuum, 8575 pixels, et son erreur | entrée des modèles |
| `apogee-rc-DR17.fits` | liste des étoiles du red clump | séparer RC et RGB |
| `occam_member-DR17.fits` | membres d'amas ouverts avec probabilités | validation du clustering |

Chaque fichier aspcapStar contient quatre HDU. Le projet lit HDU 1 (flux normalisé) et HDU 2 (erreur). La grille est log-linéaire : log10(λ) = 4,179 + 6e-6 × pixel, soit 15 100 à 17 000 Å environ.

## Étoiles exclues d'office

Fonction `catalog.usable` :

- télescope de 1 m (`apo1m`), instrument différent ;
- doublons (bit 4 de `EXTRATARG`), on garde une seule entrée par étoile ;
- standards telluriques (bit 2 de `EXTRATARG`), étoiles chaudes sans intérêt ici ;
- SNR < 5.

## Bonne ou mauvaise qualité

APOGEE ne fournit pas une étiquette unique. Le projet définit (fonction `catalog.is_good`) :

| Critère | Bon spectre |
|---|---|
| `SNR` | ≥ 70 |
| `STARFLAG` | 0, aucun problème signalé sur les visites |
| `ASPCAPFLAG` | bit 23 (`STAR_BAD`) absent |
| `TEFF`, `LOGG` | valeurs présentes |

Toute autre étoile utilisable est « bad ». Changez les seuils dans `stars/config.py` (`GOOD_SNR`) et dans `catalog.is_good`, puis relancez à partir de l'étape 02.

Une étoile « bad » n'a pas toujours de Teff ou log g : ASPCAP met NaN quand l'ajustement échoue. Ces étoiles servent au clustering et à l'autoencodeur, mais pas au supervisé.

## Classes évolutives (colonne `class`)

| Classe | Règle |
|---|---|
| `dwarf` | log g ≥ 4,0 |
| `subgiant` | 3,5 ≤ log g < 4,0 |
| `rc` | log g < 3,5 et présente dans le catalogue red clump |
| `rgb` | log g < 3,5 et absente du catalogue red clump |

Le catalogue red clump contient aussi des étoiles du secondary clump. La classe `rgb` contient donc aussi des géantes AGB.

## Étiquettes des mauvais spectres (colonne `label_source`)

ASPCAP ne publie pas Teff ni log g pour les étoiles marquées STAR_BAD. Le catalogue complet garde l'ajustement brut dans `FPARAM`. La fonction `catalog.fill_labels` complète les valeurs manquantes avec `FPARAM` (Teff : indice 0, log g : indice 1, [M/H] : indice 3).

| `label_source` | Origine |
|---|---|
| `aspcap` | valeurs publiées par ASPCAP |
| `fparam` | ajustement brut, moins fiable |
| vide | aucune valeur |

La qualité se décide avant ce complément : une étoile étiquetée par `fparam` reste un mauvais spectre.

FPARAM n'est pas calibré. Sur un échantillon de 20 000 étoiles, son Teff est en médiane 75 K plus bas que le Teff publié, et son log g 0,04 dex plus haut. [M/H] ne change pas.

## Populations (colonne `population`)

Fonction `catalog.population` :

- `thin_disk` et `thick_disk` : ligne de Weinberg et al. (2019) dans le plan [Mg/Fe] contre [Fe/H]. Une étoile est du disque épais si [Mg/Fe] > 0,12 - 0,13 × [Fe/H] pour [Fe/H] < 0, ou [Mg/Fe] > 0,12 au-delà ;
- `halo` : [Fe/H] < -1 ;
- `bulge` : [Fe/H] ≥ -1 et distance au centre galactique `R_GC` < 3 kpc.

`R_GC` vient de la distance Gaia EDR3 (`GAIAEDR3_R_MED_GEO`) et de la position (l, b), avec 8,2 kpc entre le Soleil et le centre galactique. Les distances des étoiles du bulbe restent incertaines, car leur parallaxe est faible.

## Membres OCCAM (colonne `occam_cluster`)

Une étoile est membre si `RV_PROB`, `FEH_PROB` et `PM_PROB` dépassent toutes 0,5. Par défaut, l'étape 02 ajoute tous ces membres à l'échantillon, en plus du tirage aléatoire, pour avoir assez d'étoiles par amas.

## Split train / val / test

Tirage aléatoire 70 / 15 / 15 avec la graine 42, fait une seule fois à l'étape 02 et gardé dans la colonne `split`. Tous les modèles utilisent le même split, donc les résultats se comparent directement.

- La validation utilise toujours les bons spectres, quel que soit le jeu d'entraînement.
- Le test est évalué deux fois : `test_good` (bons spectres) et `test_all` (tous).

## Prétraitement (`stars/spectra.py`)

1. Retrait des trous entre les trois détecteurs (pixels à zéro dans plus de la moitié des spectres), environ 1060 pixels.
2. Pixel douteux si flux ≤ 0, valeur non finie, ou erreur > 0,1. Il est remplacé par 1,0, le niveau du continuum.
3. Flux borné à [0 ; 1,5].

La fraction de pixels douteux de chaque étoile est gardée dans `bad_pixel_frac`. Elle donne une mesure de qualité continue, utile pour le rapport.
