# Pourquoi ces nombres

Chaque valeur du projet a une raison. Ce fichier donne la raison, puis ce qui change si on prend une autre valeur. Les tests de sensibilité (`scripts/11_sensitivity.py`) mesurent cet effet sur la version 370 000 étoiles. Leurs résultats sont à la fin.

## Données

| Valeur | Choix | Raison |
|---|---|---|
| SNR minimal d'un bon spectre | 70 | Valeur courante dans les études ASPCAP pour des paramètres fiables. APOGEE vise un SNR de 100. Sous 70, l'erreur du CNN sur Teff passe de 62 K à 167 K (tranche 30 à 70). |
| SNR minimal d'une étoile gardée | 5 | Sous 5, le spectre est presque du bruit. ASPCAP n'en tire rien d'utile. |
| Drapeaux | STARFLAG = 0, pas de STAR_BAD | STARFLAG signale un problème d'observation (persistance, satellite, bord de détecteur). STAR_BAD signale un ajustement ASPCAP rejeté. |
| Teff et log g publiés | obligatoire | Sans eux, l'étoile n'a pas d'étiquette fiable pour juger les modèles. |
| Ratio bons / mauvais | 2 pour 1 | Assez de mauvais spectres pour mesurer leur effet (123 428), sans qu'ils dominent l'entraînement. |
| Découpage | 70 / 15 / 15 | Découpage standard. 15 % de test donne 37 034 bons spectres : l'écart type de l'accuracy reste sous 0,002. |
| Validation | bons spectres seulement | Chaque run choisit sa meilleure époque sur le même jeu propre, quel que soit son entraînement. |
| Graine | 42 | Valeur arbitraire, fixée pour que chaque tirage se reproduise à l'identique. |

## Étiquettes

| Valeur | Choix | Raison |
|---|---|---|
| Naine | log g ≥ 4,0 | La séquence principale se situe vers log g 4,0 à 4,7. |
| Sous-géante | 3,5 ≤ log g < 4,0 | Zone de transition après la sortie de la séquence principale. |
| Géante | log g < 3,5 | La branche des géantes commence sous 3,5. |
| Red clump | catalogue red clump DR17 (Bovy et al. 2014) | Red clump et RGB ont la même surface. Seule une méthode dédiée, calibrée sur l'astérosismologie, les sépare. Nous ne pouvons pas le faire avec log g seul. |
| Disque épais | [Mg/Fe] > 0,12 − 0,13 min([Fe/H], 0) | Ligne de Weinberg et al. (2019), tracée sur APOGEE. |
| Halo | [Fe/H] < −1 | Seuil classique : le disque compte peu d'étoiles sous −1. |
| Bulbe | moins de 3 kpc du centre galactique | Rayon typique du bulbe de la Voie lactée. |

Ces seuils sont des définitions physiques, pas des paramètres à optimiser. Nous ne les testons pas.

## Réduction de dimension

| Valeur | Choix | Raison |
|---|---|---|
| Composantes PCA pour le clustering | 10 | 10 composantes gardent 74,7 % de la variance. La courbe s'aplatit ensuite : 20 composantes gardent 77,0 %, 50 en gardent 79,6 %. Plus de dimensions ajoutent surtout du bruit pour K-means et le GMM. |
| Composantes PCA pour la forêt | 50, puis 10 | Choix de départ : la forêt choisit elle-même ses variables, 50 axes gardent les détails fins. Le test `rf_pca` montre l'inverse : 10 axes donnent un meilleur macro F1 (0,803 contre 0,776). Les meilleurs réglages passent à 10. |
| PCA ajustée sur | 50 000 spectres | 50 000 spectres pour 50 axes suffisent à estimer les axes. La matrice complète ne tient pas en mémoire. |
| Latent de l'autoencodeur | 16 | Entre les 10 axes PCA et une taille assez grande pour une compression non linéaire. |

## Clustering

| Valeur | Choix | Raison |
|---|---|---|
| k | 3 par la silhouette, 4 pour comparer | La silhouette est maximale à k = 3. 4 est le nombre de classes réelles. |
| Graines K-means | 10 | 45 paires de partitions pour mesurer la stabilité. Assez pour un écart type fiable, en quelques minutes. |
| Taille minimale HDBSCAN | 100 étoiles | Ignore les petits paquets de quelques étoiles, garde les groupes d'au moins 0,2 % du sous-échantillon. |
| Sous-échantillon HDBSCAN | 50 000 étoiles | HDBSCAN devient trop lent au-delà. |

## Modèles supervisés

| Valeur | Choix | Raison |
|---|---|---|
| Arbres de la forêt | 300 | Au-delà de quelques centaines d'arbres, la forêt ne gagne presque plus. |
| CNN | 3 blocs, 16, 32 et 64 filtres, noyau 7 | Un noyau de 7 pixels couvre une raie d'absorption APOGEE. 3 réductions par 4 ramènent 7 514 pixels à 117 positions. 150 000 paramètres s'entraînent en 15 minutes. |
| Époques | 30 | En classification, la meilleure époque de validation tombe vers 16. Le modèle gardé est celui de la meilleure époque : les époques en trop ne nuisent pas. En régression, 30 époques sont trop courtes (voir les résultats). |
| Taille de lot | 128 | Compromis courant entre vitesse sur GPU et bruit du gradient. |
| Taux d'apprentissage | 0,001 | Valeur par défaut d'Adam. |

## Tests de sensibilité

Un paramètre change à la fois. Tout le reste garde sa valeur de référence. Chaque test écrit `STARS_DATA/sensitivity/<test>.csv`.

```bash
python scripts/11_sensitivity.py cluster_dims cluster_k rf_trees rf_pca size epochs snr badfrac lr
python scripts/12_best.py
python report/make_sensitivity_figures.py
```

| Test | Valeurs | Question |
|---|---|---|
| `snr` | 30, 50, 70, 100, 150 | Où placer le seuil des bons spectres ? |
| `badfrac` | 0, 25, 50, 100 % des mauvais | Combien de mauvais spectres ajouter ? |
| `size` | 2 à 100 % des bons spectres | Combien de données faut-il ? |
| `epochs` | 1 à 50 | Combien d'époques ? |
| `lr` | 0,0001 à 0,003 | Quel taux d'apprentissage ? |
| `rf_trees` | 10 à 500 | Combien d'arbres ? |
| `rf_pca` | 5 à 100 | Combien de composantes pour la forêt ? |
| `cluster_dims` | 2 à 50 | Combien de composantes pour le clustering ? |
| `cluster_k` | 2 à 8, 10 graines chacun | Le GMM est-il stable selon k ? |

Le seuil SNR sous 70 ajoute peu d'étoiles : 7 320 à 30 et 4 585 à 50. L'échantillon tire 2 bons pour 1 mauvais, et les spectres propres de SNR 30 à 70 viennent seulement des mauvais tirés.

## Résultats des tests (370 000 étoiles, `test_good`)

Les CSV sont dans `results/sensitivity/`. Les figures sont dans `report/figures/sens_*.png`.

### Données

| Seuil SNR | 30 | 50 | 70 | 100 | 150 |
|---|---|---|---|---|---|
| Spectres d'entraînement | 179 832 | 177 097 | 172 512 | 150 303 | 99 289 |
| Macro F1, CNN | 0,860 | 0,861 | 0,866 | 0,858 | 0,858 |
| Erreur Teff, CNN (K) | 26,5 | 26,6 | 28,3 | 28,9 | 31,0 |

| Part des mauvais spectres | 0 | 25 % | 50 % | 100 % |
|---|---|---|---|---|
| Macro F1, `test_good` | 0,854 | 0,859 | 0,860 | 0,860 |
| Macro F1, `test_all` | 0,826 | 0,841 | 0,842 | 0,842 |
| Erreur Teff, `test_good` (K) | 29,2 | 31,5 | 42,1 | 40,9 |
| Erreur Teff, SNR < 70 (K) | 298 | 202 | 202 | 207 |

| Part des bons spectres | 2 % | 5 % | 10 % | 25 % | 50 % | 100 % |
|---|---|---|---|---|---|---|
| Macro F1, CNN | 0,785 | 0,817 | 0,819 | 0,833 | 0,856 | 0,865 |
| Macro F1, forêt | 0,635 | 0,689 | 0,715 | 0,741 | 0,766 | 0,776 |
| Erreur Teff, CNN (K) | 53,9 | 45,9 | 46,5 | 34,4 | 26,9 | 27,9 |

### Hyperparamètres

| Taux d'apprentissage | 0,0001 | 0,0003 | 0,001 | 0,003 |
|---|---|---|---|---|
| Perte de validation | 0,1651 | 0,1637 | 0,1616 | 0,1647 |
| Macro F1 | 0,861 | 0,858 | 0,865 | 0,862 |

| Arbres | 10 | 50 | 100 | 300 | 500 |
|---|---|---|---|---|---|
| Macro F1 | 0,742 | 0,767 | 0,771 | 0,776 | 0,779 |
| Erreur Teff (K) | 29,1 | 26,9 | 26,5 | 26,3 | 26,2 |

| Composantes PCA, forêt | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| Macro F1 | 0,794 | 0,803 | 0,800 | 0,776 | 0,737 |
| Erreur Teff (K) | 36,2 | 26,6 | 26,0 | 26,3 | 27,0 |
| Erreur log g | 0,083 | 0,085 | 0,091 | 0,106 | 0,115 |

Époques, sur 50 :

- Classification : perte de validation minimale à l'époque 16 (0,163, macro F1 0,863). À l'époque 30, elle remonte à 0,188 et le macro F1 tombe à 0,835. Le modèle surapprend, et garder la meilleure époque corrige l'effet.
- Régression : la perte de validation baisse encore à l'époque 50. L'erreur sur Teff y vaut 25,0 K, contre 35,3 K à l'époque 30. 30 époques sont trop courtes pour la régression.

### Clustering

| Composantes PCA | 2 | 5 | 10 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|
| ARI, GMM, k = 3 | −0,03 | 0,33 | 0,41 | −0,03 | 0,40 | 0,37 |

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| ARI médian, GMM (10 graines) | −0,03 | 0,41 | 0,10 | 0,29 | 0,27 | 0,23 | 0,20 |
| ARI maximal, GMM | 0,38 | 0,43 | 0,30 | 0,31 | 0,27 | 0,23 | 0,22 |
| ARI médian, K-means | 0,07 | 0,08 | 0,06 | 0,07 | 0,07 | 0,06 | 0,22 |

### Bruit des graines

Le même CNN, avec les graines 42, 43 et 44 :

- macro F1 : 0,852 ± 0,010
- erreur Teff : 26,3 ± 1,6 K

Un écart de 0,01 entre deux runs uniques ne prouve rien. La forêt, à graine fixe, donne le même résultat à chaque run.

## Meilleurs réglages

`scripts/12_best.py` choisit chaque valeur sur la validation, puis mesure le résultat une seule fois sur le test. Chaque CNN s'entraîne avec 3 graines. Les choix et leurs raisons sont dans `results/best/selection.json`, et les runs dans `results/best/final.csv`.

| Modèle | Réglage choisi | Raison |
|---|---|---|
| CNN, classes | SNR 70, 50 % des mauvais spectres, taux 0,001 | plus faible perte de validation de chaque test |
| CNN, régression | SNR 50, aucun mauvais spectre, taux 0,001 | plus faible perte de validation de chaque test |
| Forêt, classes | 10 composantes, 300 arbres | macro F1 de validation 0,804, contre 0,773 à 50 composantes |
| Forêt, régression | 10 composantes, 500 arbres | meilleure erreur standardisée de validation |
| GMM | k = 3, graine 4 | silhouette pour k, plus faible BIC parmi 10 graines pour la graine, sans étiquette |

| Modèle | Mesure (test) | Référence | Meilleurs réglages |
|---|---|---|---|
| CNN, 3 graines | macro F1, `test_good` | 0,852 ± 0,010 | 0,865 ± 0,000 |
| | macro F1, `test_all` | 0,822 ± 0,009 | 0,847 ± 0,001 |
| | macro F1, SNR < 70 | 0,642 ± 0,023 | 0,738 ± 0,022 |
| | erreur Teff (K) | 26,3 ± 1,6 | 26,6 ± 1,0 |
| | erreur log g | 0,061 ± 0,002 | 0,055 ± 0,004 |
| | erreur [M/H] | 0,023 ± 0,003 | 0,021 ± 0,003 |
| Forêt | macro F1, `test_good` | 0,776 | 0,803 |
| | macro F1, SNR < 70 | 0,614 | 0,683 |
| | erreur Teff (K) | 26,3 | 26,6 |
| | erreur log g | 0,106 | 0,085 |
| | erreur [M/H] | 0,040 | 0,035 |
| GMM, k = 3 | ARI | 0,41 | 0,43 |

En classification, les meilleurs réglages gagnent 0,013 de macro F1 pour le CNN et 0,027 pour la forêt. Le plus grand gain porte sur les spectres bruités : 0,738 contre 0,642 pour le CNN. En régression, l'erreur sur Teff ne bouge pas, et l'écart reste dans le bruit des graines. log g gagne 10 % avec le CNN et 20 % avec la forêt.

Prochaine étape : entraîner le CNN de régression sur 50 époques ou plus.
