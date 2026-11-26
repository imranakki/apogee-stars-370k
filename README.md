# Classification d'étoiles à partir des spectres APOGEE DR17 : 370 000 étoiles

Félix Dubé, Imran Akki, Charles-Antoine Fournier, Youssef Lahrichi, Bastien Posez
Projet de session GIF-4101 / GIF-7005 / GIF-7015, Université Laval, décembre 2026

Rapport complet : [`report/rapport.pdf`](report/rapport.pdf). Présentation : [`report/presentation.pdf`](report/presentation.pdf). Vue 3D des groupes : [`report/clusters3d/index.html`](report/clusters3d/index.html).

---

## Résumé

Nous classons 370 000 étoiles APOGEE DR17 à partir de leur seul spectre infrarouge. Le code tient dans 15 Go de mémoire, sur un portable avec une RTX 3060.

| Résultat | Valeur |
|---|---|
| Macro F1, CNN 1-D, 4 classes, meilleurs réglages (3 graines) | **0,865 ± 0,000** |
| Erreur sur Teff, CNN entraîné sur les bons spectres | **24,7 K** |
| Erreur sur Teff, CNN de régression sur 100 époques (3 graines) | **22,1 ± 0,4 K** |
| ARI du GMM à 3 groupes, sans étiquettes | **0,41** |
| Naines regroupées dans un même groupe du GMM | **73 024 sur 73 040** |
| Red clump manqué par le catalogue red clump DR17 | **45 %** |
| F1 red clump / RGB contre l'astérosismologie APOKASC-3 | **0,978 ± 0,003** |
| Durée de l'expérience principale (RTX 3060 portable) | **2 heures** |

Trois messages :

1. Sans étiquettes, le spectre sépare naines, haut de la RGB et red clump avec bas de la RGB. À k = 4, l'ARI tombe à 0,10 et dépend de la graine.
2. Avec étiquettes, le CNN dépasse la forêt sur 2 % des données d'entraînement déjà. Les mauvais spectres aident la classification et nuisent à la régression sur les bons spectres.
3. La confusion red clump / RGB venait des étiquettes, pas du spectre. Avec les étiquettes d'astérosismologie, un CNN atteint 99 % de rappel et 3 % de contamination.

---

## 1. Introduction

APOGEE DR17 publie les spectres infrarouges de 657 000 étoiles. Le pipeline ASPCAP en tire la température effective Teff, la gravité de surface log g et la métallicité [M/H]. Notre question : un algorithme regroupe-t-il les étoiles en classes physiques à partir du spectre seul ?

Garcia-Dias et al. (2018) ont montré que K-means sépare naines, sous-géantes, red clump et RGB dans APOGEE. Leur résultat dépendait de l'initialisation. Nous testons ce résultat sur 370 000 étoiles et suivons les trois demandes du cours :

1. Réduction de dimension (PCA, t-SNE, UMAP) pour voir si les classes se séparent.
2. Apprentissage supervisé avec un CNN 1-D, comparé à une forêt aléatoire.
3. Impact de la qualité des données : entraînement sur les bons spectres, puis sur tous.

Nous mesurons aussi l'effet de la taille d'entraînement, de 2 % à 100 % des bons spectres, et testons chaque paramètre du projet.

---

## 2. Données

### 2.1 Échantillon

APOGEE observe en bande H (15 150 à 16 940 Å) à une résolution R ≈ 22 500. Chaque spectre normalisé (`aspcapStar`) compte 8 575 pixels. Le catalogue `allStar` compte 733 901 lignes.

```mermaid
flowchart LR
    A["allStar DR17<br/>733 901 lignes"] --> B["Filtres<br/>pas de télescope 1 m<br/>pas de doublons<br/>pas de standards telluriques<br/>SNR ≥ 5"]
    B --> C["Tirage, graine 42<br/>246 200 bons<br/>123 100 mauvais"]
    C --> D["+ 785 membres<br/>d'amas OCCAM"]
    D --> E["370 085 étoiles<br/>demandées"]
    E --> F["370 000 spectres reçus<br/>246 572 bons<br/>123 428 mauvais"]
    E -.-> G["85 manquants<br/>57 non publiés<br/>28 échecs réseau"]
```

Un bon spectre vérifie quatre critères : SNR ≥ 70, `STARFLAG = 0`, pas de `STAR_BAD` dans `ASPCAPFLAG`, Teff et log g publiés. Le SNR médian vaut 168 pour les bons et 103 pour les mauvais. 69 % des mauvais spectres ont un SNR ≥ 70 : un drapeau les écarte, pas le bruit.


### 2.2 Étiquettes

```mermaid
flowchart TD
    S["Étoile avec log g"] --> Q1{"log g ≥ 4,0 ?"}
    Q1 -- oui --> N["Naine<br/>73 040"]
    Q1 -- non --> Q2{"log g ≥ 3,5 ?"}
    Q2 -- oui --> SG["Sous-géante<br/>13 147"]
    Q2 -- non --> Q3{"Dans le catalogue<br/>red clump DR17 ?"}
    Q3 -- oui --> RC["Red clump<br/>22 931"]
    Q3 -- non --> RGB["RGB<br/>137 454"]
```

Effectifs sur les bons spectres. Les populations suivent [Mg/Fe], [Fe/H] et la distance Gaia EDR3 :

| Population | Critère | Bons spectres |
|---|---|---:|
| Disque mince | sinon | 175 184 |
| Disque épais | [Mg/Fe] > 0,12 − 0,13 min([Fe/H], 0) | 49 879 |
| Bulbe | moins de 3 kpc du centre galactique | 9 936 |
| Halo | [Fe/H] < −1 | 8 164 |

359 639 étoiles portent une étiquette ASPCAP et 10 361 une étiquette FPARAM, l'ajustement brut qu'ASPCAP a rejeté.

### 2.3 Prétraitement et découpage

Nous retirons les 1 061 pixels des trous entre détecteurs (il reste 7 514 pixels), remplaçons chaque pixel douteux par 1 et bornons le flux entre 0 et 1,5. Les réseaux reçoivent 1 − flux, la profondeur des raies.

```mermaid
pie showData
    title Découpage 70 / 15 / 15 (370 000 étoiles)
    "Entraînement" : 258997
    "Validation" : 55498
    "Test" : 55505
```

Le test compte 37 034 bons spectres (`test_good`). Tous les scores ci-dessous portent sur `test_good`, sauf mention contraire.

---

## 3. Tenir 370 000 spectres dans 15 Go

En float32, 370 000 spectres de 7 514 pixels occupent 11,1 Go. La machine a 15 Go de mémoire, dont 10 Go libres. Un code qui charge tout en float32, copie le jeu d'entraînement pour PyTorch et ajuste la PCA sur la matrice entière dépasse la mémoire à trois endroits.

| Changement | Approche directe | Ce dépôt |
|---|---:|---:|
| Spectres stockés en float16 (`features_X.npy`) | 11,1 Go | 5,6 Go |
| Lots lus ligne par ligne, convertis en float32 au vol | copie de 3,9 Go | un lot |
| PCA ajustée sur 50 000 spectres, projection par morceaux | 11 Go | 1,5 Go |
| HDBSCAN sur 50 000 étoiles tirées au hasard | toutes | 50 000 |

Float16 suffit. Près du continuum, son pas vaut 0,0005, vingt fois moins que le bruit d'un bon spectre (environ 0,01 par pixel). Détails dans [`docs/memoire.md`](docs/memoire.md).

---

## 4. Méthodes

### 4.1 Pipeline

```mermaid
flowchart TD
    subgraph D["Données"]
        S1["01 catalogues<br/>allStar, red clump, OCCAM"] --> S2["02 sélection<br/>qualité, étiquettes, split"]
        S2 --> S3["03 téléchargement<br/>370 000 spectres"]
    end
    S3 --> P["Prétraitement<br/>7 514 pixels, float16"]
    P --> E["04 exploration<br/>PCA, t-SNE, UMAP"]
    P --> PCA
    P --> AE
    P --> RF
    P --> CNN
    subgraph NS["Non supervisé"]
        PCA["PCA 10 axes"] --> CL["07 K-means, GMM, HDBSCAN"]
        AE["Autoencodeur<br/>latent 16"] --> CL
    end
    subgraph SU["Supervisé"]
        RF["05 forêt aléatoire<br/>sur PCA"]
        CNN["06 CNN 1-D<br/>spectre complet"]
    end
    CL --> V["Validation<br/>ARI, NMI, pureté OCCAM"]
    RF --> M["Mesures<br/>macro F1, erreur Teff, log g, [M/H]"]
    CNN --> M
    M --> Q["09 qualité par SNR<br/>10 origine des étiquettes"]
    M --> SE["11 sensibilité<br/>12 meilleurs réglages"]
    CNN --> RR["13 red clump / RGB<br/>14 APOKASC-3"]
```

### 4.2 Modèles

- **Réduction de dimension.** PCA à 50 composantes (10 axes portent 74,7 % de la variance), t-SNE sur 10 000 étoiles.
- **Autoencodeur.** Réseau dense 7 514 → 512 → 128 → 16 et retour, perte quadratique.
- **Clustering.** K-means, GMM à covariances pleines et HDBSCAN, sur 10 composantes PCA standardisées ou sur le latent de l'autoencodeur. k choisi par la silhouette. Stabilité mesurée par l'ARI moyen entre 10 graines.
- **Validation du clustering.** ARI, NMI, pureté des 120 amas ouverts OCCAM.
- **Forêt aléatoire.** 300 arbres sur 50 composantes PCA.
- **CNN 1-D.** 150 000 paramètres, Adam, taux 0,001, lots de 128, 30 époques, meilleure époque de validation gardée. Deux tâches : 4 classes, et régression de Teff, log g, [M/H].

```mermaid
flowchart LR
    X["Spectre<br/>1 × 7 514"] --> C1["Conv1d 16, noyau 7<br/>BatchNorm, ReLU<br/>MaxPool 4"]
    C1 -- "16 × 1 878" --> C2["Conv1d 32, noyau 7<br/>BatchNorm, ReLU<br/>MaxPool 4"]
    C2 -- "32 × 469" --> C3["Conv1d 64, noyau 7<br/>BatchNorm, ReLU<br/>MaxPool 4"]
    C3 -- "64 × 117" --> AP["AdaptiveAvgPool 16<br/>Flatten 1 024"]
    AP --> FC["Linear 128<br/>ReLU, Dropout 0,2"]
    FC --> O["Sortie<br/>4 classes ou<br/>Teff, log g, [M/H]"]
```

Un noyau de 7 pixels couvre une raie. Trois blocs ramènent 7 514 pixels à 117.

### 4.3 Expérience qualité

Chaque modèle s'entraîne deux fois : sur les bons spectres (172 512 étoiles), puis sur tous (258 997). La validation reste sur les bons spectres pour que chaque run choisisse son époque sur le même jeu propre.

---

## 5. Résultats

### 5.1 Exploration

![t-SNE](report/figures/tsne.png)
*Figure 1. t-SNE de 10 000 étoiles. Gauche : classes. Droite : qualité.*

Naines et géantes forment deux régions séparées, reliées par les sous-géantes. Le red clump reste mêlé à la RGB. Les mauvais spectres se mêlent aux bons.

![Spectres moyens](report/figures/mean_spectra_zoom.png)
*Figure 2. Spectre moyen de chaque classe sur 130 Å, décalé verticalement.*

![Variance PCA](report/figures/pca_variance.png)
*Figure 3. Variance expliquée par la PCA.*

### 5.2 Clustering

| Entrée | Méthode | ARI | NMI | Pureté OCCAM |
|---|---|---:|---:|---:|
| PCA, bons, k = 3 | **GMM** | **0,41** | **0,52** | 0,90 |
| PCA, bons, k = 4 | GMM | 0,10 | 0,32 | 0,85 |
| PCA, bons, k = 3 | K-means | 0,08 | 0,26 | 0,95 |
| PCA, bons, k = 4 | K-means | 0,06 | 0,26 | 0,91 |
| PCA, bons | HDBSCAN (24 % de bruit) | −0,03 | 0,03 | 0,52 |
| Latent, bons, k = 2 | K-means | −0,07 | 0,13 | 0,99 |
| PCA, tous, k = 2 | GMM | −0,04 | 0,04 | 0,94 |
| PCA, tous, k = 2 | K-means | 0,00 | 0,00 | 1,00 |

Le GMM à trois groupes reste la meilleure méthode. Composition de ses groupes :

| Groupe | Naines | Sous-géantes | RGB | Red clump |
|---|---:|---:|---:|---:|
| A : séquence principale | **73 024** | 10 796 | 7 126 | 201 |
| B : haut de la RGB | 3 | 9 | **55 943** | 463 |
| C : red clump et bas de la RGB | 13 | 2 342 | 74 385 | **22 267** |

```mermaid
flowchart LR
    N["Naines<br/>73 040"] -- "73 024" --> A["A : séquence principale<br/>91 147"]
    N -- "13" --> C
    SG["Sous-géantes<br/>13 147"] -- "10 796" --> A
    SG -- "9" --> B
    SG -- "2 342" --> C
    RGB["RGB<br/>137 454"] -- "7 126" --> A
    RGB -- "55 943" --> B["B : haut de la RGB<br/>56 418"]
    RGB -- "74 385" --> C["C : red clump et bas de la RGB<br/>99 007"]
    RC["Red clump<br/>22 931"] -- "201" --> A
    RC -- "463" --> B
    RC -- "22 267" --> C
```

![Kiel GMM](report/figures/kiel_gmm.png)
*Figure 4. Diagramme de Kiel des bons spectres, coloré par les groupes du GMM à k = 3.*

À k = 4, l'ARI tombe à 0,10. Le quatrième groupe coupe les naines en deux, et l'un des morceaux absorbe 18 753 étoiles du red clump et 62 668 RGB. Sur 10 graines, l'ARI à k = 4 ne dépasse pas 0,30 (section 6). Le choix k = 4 reste fragile.

![Scan K-means](report/figures/kmeans_scan.png)
*Figure 5. Silhouette et stabilité de K-means selon k (PCA, bons spectres).*

La silhouette reste basse (maximum 0,28 à k = 3) : les spectres forment un continuum. K-means trouve des groupes stables à k = 4 (stabilité 0,87), mais ces groupes ne suivent pas les classes (ARI 0,06). Les populations galactiques restent introuvables (ARI ≤ 0,11). Sur tous les spectres, la silhouette choisit k = 2 et K-means isole 1 004 étoiles atypiques.

### 5.3 Apprentissage supervisé

| Modèle | Entraînement | Accuracy | Macro F1 | Teff (K) | log g (dex) | [M/H] (dex) |
|---|---|---:|---:|---:|---:|---:|
| Forêt | bons | 0,898 | 0,776 | 26,3 | 0,106 | 0,040 |
| Forêt | tous | 0,902 | 0,791 | 25,2 | 0,093 | 0,037 |
| CNN 1-D | bons | **0,918** | 0,848 | **24,7** | **0,050** | **0,021** |
| CNN 1-D | tous | **0,918** | **0,857** | 45,1 | 0,081 | 0,026 |

*Erreurs absolues moyennes par rapport à ASPCAP, sur `test_good` (37 034 étoiles).*

![Confusion CNN](report/figures/confusion_cnn.png)
*Figure 6. Matrice de confusion du CNN entraîné sur les bons spectres.*

```mermaid
xychart-beta
    title "Rappel du CNN par classe (%)"
    x-axis ["Naines", "Sous-géantes", "RGB", "Red clump"]
    y-axis "Rappel (%)" 0 --> 100
    bar [98, 85, 95, 58]
```

Le CNN reconnaît 98 % des naines, 85 % des sous-géantes et 95 % des RGB. Le red clump plafonne à 58 % : 42 % sont classées RGB. Le CNN bat aussi la forêt sur Teff : 24,7 K contre 26,3 K.

### 5.4 Impact de la qualité des données

```mermaid
flowchart LR
    B["Ajouter les<br/>mauvais spectres"] --> C["Classification<br/>macro F1 0,848 → 0,857<br/>gain dans toutes les tranches de SNR"]
    B --> R["Régression sur bons spectres<br/>Teff 24,7 → 45,1 K<br/>log g 0,050 → 0,081 dex"]
    B --> RN["Régression sur spectres bruités<br/>SNR 5-30 : Teff 611 → 525 K"]
```

| SNR | n | F1 bons | F1 tous | log g bons | log g tous | Teff bons | Teff tous |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 à 30 | 1 461 | 0,45 | **0,54** | 0,61 | **0,45** | 611 | **525** |
| 30 à 70 | 4 240 | 0,68 | **0,76** | 0,15 | 0,15 | 167 | **123** |
| 70 à 150 | 22 931 | 0,80 | **0,82** | **0,070** | 0,091 | **62** | 67 |
| > 150 | 26 873 | 0,85 | **0,86** | **0,054** | 0,082 | 58 | **50** |

*CNN par tranche de SNR sur `test_all`, selon le jeu d'entraînement. log g en dex, Teff en K.*

![SNR](report/figures/snr.png)
*Figure 7. Erreur du CNN de régression par tranche de SNR.*

L'erreur vient des étiquettes. Sur les 1 504 étoiles de test étiquetées par FPARAM, l'erreur sur Teff atteint 882 K. Le modèle entraîné sur tous les spectres apprend une partie de ce bruit.

| Étiquette | Qualité | n | Teff bons (K) | Teff tous (K) | log g bons / tous (dex) |
|---|---|---:|---:|---:|---:|
| ASPCAP | bonne | 37 034 | 24,7 | 45,1 | 0,050 / 0,081 |
| ASPCAP | mauvaise | 16 967 | 138 | 101 | 0,113 / 0,115 |
| FPARAM | mauvaise | 1 504 | 882 | 514 | 0,541 / 0,412 |

### 5.5 Effet de la taille d'entraînement

Le test de sensibilité `size` entraîne chaque modèle sur une fraction des 172 512 bons spectres d'entraînement. Le test reste `test_good`.

| Fraction | Étoiles | Macro F1, CNN | Macro F1, forêt | Teff CNN (K) | Teff forêt (K) | log g CNN (dex) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 % | 3 450 | 0,785 | 0,635 | 53,9 | 47,2 | 0,105 |
| 5 % | 8 626 | 0,817 | 0,689 | 45,9 | 39,0 | 0,090 |
| 10 % | 17 251 | 0,819 | 0,715 | 46,5 | 35,1 | 0,085 |
| 25 % | 43 128 | 0,833 | 0,741 | 34,4 | 30,2 | 0,067 |
| 50 % | 86 256 | 0,856 | 0,766 | 26,9 | 28,2 | 0,060 |
| 100 % | 172 512 | **0,865** | **0,776** | 27,9 | **26,3** | **0,058** |

```mermaid
xychart-beta
    title "Macro F1 selon la taille d'entraînement (CNN en haut, forêt en bas)"
    x-axis ["2 %", "5 %", "10 %", "25 %", "50 %", "100 %"]
    y-axis "Macro F1" 0.6 --> 0.9
    line [0.785, 0.817, 0.819, 0.833, 0.856, 0.865]
    line [0.635, 0.689, 0.715, 0.741, 0.766, 0.776]
```

Le CNN entraîné sur 2 % des bons spectres dépasse déjà la forêt entraînée sur 100 % (0,785 contre 0,776). L'erreur du CNN sur Teff descend de 54 K à 27 K entre 2 et 50 %, puis ne bouge plus. La forêt gagne encore entre 50 et 100 %. Le macro F1 du CNN à 100 % (0,865) diffère du run principal (0,848) : la section 6.1 mesure ce bruit des graines.

### 5.6 Temps de calcul

| Étape | Bons spectres (s) | Tous les spectres (s) |
|---|---:|---:|
| Téléchargement des 270 312 spectres manquants | 21 000 env. (une fois) | |
| PCA et t-SNE | 539 (une fois) | |
| Forêt, classes | 99 | 131 |
| Forêt, régression | 308 | 486 |
| CNN, classes (30 époques) | 841 | 1 228 |
| CNN, régression (30 époques) | 842 | 1 214 |
| Autoencodeur | 280 | 383 |
| Clustering sur PCA | 116 | 134 |
| Clustering sur le latent | 125 | 191 |
| **Total** | **7 206 (2 h)** | |

*Portable avec une RTX 3060 (6 Go), 16 threads, 15 Go de mémoire.* Le CNN domine le temps de calcul.

---

## 6. Choix des paramètres et tests de sensibilité

Chaque valeur du projet a une raison, listée dans [`docs/choix_parametres.md`](docs/choix_parametres.md). Les tests changent un paramètre à la fois (`scripts/11_sensitivity.py`).

| Paramètre | Valeur | Raison |
|---|---|---|
| SNR d'un bon spectre | 70 | seuil courant des études ASPCAP |
| Découpage | 70 / 15 / 15 | écart type de l'accuracy de test sous 0,002 |
| Seuils de log g | 4,0 et 3,5 | définitions physiques, non optimisées |
| PCA du clustering | 10 axes | 74,7 % de la variance, 77,0 % à 20 axes |
| k | 3 | silhouette maximale |
| CNN | noyau 7, 3 blocs | un noyau couvre une raie |
| Époques | 30, 100 en régression (section 6.2) | meilleure époque de validation gardée |
| Taux d'apprentissage | 0,001 | défaut d'Adam |
| Forêt | 300 arbres, 50 axes | le gain s'arrête vers quelques centaines d'arbres |

![Sensibilité données](report/figures/sens_data.png)
*Figure 8. Effet du seuil de SNR, de la part de mauvais spectres et de la taille d'entraînement.*

- **Seuil de SNR.** 70 donne le meilleur macro F1 (0,866). Pour la régression, 50 fait mieux (26,6 K contre 28,3 K).
- **Mauvais spectres.** 25 % de mauvais spectres font gagner 0,006 de macro F1 sur `test_good` et 0,015 sur `test_all`. En régression, ils nuisent aux bons spectres (29 à 42 K à 50 %).
- **Taille.** Le CNN entraîné sur 2 % des bons spectres (3 450) atteint 0,785 de macro F1, plus que la forêt sur 100 % (0,776).

![Sensibilité hyperparamètres](report/figures/sens_hyper.png)
*Figure 9. Courbes du CNN sur 50 époques, taux d'apprentissage, arbres et composantes de la forêt.*

- **Époques.** En classification, la perte de validation atteint son minimum à l'époque 16, puis remonte. En régression, elle baisse encore à 50 : Teff vaut 25,0 K à l'époque 50 contre 35,3 K à 30. La section 6.2 prolonge la régression à 100 époques.
- **Forêt.** 10 composantes donnent 0,803 de macro F1, contre 0,776 avec 50. Avec trop d'axes, chaque arbre tire surtout du bruit.

![Sensibilité clustering](report/figures/sens_cluster.png)
*Figure 10. ARI selon le nombre de composantes et selon k, sur 10 graines.*

Sur 10 graines, l'ARI médian du GMM vaut 0,41 à k = 3. À k = 4, la médiane tombe à 0,10 et le maximum atteint 0,30. Aucune graine ne dépasse 0,30 à k = 4.

### 6.1 Bruit des graines et meilleurs réglages

Le même CNN relancé avec les graines 42, 43 et 44 donne 0,852 ± 0,010 de macro F1. Un écart de 0,01 entre deux runs uniques ne prouve rien. Les réglages finaux sont choisis sur la validation, puis mesurés une seule fois sur le test avec 3 graines (`scripts/12_best.py`).

```mermaid
flowchart LR
    T["Tests de sensibilité<br/>un paramètre à la fois"] --> V["Choix sur la validation<br/>perte minimale"]
    V --> B["Meilleurs réglages<br/>CNN classes : SNR 70, 50 % mauvais<br/>CNN régression : SNR 50, 0 % mauvais<br/>Forêt : 10 axes"]
    B --> TE["Test, une seule fois<br/>3 graines par CNN"]
```

| Modèle | Mesure | Référence | Meilleurs réglages |
|---|---|---:|---:|
| CNN, 3 graines | macro F1, `test_good` | 0,852 ± 0,010 | **0,865 ± 0,000** |
| | macro F1, `test_all` | 0,822 ± 0,009 | **0,847 ± 0,001** |
| | macro F1, SNR < 70 | 0,642 ± 0,023 | **0,738 ± 0,022** |
| | erreur Teff (K) | 26,3 ± 1,6 | 26,6 ± 1,0 |
| | erreur log g (dex) | 0,061 ± 0,002 | **0,055 ± 0,004** |
| Forêt | macro F1, `test_good` | 0,776 | **0,803** |
| | macro F1, SNR < 70 | 0,614 | **0,683** |
| | erreur log g (dex) | 0,106 | **0,085** |
| GMM, k = 3 | ARI | 0,41 (graine 42) | 0,43 (graine 4, BIC minimale) |

![Meilleurs réglages](report/figures/sens_best.png)
*Figure 11. Référence (gris) contre meilleurs réglages (bleu). Barres : écart type sur 3 graines.*

Le gain le plus net porte sur les spectres de SNR < 70 : 0,738 contre 0,642 pour le CNN.

### 6.2 CNN de régression sur 100 époques

Le test `epochs` montrait une perte de validation encore en baisse à l'époque 50. `scripts/15_regress_long.py` reprend les meilleurs réglages de la régression (SNR ≥ 50, sans mauvais spectres, 177 097 étoiles) et entraîne le même CNN sur 100 époques, avec les graines 42, 43 et 44. Le modèle gardé reste celui de la meilleure époque de validation. Rien d'autre ne change : même architecture, même taux 0,001, pas de décroissance du taux.

| Mesure | 30 époques (section 6.1) | 100 époques | Gain |
|---|---:|---:|---:|
| Teff, `test_good` (K) | 26,6 ± 1,0 | **22,1 ± 0,4** | −17 % |
| log g, `test_good` (dex) | 0,055 ± 0,004 | **0,045 ± 0,002** | −18 % |
| [M/H], `test_good` (dex) | 0,021 ± 0,003 | **0,017 ± 0,001** | −17 % |
| Teff, `test_all` (K) | 85,4 ± 0,7 | **79,4 ± 1,2** | −7 % |
| log g, `test_all` (dex) | 0,087 ± 0,004 | **0,076 ± 0,001** | −12 % |
| Teff, SNR < 70 (K) | 283 ± 4 | **275 ± 7** | −3 % |

*Erreurs absolues moyennes, moyenne ± écart type sur 3 graines.*

![Régression sur 100 époques](report/figures/regress_long.png)
*Figure 12. CNN de régression sur 100 époques. Trait fin : chaque époque. Trait épais : le modèle gardé si l'entraînement s'arrêtait à cette époque. Pointillé : 30 époques.*

Dans ces mêmes runs, le modèle gardé à 30, 50 et 100 époques donne :

| Arrêt | Teff (K) | log g (dex) | [M/H] (dex) |
|---:|---:|---:|---:|
| 30 | 29,4 ± 2,2 | 0,057 ± 0,004 | 0,020 ± 0,001 |
| 50 | 24,6 ± 0,9 | 0,054 ± 0,003 | 0,019 ± 0,001 |
| 100 | **22,1 ± 0,4** | **0,045 ± 0,002** | **0,017 ± 0,001** |

- La meilleure époque de validation tombe à 94, 95 et 100. Le modèle n'a pas fini d'apprendre à 100 époques.
- L'écart entre graines baisse de 1,0 K à 0,4 K sur Teff. Un entraînement plus long rend le résultat moins dépendant de la graine.
- Le gain se concentre sur les bons spectres. Sur les spectres de SNR < 70, l'erreur reste dominée par le bruit et les étiquettes (section 5.4).
- La perte de validation saute d'une époque à l'autre, jusqu'à un facteur 2. Avec un taux constant, l'époque gardée dépend de ce bruit. Une décroissance du taux réduirait ces sauts.
- Coût : 28 s par époque, 47 min par graine, 2 h 22 pour les 3 graines.

---

## 7. Red clump contre RGB

Le CNN à quatre classes ne retrouve que 58 % du red clump. Nous entraînons un classifieur binaire sur les géantes à bon spectre : 111 992 en entraînement, 24 221 en test, dont 3 500 du red clump. La perte pondère chaque classe par l'inverse de sa fréquence. Le seuil maximise le F1 du red clump sur la validation.

| Modèle | F1 red clump | Rappel | Précision | AUC |
|---|---:|---:|---:|---:|
| Forêt, Teff, log g, [M/H] | **0,689** | **0,88** | 0,57 | 0,943 |
| Forêt, avec [C/Fe], [N/Fe] | 0,687 | 0,85 | 0,57 | 0,943 |
| Forêt, 10 composantes PCA | 0,646 | 0,82 | 0,53 | 0,931 |
| CNN, 3 graines | 0,683 ± 0,004 | 0,87 | 0,56 | 0,943 |

Le rappel passe de 58 % à 87 %. La précision plafonne à 0,57 pour tous les modèles. Ce plafond commun pointe vers les étiquettes : le catalogue red clump choisit ses étoiles par des coupes réglées pour la pureté.

### 7.1 Étiquettes d'astérosismologie : APOKASC-3

APOKASC-3 (Pinsonneault et al. 2025) donne l'état évolutif de 15 808 géantes du champ Kepler. L'écart de période des modes mixtes sépare un cœur d'hélium inerte (RGB) d'un cœur qui brûle l'hélium (red clump). 10 430 de ces étoiles figurent dans nos spectres. Nous gardons les 7 494 dont l'état vient de l'astérosismologie seule, pour éviter un test circulaire.

```mermaid
flowchart LR
    subgraph CAT["Étiquette du catalogue"]
        CRC["Red clump<br/>1 879"]
        CRGB["RGB<br/>5 615"]
    end
    subgraph SEIS["État d'astérosismologie"]
        SRC["Red clump<br/>3 156"]
        SRGB["RGB<br/>4 338"]
    end
    CRC -- "1 722" --> SRC
    CRC -- "157" --> SRGB
    CRGB -- "1 434 manqués" --> SRC
    CRGB -- "4 181" --> SRGB
```

Le catalogue a raison 92 fois sur 100 quand il dit red clump. Il manque 45 % des vraies étoiles du red clump et les étiquette RGB.

Test : 844 géantes de test avec un état d'astérosismologie, dont 339 du red clump.

| Modèle | Étiquettes d'entraînement | F1 red clump | Rappel | Précision | AUC |
|---|---|---:|---:|---:|---:|
| Catalogue red clump | aucune | 0,675 | 0,53 | 0,92 | 0,751 |
| Forêt, paramètres + C, N | catalogue (111 769) | 0,933 | 0,96 | 0,91 | 0,972 |
| CNN | catalogue (111 992) | 0,933 ± 0,012 | 0,95 | 0,92 | 0,983 |
| Forêt, paramètres + C, N | sismologie (5 272) | 0,960 | 0,97 | 0,95 | 0,992 |
| CNN | sismologie (5 284) | 0,971 ± 0,009 | **0,99** | 0,96 | 0,992 |
| **CNN affiné** | catalogue puis sismologie | **0,978 ± 0,003** | **0,99** | **0,97** | **0,996** |

```mermaid
xychart-beta
    title "F1 red clump jugé par l'astérosismologie"
    x-axis ["Catalogue", "CNN catalogue", "Forêt sismo", "CNN sismo", "CNN affiné"]
    y-axis "F1" 0.6 --> 1.0
    bar [0.675, 0.933, 0.960, 0.971, 0.978]
```

Trois résultats :

1. Nos modèles marchaient déjà. Le CNN entraîné sur le catalogue passe d'un F1 de 0,68 contre le catalogue à 0,93 contre l'astérosismologie. Ses fausses alertes étaient en majorité de vraies étoiles du red clump.
2. De bonnes étiquettes valent plus que beaucoup d'étiquettes. 5 284 étoiles d'astérosismologie portent le CNN à 0,971, avec vingt fois moins de données.
3. Le spectre complet bat les cinq paramètres ASPCAP (0,971 contre 0,960). Le CNN affiné atteint 99 % de rappel et 3 % de contamination, l'ordre de grandeur de Ting, Hawkins et Rix (2018).

Réserves : le test compte 844 étoiles, et toutes viennent du champ Kepler, surtout du disque mince.

---

## 8. Discussion

| | Constat |
|---|---|
| **Tient** | Séparation naines / géantes sans étiquette. Supériorité du CNN en classification. Effet opposé des mauvais spectres sur classification et régression. |
| **Plafonne** | Le CNN à quatre classes confond 42 % du red clump avec la RGB. La cause : les étiquettes du catalogue. Contre l'astérosismologie, le F1 atteint 0,978. |
| **Fragile** | Le GMM à k = 4. Sur 10 graines, l'ARI ne dépasse pas 0,30. k = 3 reste le choix robuste. |
| **Règle pratique** | Entraîner la régression sur les spectres bruités seulement avec des étiquettes fiables. Les étiquettes FPARAM donnent 882 K d'erreur. |

## 9. Limites et travaux futurs

Limites :

- Les étiquettes viennent d'ASPCAP. Le CNN apprend à reproduire ASPCAP.
- Hors des sections 6.1 et 7, chaque CNN s'entraîne avec une seule graine.
- Le CNN de régression sur 100 époques a sa meilleure époque entre 94 et 100 : il n'a pas convergé.
- La PCA s'ajuste sur 50 000 spectres et HDBSCAN tourne sur 50 000 étoiles.

Travaux futurs, par priorité :

1. Entraîner la régression sur tous les spectres avec les étiquettes ASPCAP seulement.
2. Entraîner le CNN de régression au-delà de 100 époques, avec une décroissance du taux d'apprentissage.
3. Répéter chaque entraînement avec 5 graines.
4. Vérifier le classifieur red clump / RGB hors du champ Kepler, avec TESS ou K2.
5. Réétiqueter les 137 454 RGB avec le CNN affiné, puis réentraîner le CNN à quatre classes.

## 10. Conclusion

Sur 370 000 spectres APOGEE, le spectre seul sépare naines et géantes sans étiquette : un GMM à trois groupes atteint un ARI de 0,41 et place 73 024 des 73 040 naines dans un même groupe. Avec étiquettes, le CNN 1-D atteint un macro F1 de 0,865 sur 3 graines. Entraîné sur 100 époques, le CNN de régression atteint 22,1 ± 0,4 K sur Teff. Le CNN entraîné sur 2 % des bons spectres dépasse déjà la forêt entraînée sur 100 %. La limite red clump / RGB venait des étiquettes : avec APOKASC-3, un CNN sépare les deux classes avec un F1 de 0,978 et 3 % de contamination.

---

## Reproduire

### Installation

```bash
pip install -r requirements.txt
export STARS_DATA=/chemin/vers/un/disque/avec/60Go   # environ 45 Go de données
```

Sans `STARS_DATA`, les scripts écrivent dans `data/` à la racine du projet.

### Pipeline

```bash
python scripts/01_get_catalogs.py        # catalogues APOGEE, red clump, OCCAM (4,2 Go)
python scripts/02_select_stars.py        # échantillon, qualité, étiquettes, split
python scripts/03_download_spectra.py    # spectres, reprise automatique
python scripts/04_explore.py             # PCA, t-SNE, UMAP
python scripts/05_baseline.py            # forêt aléatoire
python scripts/06_train.py --model cnn   # CNN 1-D, reprise automatique
python scripts/07_cluster.py             # K-means, GMM, HDBSCAN
python scripts/08_report.py              # tableaux de résultats
python scripts/09_quality_by_snr.py      # performance par tranche de SNR
python scripts/10_label_source.py        # erreurs par origine d'étiquette
python scripts/11_sensitivity.py cluster_dims cluster_k rf_trees rf_pca size epochs snr badfrac lr   # environ 8 h
python scripts/12_best.py                # meilleurs réglages, 3 graines, environ 4 h
python scripts/13_rc_rgb.py              # classifieur red clump / RGB, environ 40 min
python scripts/14_apokasc.py             # étiquettes APOKASC-3 depuis VizieR, environ 45 min
python scripts/15_regress_long.py        # CNN de régression sur 100 époques, 3 graines, environ 2 h 30
python report/make_figures.py
python report/make_sensitivity_figures.py
```

Toute l'expérience qualité d'un coup : `bash scripts/run_quality_experiment.sh`. Les étapes 01, 03 et 06 reprennent où elles se sont arrêtées. Si `data.sdss.org` ne répond plus, `STARS_SAS=https://dr17.sdss.org` bascule sur le miroir officiel.

### Organisation

```
stars/      code partagé : configuration, catalogue, spectres, modèles, entraînement, figures
scripts/    une étape du pipeline par script, numérotés dans l'ordre d'exécution
docs/       explication de chaque script et des choix de données (commencez par 00_vue_ensemble.md)
results/    tableaux de résultats, tests de sensibilité, red clump / RGB, APOKASC-3, régression sur 100 époques
report/     rapport LaTeX, présentation beamer, figures, vue 3D des groupes
```

## Références

- Abdurro'uf et al. (2022). The Seventeenth Data Release of the Sloan Digital Sky Surveys. ApJS, 259, 35.
- Majewski, S. R. et al. (2017). The Apache Point Observatory Galactic Evolution Experiment (APOGEE). AJ, 154, 94.
- García Pérez, A. E. et al. (2016). ASPCAP: The APOGEE Stellar Parameter and Chemical Abundances Pipeline. AJ, 151, 144.
- Garcia-Dias, R. et al. (2018). Machine learning in APOGEE: Unsupervised spectral classification with K-means. A&A, 612, A98.
- Bovy, J. et al. (2014). The APOGEE Red-clump Catalog. ApJ, 790, 127.
- Myers, N. et al. (2022). The Open Cluster Chemical Abundances and Mapping Survey VI. AJ, 164, 85.
- Weinberg, D. H. et al. (2019). Chemical Cartography with APOGEE: Multi-element Abundance Ratios. ApJ, 874, 102.
- van der Maaten, L. et Hinton, G. (2008). Visualizing Data using t-SNE. JMLR, 9, 2579-2605.
- Campello, R. J. G. B., Moulavi, D. et Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. PAKDD.
- Hubert, L. et Arabie, P. (1985). Comparing partitions. Journal of Classification, 2, 193-218.
- Ting, Y.-S., Hawkins, K. et Rix, H.-W. (2018). A Large and Pristine Sample of Standard Candles across the Milky Way. ApJL, 858, L7.
- Pinsonneault, M. H. et al. (2025). APOKASC-3. ApJS, 276, 69.
