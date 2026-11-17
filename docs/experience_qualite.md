# Expérience qualité (`run_quality_experiment.sh`)

## Question

Le superviseur demande l'impact d'inclure toutes les données plutôt que les seuls bons spectres. Le script répond pour chaque méthode du projet.

## Ce que lance le script

Pour `train_on` = `good`, puis `all` :

| Étape | Runs |
|---|---|
| forêt aléatoire | `rf_class_*`, `rf_regress_*` |
| CNN | `cnn_class_*`, `cnn_regress_*` |
| autoencodeur | `autoencoder_none_*` |
| clustering PCA | `cluster_pca_*` |
| clustering latent | `cluster_latent_*` |

Puis `08_report.py` rassemble tout.

## Commande

```bash
bash scripts/run_quality_experiment.sh
```

Le script s'arrête à la première erreur. Relancez-le : les CNN et autoencodeurs déjà finis rechargent leur checkpoint et passent directement à l'évaluation. Les forêts aléatoires et le clustering sont recalculés.

## Plan d'évaluation

| Entraînement | Test | Question |
|---|---|---|
| good | test_good | performance de référence sur données propres |
| all | test_good | les mauvais spectres dégradent-ils l'apprentissage ? |
| good | test_all | un modèle propre résiste-t-il au bruit ? |
| all | test_all | performance sur des données réelles |

Pour le clustering, `--subset` change à la fois les étoiles regroupées et celles évaluées. Comparez `ari_class` et `occam_purity` entre `cluster_*_good` et `cluster_*_all`.

## Pistes pour le rapport

- Tracer le macro F1 en fonction du SNR sur `test_all`, par tranches (5 à 30, 30 à 70, plus de 70).
- Utiliser `bad_pixel_frac` (dans le catalogue chargé par `spectra.load_dataset`) comme mesure continue de qualité.
- Vérifier dans `figures/explore/tsne.png` si les mauvais spectres forment un groupe à part.
