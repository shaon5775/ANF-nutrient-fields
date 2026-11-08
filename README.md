# Atwater Nutrient Fields (ANF) and the Ingredient Atwater Field (IAF)

Code and result files for the paper

> **Atwater Nutrient Fields: Physically Consistent Nutrient and Ingredient Maps Learned from Dish-Level Labels**
> Shaon Bandyopadhyay, Dipankar Dutta, Sanjoy Pratihar

ANF is a small head on frozen DINOv2 patch features. For every cell of a 16×16 grid it predicts the grams of food and
their split into fat, carbohydrate, protein and a remainder. Dish totals are sums over cells, and energy follows from
three learned Atwater factors, so energy, mass and the three macronutrients always agree. IAF replaces the free
composition with a distribution over 40 ingredient groups whose composition comes from a fixed food table, and learns
ingredient-gram maps from dish-level ingredient weights only.

## Repository layout

```
experiments/   the exact scripts that produced every number in the paper (one script per experiment)
analysis/      summaries of experiment outputs, the ingredient-name-to-group matching, error analysis
figures/       scripts that draw the paper and supplement figures from the result files
data/          ingredient_group_map.json (Nutrition5k ingredient name -> food group, built by analysis/map_ingredients.py)
results/       the outputs of our runs (per-dish predictions, tables, learned factors, summaries) for checking
```

No dataset files are included. Every experiment script downloads the public data itself.

## Data

| Dataset | Source | Licence | Use |
|---|---|---|---|
| Nutrition5k (Thames et al., CVPR 2021) | `storage.googleapis.com/nutrition5k_dataset` (official RGB train/test split) | CC BY 4.0 | training and testing |
| NutritionVerse-Real (Tai et al., 2024) | Kaggle dataset `nutritionverse/nutritionverse-real` | CC BY-NC-SA 4.0 | 5-fold cross-validation grouped by dish (non-commercial research only) |
| Nutrition5k ingredient masks (Hand et al., 2026) | `github.com/liamhand/ingredient-level-nutrition-regression` | CC BY 4.0 | evaluation of IAF maps only, never used for training |

## Environment

All experiments ran as Kaggle notebooks on one Tesla T4 GPU (Python 3.13, PyTorch 2.11, CUDA; see `results/FA/env.json`).
Backbones are `facebook/dinov2-base` and `facebook/dinov2-large` from Hugging Face.

```
pip install -r requirements.txt
```

## How to run

Each file in `experiments/` is a complete, self-contained script (data download, feature extraction, training,
evaluation, output files). The easiest way is to paste it into a Kaggle notebook with a GPU and run it; outputs are
written to `/kaggle/working/<name>/`. To run elsewhere, set an output folder:

```
DRY=./out_full python experiments/01_full_study.py
```

`SMOKE=1` runs a tiny version (a few dishes, 3 epochs) to check that everything works. The NutritionVerse-Real download
uses the public Kaggle dataset URL and works inside Kaggle; outside Kaggle you may need to be signed in to Kaggle.

## Which script gives which result

| Paper item | Script | Our output |
|---|---|---|
| Main Nutrition5k table (both backbones, five seeds 0–4), consistency gap | `experiments/01_full_study.py` | `results/FA/tables/T2_main_n5k.csv`, `results/FA/results.json` |
| Learning curves (5–50% of training dishes, subsets 200–202) | `experiments/01_full_study.py` | `results/FA/tables/T3_learning_curve.csv` (Supplementary Table S2) |
| NutritionVerse-Real table and cross-dataset row | `experiments/01_full_study.py` | `results/FA/tables/T5_nvr.csv`, `T5b_cross_n5k_to_nvr.csv` |
| Learned energy factors | `experiments/01_full_study.py` | `results/FA/factors.json`, `tables/T4_factors.csv` |
| Ablation table, grid sizes, cost | `experiments/01_full_study.py` | `results/FA/tables/T6_ablations.csv`, `T7_cost.csv` |
| Error analysis by dish mass and ingredient count (Supplementary Table S3) | `01_full_study.py` + `analysis/error_analysis_extra.py` | `results/FA/tables/T8_error_analysis.csv`, `T8c_error_extra.json` |
| Independent repetition of the learning curves (gains 12.1 / 8.4 / 6.2% at 5 / 10 / 25%) and of NutritionVerse-Real | `experiments/02_learning_curve_repeat.py` | printed and saved by the script |
| Deletion test and border energy share (Supplementary Fig. S1) | `experiments/03_map_checks.py` | printed and saved by the script; figure: `figures/make_fig_deletion.py` |
| Factors started away from 9/4/4 | `experiments/04_factor_starts.py`, `analysis/analyse_factor_starts.py` | `results/factor_starts_summary.json` |
| Cheap partial labels (limitation vi) | `experiments/05_cheap_partial_labels.py`, `analysis/analyse_cheap_labels.py` | `results/cheap_labels_summary.json` |
| IAF table and localisation (H − H0, pointing) on 980 regions | `experiments/06_ingredient_field.py`, `analysis/analyse_ingredient_field.py` | `results/ingredient_field_summary.json`, `ingredient_field_by_seed.json` |
| IAF with 10% and 5% of the training dishes | `experiments/07_ingredient_field_few_labels.py` | `results/ingredient_field_few_labels_summary.json` |
| Ingredient-name-to-group matching (167 of 250 names, 93% of training mass) | `analysis/map_ingredients.py` | `data/ingredient_group_map.json` |
| Paper figures: method, learning curves, factors, energy maps, IAF maps | `figures/make_fig_method.py`, `make_figs_paper.py`, `make_fig_iaf.py` | `figures_out/` |
| Predicted-vs-weighed scatter, consistency histograms, training curves, error by dish mass | `experiments/01_full_study.py` | written to its `figures/` output folder |

Notes:
* `01_full_study.py` also writes a first deletion curve that replaced cells by the mean training feature. That feature
  resembles food, so this curve is not used; the deletion results in the paper come from `03_map_checks.py`, which
  replaces cells by the mean feature of empty-plate cells.
* `make_figs_paper.py` and `make_fig_method.py` need `thumbs_n5k_test.npz` (small test images), which is written by
  `01_full_study.py` and is not included here because it contains dataset images.
* Splits: Nutrition5k uses the official `rgb_train_ids.txt` / `rgb_test_ids.txt`; NutritionVerse-Real folds are assigned
  per dish with `numpy.random.default_rng(0)`. All seeds are written in the scripts.
* GPU results can differ in the last digit between runs and hardware.
* Inside the scripts, section markers such as "FA", "DX", "DY", "DZ", "EA", "EB" and "EC" are our internal run names for
  scripts 01–07.

## Citation

If you use this code, please cite the paper (full reference to be added after publication).

## Licence

Code: MIT (see `LICENSE`). The datasets keep their own licences listed above.
