# DS4420 Project: Predicting Beer Style and Ratings from Recipes

Authors: TODO (both names)

## Data
| Source | Used for | How to get it |
|---|---|---|
| Brewing recipe table (`Table_S1.txt`, ~62k recipes with BJCP style labels) | Method 1 | Table S1 of Bonatto, D. (2025), arXiv:2505.17039 (recipes from Brewer's Friend). Source link: TODO. Save as `data/raw/Table_S1.txt`. |
| BrewDog DIY Dog recipes (`data/diydog/*.json`) | Method 2 | Included; from github.com/alxiw/punkapi |
| BeerAdvocate reviews (Kaggle) | Method 2 | Downloaded automatically by `prep_brewdog.py` (kagglehub) |

Processed data (`data/recipe_vectors.csv.gz`, `data/brewdog_reviews_recipes.csv`, `data/diydog_recipe_features.csv`) is included, so the prep steps can be skipped.

## Run
```
pip install -r requirements.txt
python prep_recipes.py [path/to/Table_S1.txt]   # -> data/recipe_vectors.csv.gz
python prep_brewdog.py                          # -> data/brewdog_reviews_recipes.csv, data/diydog_recipe_features.csv
python mlp_style.py [--quick]                   # Method 1 -> results/, figures/  (--quick reuses results/hparam_search.csv)
```

## Methods
1. **MLP (PyTorch)**: predicts a recipe's BJCP category (34 classes) and style (120 classes) from 155 recipe features.
2. **Bayesian regression (NumPy, implemented manually)**: TODO.
