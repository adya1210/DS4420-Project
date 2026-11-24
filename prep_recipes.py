# Builds one feature row per recipe from Table_S1.txt (Bonatto 2025) -> data/recipe_vectors.csv.gz
# usage: python prep_recipes.py path/to/Table_S1.txt
#
# features: brewing stats (OG, FG, ABV, IBU, SRM, ...), malt bill as % of total weight,
# hops in g/L by use and by variety. Region + the paper's cluster columns are left out
# since they come from the style label.
import re
import sys

import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/raw/Table_S1.txt"
OUT = "data/recipe_vectors.csv.gz"
TOP_FERMENTABLES = 60
TOP_HOPS = 60
HOP_USE = {"Dry Hop (High Krausen)": "Dry Hop", "Hop Stand": "Whirlpool"}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def build(src=SRC):
    r = pd.read_csv(src, sep="\t")
    # the raw table has a row for every (hop addition, fermentable) pair and `time`
    # numbers the fermentables, so split it back into a hop table and a malt table
    r["_tmin"] = r.groupby("recipe_code")["time"].transform("min")
    hops = r[r["time"] == r["_tmin"]].copy()
    ferm = r.drop_duplicates(["recipe_code", "time"])

    base = r.drop_duplicates("recipe_code").set_index("recipe_code")
    X = pd.DataFrame(index=base.index)
    X["category"] = base["Category.x"]
    X["style"] = base["style"]
    X["og"] = base["SG"]
    X["fg"] = base["FG"]
    X["abv"] = base["ABV"]
    X["srm"] = base["SRM"]
    X["attenuation"] = base["apparent_attenuation_percent"]
    X["efficiency"] = base["efficiency_porcent"]
    X["cold_fermented"] = (base["Fermentation_type"] == "Cold fermented").astype(int)
    X["ibu"] = hops.groupby("recipe_code")["IBU"].sum()
    vol = base["batch_vol_liter"]

    # Malt bill (% of fermentable kg)
    total_kg = ferm.groupby("recipe_code")["quantities_kg"].sum()
    by_type = ferm.pivot_table(index="recipe_code", columns="Type", values="quantities_kg",
                               aggfunc="sum", fill_value=0)
    X = X.join(by_type.div(total_kg, axis=0).mul(100).add_prefix("malttype_pct_").rename(columns=slug))
    top_f = ferm.drop_duplicates(["recipe_code", "Fermentable"])["Fermentable"].value_counts().head(TOP_FERMENTABLES).index
    ferm = ferm.assign(F=ferm["Fermentable"].where(ferm["Fermentable"].isin(top_f), "other"))
    by_f = ferm.pivot_table(index="recipe_code", columns="F", values="quantities_kg", aggfunc="sum", fill_value=0)
    X = X.join(by_f.div(total_kg, axis=0).mul(100).add_prefix("malt_pct_").rename(columns=slug))

    # Hops (g per litre) by use and by variety
    hops["use"] = hops["hops_use_1"].replace(HOP_USE)
    hops["g_per_l"] = hops["quantity_used"] / hops["recipe_code"].map(vol)
    by_use = hops.pivot_table(index="recipe_code", columns="use", values="g_per_l", aggfunc="sum", fill_value=0)
    X = X.join(by_use.add_prefix("hops_gpl_use_").rename(columns=slug))
    X["hops_gpl_total"] = hops.groupby("recipe_code")["g_per_l"].sum()
    X["n_hop_varieties"] = hops.groupby("recipe_code")["hops_name_1"].nunique()
    X["n_fermentables"] = ferm.groupby("recipe_code")["Fermentable"].nunique()
    top_h = hops.drop_duplicates(["recipe_code", "hops_name_1"])["hops_name_1"].value_counts().head(TOP_HOPS).index
    hops["H"] = hops["hops_name_1"].where(hops["hops_name_1"].isin(top_h), "other")
    by_h = hops.pivot_table(index="recipe_code", columns="H", values="g_per_l", aggfunc="sum", fill_value=0)
    X = X.join(by_h.add_prefix("hop_gpl_").rename(columns=slug))

    feat = X.columns.drop(["category", "style"])
    X[feat] = X[feat].fillna(0).replace([np.inf, -np.inf], 0)
    return X.reset_index()


if __name__ == "__main__":
    X = build()
    X.to_csv(OUT, index=False)
    print(f"{len(X):,} recipes x {X.shape[1] - 3} features -> {OUT}")
    print(X.describe().T[["mean", "50%", "max"]].round(2).sort_values("max", ascending=False).head(12))
