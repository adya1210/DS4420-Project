# Method 1: MLP that predicts a beer's BJCP category (34) and style (120) from the recipe.
# Reads data/recipe_vectors.csv.gz (from prep_recipes.py).
#
# 1. tune the MLP on the validation set (category only)
# 2. compare majority class / linear model / MLP on the test set, 5 seeds each
#    (also tried class-weighted loss for the MLP since some categories are rare)
# 3. confusion matrix + training curves for the final model

import itertools
import json
import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

SPLIT_SEED = 42
SEEDS = [0, 1, 2, 3, 4]
FINAL_WEIGHTING = "none"  # sqrt weighting only helped macro-F1 by ~0.3 and cost 3 pts of accuracy
SEARCH = dict(hidden=[(256, 128), (512, 256), (512, 256, 128)],
              p_drop=[0.2, 0.3, 0.5],
              lr=[1e-3, 3e-3],
              wd=[1e-4, 1e-3])
QUICK = "--quick" in sys.argv

for d in ("results", "figures", "models"):
    os.makedirs(d, exist_ok=True)

# plot colours
BLUE, ORANGE = "#2a78d6", "#eb6834"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SEQ_BLUES = ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]


df = pd.read_csv("data/recipe_vectors.csv.gz")
feat_cols = [c for c in df.columns if c not in ("recipe_code", "category", "style")]
SKEWED = [c for c in feat_cols if c.startswith(("hop", "ibu", "srm"))]

# 70/15/15 split, stratified by style so rare styles show up in every set
idx = np.arange(len(df))
tr_idx, tmp = train_test_split(idx, test_size=0.30, stratify=df["style"], random_state=SPLIT_SEED)
va_idx, te_idx = train_test_split(tmp, test_size=0.50, stratify=df["style"].iloc[tmp], random_state=SPLIT_SEED)


def preprocess(X):
    # hop amounts / IBU / colour are very right-skewed so log them first,
    # then clip outliers and standardize (using training rows only)
    X = X.copy()
    X[SKEWED] = np.log1p(X[SKEWED].clip(lower=0))
    lo, hi = X.iloc[tr_idx].quantile(0.001), X.iloc[tr_idx].quantile(0.999)
    X = X.clip(lo, hi, axis=1)
    mu, sd = X.iloc[tr_idx].mean(), X.iloc[tr_idx].std().replace(0, 1)
    return ((X - mu) / sd).to_numpy(np.float32)


X = preprocess(df[feat_cols])


def labels(target):
    classes = np.array(sorted(df[target].unique()))
    y = pd.Categorical(df[target], categories=classes).codes.astype(np.int64)
    return classes, y


## model

class MLP(nn.Module):
    def __init__(self, d_in, n_out, hidden=(256, 128), p_drop=0.3):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def class_weights(y, n_classes, scheme):
    # "balanced" = 1/freq, "sqrt" = 1/sqrt(freq), scaled so the average weight is 1
    if scheme == "none":
        return None
    freq = np.bincount(y, minlength=n_classes).astype(float)
    w = 1 / np.maximum(freq, 1)
    if scheme == "sqrt":
        w = np.sqrt(w)
    w = w / (w * freq).sum() * freq.sum()
    return torch.tensor(w, dtype=torch.float32)


def train(model, Xtr, ytr, Xva, yva, weight=None, lr=1e-3, wd=1e-4, epochs=100, batch=256, patience=8):
    # stops when val loss hasn't improved for `patience` epochs and goes back to the best epoch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    Xtr, ytr, Xva, yva = map(torch.tensor, (Xtr, ytr, Xva, yva))
    hist, best, best_state, bad = [], np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            va_logits = model(Xva)
            va_loss = loss_fn(va_logits, yva).item()
            va_pred = va_logits.argmax(1)
        hist.append(dict(epoch=ep + 1, train_loss=tot / len(Xtr), val_loss=va_loss,
                         val_acc=(va_pred == yva).float().mean().item(),
                         val_macro_f1=f1_score(yva, va_pred, average="macro")))
        if va_loss < best - 1e-4:
            best, bad = va_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)


def predict(model, Xs):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(Xs)).numpy()


def metrics(y_true, logits, k=3):
    pred = logits.argmax(1)
    topk = np.argsort(-logits, axis=1)[:, :k]
    return dict(accuracy=(pred == y_true).mean(),
                top3_accuracy=(topk == y_true[:, None]).any(1).mean(),
                macro_f1=f1_score(y_true, pred, average="macro"))


## 1. hyperparameter tuning (36 combos)

classes, y = labels("category")
if QUICK and os.path.exists("results/hparam_search.csv"):
    search = pd.read_csv("results/hparam_search.csv")
else:
    rows = []
    for hidden, p_drop, lr, wd in itertools.product(*SEARCH.values()):
        torch.manual_seed(0)
        m, h = train(MLP(X.shape[1], len(classes), hidden, p_drop),
                     X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], lr=lr, wd=wd)
        va = metrics(y[va_idx], predict(m, X[va_idx]))
        rows.append(dict(hidden=str(hidden), p_drop=p_drop, lr=lr, wd=wd, epochs=len(h),
                         val_accuracy=va["accuracy"], val_macro_f1=va["macro_f1"]))
        print("search", rows[-1])
    search = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
    search.to_csv("results/hparam_search.csv", index=False)
best = search.iloc[0]
BEST = dict(hidden=tuple(int(v) for v in best["hidden"].strip("()").split(",") if v.strip()),
            p_drop=float(best["p_drop"]), lr=float(best["lr"]), wd=float(best["wd"]))
print("\nBest MLP config (validation accuracy):", BEST)

## 2. test set results (same split, different seeds)

runs, keep = [], {}
for target in ["category", "style"]:
    classes, y = labels(target)
    ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]
    counts = np.bincount(ytr, minlength=len(classes))
    # baseline: always guess the most common class (ranked by count for top-3)
    maj_logits = np.tile(-np.argsort(np.argsort(-counts)).astype(float), (len(yte), 1))
    runs.append(dict(target=target, model="majority", seed=0, **metrics(yte, maj_logits)))

    configs = [("linear", "none")] + [("mlp", w) for w in ("none", "sqrt", "balanced")]
    for (kind, weighting), seed in itertools.product(configs, SEEDS):
        torch.manual_seed(seed)
        # linear = no hidden layer, i.e. multinomial logistic regression
        model = (nn.Linear(X.shape[1], len(classes)) if kind == "linear"
                 else MLP(X.shape[1], len(classes), BEST["hidden"], BEST["p_drop"]))
        lr, wd = (1e-3, 1e-4) if kind == "linear" else (BEST["lr"], BEST["wd"])
        model, hist = train(model, X[tr_idx], ytr, X[va_idx], yva,
                            weight=class_weights(ytr, len(classes), weighting), lr=lr, wd=wd)
        logits = predict(model, X[te_idx])
        runs.append(dict(target=target, model=kind, weighting=weighting, seed=seed,
                         epochs=len(hist), **metrics(yte, logits)))
        print(runs[-1])
        if kind == "mlp" and weighting == FINAL_WEIGHTING and seed == SEEDS[0]:
            keep[target] = (classes, yte, logits.argmax(1), hist)
            torch.save(model.state_dict(), f"models/mlp_{target}.pt")

runs = pd.DataFrame(runs)
runs["weighting"] = runs["weighting"].fillna("-")
runs.to_csv("results/test_runs_all_seeds.csv", index=False)
summary = (runs.groupby(["target", "model", "weighting"], sort=False)[["accuracy", "top3_accuracy", "macro_f1"]]
           .agg(["mean", "std"]))
summary.columns = [f"{m}_{s}" for m, s in summary.columns]
summary = summary.reset_index()
summary.to_csv("results/style_mlp_results.csv", index=False)
# "mean ± sd" strings for the report table
fmt = lambda r, m: f"{100 * r[m + '_mean']:.1f} ± {100 * np.nan_to_num(r[m + '_std']):.1f}"
table = pd.DataFrame({"target": summary.target, "model": summary.model, "weighting": summary.weighting,
                      **{m: summary.apply(fmt, axis=1, m=m) for m in ["accuracy", "top3_accuracy", "macro_f1"]}})
table.to_csv("results/style_mlp_results_table.csv", index=False)
print("\nTest set, mean ± sd over seeds (%):\n", table.to_string(index=False))

## 3. which categories does it get wrong? (category model, seed 0)

classes, yte_c, pred_c, hist = keep["category"]
support = np.bincount(yte_c, minlength=len(classes))
f1s = f1_score(yte_c, pred_c, average=None, labels=range(len(classes)))
per_class = pd.DataFrame(dict(category=classes, f1=f1s, test_support=support)).sort_values("f1")
per_class.to_csv("results/per_class_f1_category.csv", index=False)
cm = confusion_matrix(yte_c, pred_c, labels=range(len(classes)))
cm_norm = cm / cm.sum(1, keepdims=True)
# most common mix-ups (ignoring pairs with < 5 test beers)
conf = pd.DataFrame([(classes[i], classes[j], cm_norm[i, j], cm[i, j])
                     for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] >= 5],
                    columns=["true", "predicted", "share_of_true", "count"]).sort_values("share_of_true", ascending=False)
conf.head(15).to_csv("results/top_confusions_category.csv", index=False)
print("\nHardest categories:\n", per_class.head(8).round(3).to_string(index=False))
print("\nTop confusions:\n", conf.head(8).round(3).to_string(index=False))

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT})

# confusion matrix, biggest categories first
order = np.argsort(-support)
fig, ax = plt.subplots(figsize=(10, 9))
cmap = mcolors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)
im = ax.imshow(cm_norm[np.ix_(order, order)], cmap=cmap, vmin=0, vmax=1)
ax.set_xticks(range(len(classes)), [classes[i] for i in order], rotation=90, fontsize=7)
ax.set_yticks(range(len(classes)), [f"{classes[i]} (n={support[i]})" for i in order], fontsize=7)
ax.set_xlabel("Predicted category")
ax.set_ylabel("True category (test-set count)")
ax.set_title("MLP confusion matrix, BJCP category (row-normalised, test set)", loc="left", fontsize=11)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("Share of true category", color=MUTED)
cb.outline.set_visible(False)
fig.tight_layout()
fig.savefig("figures/confusion_category.png", dpi=200)
plt.close(fig)

# training curves
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax in axes:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Epoch")
axes[0].plot(hist.epoch, hist.train_loss, color=BLUE, lw=2, label="Train")
axes[0].plot(hist.epoch, hist.val_loss, color=ORANGE, lw=2, label="Validation")
axes[0].set_title("Cross-entropy loss" if FINAL_WEIGHTING == "none" else "Weighted cross-entropy loss", loc="left")
axes[0].legend(frameon=False)
axes[1].plot(hist.epoch, hist.val_acc, color=BLUE, lw=2, label="Accuracy")
axes[1].plot(hist.epoch, hist.val_macro_f1, color=ORANGE, lw=2, label="Macro-F1")
axes[1].set_title("Validation metrics", loc="left")
axes[1].legend(frameon=False)
fig.tight_layout()
fig.savefig("figures/training_curves.png", dpi=200)
plt.close(fig)

json.dump({"train": len(tr_idx), "val": len(va_idx), "test": len(te_idx), "n_features": len(feat_cols),
           "best_config": {k: list(v) if isinstance(v, tuple) else v for k, v in BEST.items()},
           "final_weighting": FINAL_WEIGHTING, "seeds": SEEDS}, open("results/run_info.json", "w"), indent=1)
