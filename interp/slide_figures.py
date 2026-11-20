"""Presentation figures (template palette: navy 1E2B42, gold C39D53, sage 7F9A85, brick AE5449; Helvetica Neue).
Usage: python slide_figures.py -> figures/slides/*.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent
R, G = HERE / "results", HERE.parent / "gpu_results" / "results"
OUT = HERE / "figures" / "slides"
OUT.mkdir(parents=True, exist_ok=True)
NAVY, GOLD, SAGE, BRICK, GREY, MUTED = "#1E2B42", "#C39D53", "#7F9A85", "#AE5449", "#A9AEB6", "#5B6472"
fam = "Helvetica Neue" if any("Helvetica Neue" in f.name for f in font_manager.fontManager.ttflist) else "Arial"
plt.rcParams.update({"font.family": fam, "font.size": 13, "axes.edgecolor": "#C9CCD1", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#E6E2D8", "grid.linewidth": 0.8, "legend.frameon": False,
                     "savefig.dpi": 220, "savefig.transparent": True, "axes.facecolor": "none"})
SH = {"motte_bailey": "Motte-and-bailey", "strawman": "Strawman"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=color, fontsize=12,
                fontweight="bold", va="center")


probe = pd.concat([pd.read_csv(R / f"probe_{e}.csv") for e in ["nli_pair", "plain_pair", "random_pair", "nli_b_only"]])
pg = probe.groupby(["encoder", "target", "shape", "segment", "layer"])[["auc", "auc_control"]].mean().reset_index()


def P(enc, target, shape, seg):
    return pg[(pg.encoder == enc) & (pg.target == target) & (pg["shape"] == shape) & (pg.segment == seg)].sort_values("layer")


# 1. trigger vs label decodability (NLI, claim-B tokens)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
for ax, shape in zip(axes, SH):
    t, l = P("nli_pair", "trigger", shape, "B"), P("nli_pair", "label", shape, "B")
    c = pg[(pg.target == "label") & (pg["shape"] == shape) & (pg.segment == "B")].groupby("layer").auc_control
    ax.fill_between(c.min().index, c.min(), c.max(), color="#E9E5DA", lw=0)
    ax.plot(t.layer, t.auc, color=BRICK, lw=3)
    ax.plot(l.layer, l.auc, color=NAVY, lw=3)
    label_end(ax, 24, t.auc.iloc[-1], "trigger", BRICK)
    label_end(ax, 24, l.auc.iloc[-1], "fallacy", NAVY)
    ax.text(1, 0.535, "shuffled-label control", color=MUTED, fontsize=10.5)
    ax.set_title(SH[shape], loc="left", color=NAVY, fontsize=14, fontweight="bold")
    ax.set_xlabel("layer"); ax.set_ylim(0.45, 1.0); ax.set_xlim(0, 28.5); ax.set_xticks([0, 6, 12, 18, 24])
axes[0].set_ylabel("probe AUC on claim-B tokens")
save(fig, "s_trigger_vs_label")

# 2. context needed (motte-and-bailey, B tokens): pair vs B-only vs random
fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
for ax, shape in zip(axes, SH):
    for enc, col, name in [("nli_pair", NAVY, "A + C + B"), ("nli_b_only", GOLD, "B alone"), ("random_pair", GREY, "random init")]:
        x = P(enc, "label", shape, "B")
        ax.plot(x.layer, x.auc, color=col, lw=3)
        label_end(ax, 24, x.auc.iloc[-1], name, col)
    ax.axhline(0.5, color="#C9CCD1", lw=1, ls=":")
    ax.set_title(SH[shape], loc="left", color=NAVY, fontsize=14, fontweight="bold")
    ax.set_xlabel("layer"); ax.set_ylim(0.42, 0.85); ax.set_xlim(0, 31); ax.set_xticks([0, 6, 12, 18, 24])
axes[0].set_ylabel("fallacy AUC on claim-B tokens")
save(fig, "s_context")

# 3. minimal-pair geometry, NLI, all tokens
S = pd.read_csv(R / "geometry_summary.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
for ax, shape in zip(axes, SH):
    x = S[(S.encoder == "nli_pair") & (S["shape"] == shape) & (S.segment == "mean")].sort_values("layer")
    ax.plot(x.layer, x.fallacy_shift_signed_cf, color=NAVY, lw=3)
    ax.plot(x.layer, x.fallacy_shift_abs_ts, color=BRICK, lw=3)
    label_end(ax, 24, x.fallacy_shift_signed_cf.iloc[-1], "relation edit", NAVY)
    label_end(ax, 24, x.fallacy_shift_abs_ts.iloc[-1], "phrase edit", BRICK, dy=-6)
    ax.axhline(0, color="#C9CCD1", lw=1)
    ax.set_title(SH[shape], loc="left", color=NAVY, fontsize=14, fontweight="bold")
    ax.set_xlabel("layer"); ax.set_xlim(0, 31); ax.set_xticks([0, 6, 12, 18, 24])
axes[0].set_ylabel("shift along fallacy direction (sd)")
save(fig, "s_geometry")

# 4. patching heatmap, NLI, counterfactual pairs
pt = pd.read_csv(G / "patching_nli_L24.csv")
cmap = LinearSegmentedColormap.from_list("c", ["#F9F7F1", "#E8D9B5", GOLD, NAVY])
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
for ax, shape in zip(axes, SH):
    segs = ["cls", "A", "C", "B"] if shape == "motte_bailey" else ["cls", "A", "B"]
    x = pt[(pt["shape"] == shape) & (pt.edit_type == "cf")]
    M = x.groupby(["layer", "segment"]).recovery.median().unstack()[segs].clip(0, 1)
    im = ax.imshow(M.values, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(segs)), ["[CLS]", "A", "C", "B"][:len(segs)] if shape == "motte_bailey" else ["[CLS]", "A", "B"])
    ax.set_yticks([0, 6, 12, 18, 24]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(SH[shape], loc="left", color=NAVY, fontsize=14, fontweight="bold")
axes[0].set_ylabel("layer")
cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.03)
cb.set_label("share of the decision restored", color=MUTED); cb.outline.set_visible(False)
save(fig, "s_patching")

# 5. erasure inside the forward pass, NLI
fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=False)
F = pd.read_csv(G / "erasure_forward_nli.csv")
for ax, shape in zip(axes, SH):
    f = F[F["shape"] == shape]
    base = roc_auc_score(f[f.erased == "none"].y, f[f.erased == "none"].logit)
    ax.axhline(base, color=GREY, lw=1.5, ls="--")
    ax.text(0.3, base + 0.024, "dashed line: no erasure", color=MUTED, fontsize=10.5)
    for er, col, name in [("trigger", BRICK, "erase trigger"), ("entail", NAVY, "erase entailment")]:
        r = f[f.erased == er].groupby("layer").apply(lambda g: roc_auc_score(g.y, g.logit))
        ax.plot(r.index, r.values, color=col, lw=3, marker="o", ms=5)
        label_end(ax, 24, r.values[-1], name, col)
    ax.set_title(SH[shape], loc="left", color=NAVY, fontsize=14, fontweight="bold")
    ax.set_xlabel("layer where the direction is erased"); ax.set_xlim(0, 33); ax.set_xticks([0, 6, 12, 18, 24])
    lo = min(base, 0.6); ax.set_ylim(base - 0.07, base + 0.03)
axes[0].set_ylabel("fallacy AUC (held-out)")
save(fig, "s_erasure")

# 6. head ablation, strawman, NLI
B = pd.read_csv(G / "heads_ablation_nli.csv")
b = B[B["shape"] == "strawman"]
vals = {c: roc_auc_score(g.y, g.logit) for c, g in b.groupby("condition")}
fig, ax = plt.subplots(figsize=(6.4, 4.2))
rows = [("intact encoder", vals["none"], NAVY), ("10 random heads removed", vals["random"], GREY), ("top-10 heads removed", vals["top"], BRICK)]
for i, (n, v, c) in enumerate(rows[::-1]):
    ax.barh(i, v - 0.4, left=0.4, color=c, height=0.55)
    ax.text(v + 0.006, i, f"{v:.2f}", va="center", color=NAVY, fontsize=14, fontweight="bold")
ax.set_yticks(range(3), [r[0] for r in rows[::-1]], fontsize=13)
ax.set_xlim(0.4, 0.78); ax.axvline(0.5, color="#C9CCD1", lw=1, ls=":"); ax.grid(axis="y", visible=False)
ax.set_xlabel("strawman AUC on held-out real examples")
save(fig, "s_heads")
print("wrote", sorted(p.name for p in OUT.glob("*.png")), "font:", fam)
