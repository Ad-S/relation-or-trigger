"""Figures and RESULTS.md from everything in results/ (local) and gpu_results/results/ (GPU session).

Skips any experiment whose result files are missing. Confidence intervals: cluster bootstrap over
conversation_id (1000 resamples) - the unit that grouped cross-validation keeps together.
Usage: python analyze.py      -> figures/*.png|pdf, RESULTS.md
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from common import load_all  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = [HERE / "results", HERE.parent / "gpu_results" / "results"]
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
SHAPES = {"motte_bailey": "Motte-and-bailey", "strawman": "Strawman"}
ENC = {"nli_pair": ("DeBERTa-NLI", "#2a78d6"), "plain_pair": ("DeBERTa plain", "#eb6834"),
       "random_pair": ("random init", "#8a8985"), "nli_b_only": ("DeBERTa-NLI, B only", "#1baf7a"),
       "nli": ("DeBERTa-NLI", "#2a78d6"), "plain": ("DeBERTa plain", "#eb6834")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#a9c9f0", "#2a78d6", "#0d3d7a"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f1f0ec", "#eb6834"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "pdf.fonttype": 42, "savefig.dpi": 200, "savefig.bbox": "tight"})
MD = []


def find(name):
    for d in RES:
        if (d / name).exists():
            return d / name
    return None


def read(name):
    p = find(name)
    return pd.read_csv(p) if p else None


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    MD.append(f"\n![{name}](figures/{name}.png)\n")


def cluster_boot(groups, stat_fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        take = np.concatenate([idx_by[g] for g in rng.choice(uniq, len(uniq))])
        try:
            vals.append(stat_fn(take))
        except ValueError:
            continue
    return np.percentile(vals, [2.5, 97.5])


def table(df, floatfmt=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.{floatfmt}f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------ Exp 1
def exp1():
    P = [read(f"probe_{e}.csv") for e in ENC if e.endswith(("pair", "only"))]
    P = [p for p in P if p is not None]
    if not P:
        return
    P = pd.concat(P)
    g = P.groupby(["encoder", "target", "shape", "segment", "layer"])[["auc", "auc_control", "selectivity"]].mean().reset_index()
    MD.append("\n## Experiment 1 - layer x segment probing\n")
    MD.append("Linear probes (standardised logistic regression, C=0.1), grouped 5-fold CV by conversation, "
              "3 seeds. Control = same probe on randomly permuted labels; selectivity = AUC - control AUC.\n")
    # fig 1: label AUC by layer for the four encoders (segment B and whole sequence)
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.6), sharex=True, sharey=True)
    for i, (shape, sname) in enumerate(SHAPES.items()):
        for j, seg in enumerate(["B", "mean"]):
            ax = axes[j, i]
            ctrl = g[(g.target == "label") & (g["shape"] == shape) & (g.segment == seg)].groupby("layer").auc_control
            ax.fill_between(ctrl.min().index, ctrl.min(), ctrl.max(), color=GRID, lw=0, label="control (permuted labels)")
            for enc in ["nli_pair", "plain_pair", "random_pair", "nli_b_only"]:
                x = g[(g.encoder == enc) & (g.target == "label") & (g["shape"] == shape) & (g.segment == seg)]
                if len(x):
                    ax.plot(x.layer, x.auc, color=ENC[enc][1], lw=2, label=ENC[enc][0])
            ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
            ax.set_title(f"{sname} - {'claim B tokens' if seg == 'B' else 'all tokens'}", color=INK, fontsize=9, loc="left")
            if j == 1:
                ax.set_xlabel("layer")
            if i == 0:
                ax.set_ylabel("fallacy-label AUC")
    axes[0, 0].legend(fontsize=7.5, loc="upper left")
    save(fig, "fig1_label_probe_by_layer")
    # fig 2: selectivity heatmaps for the NLI encoder
    targets = [t for t in ["label", "trigger", "entail", "narrow"] if t in set(g.target)]
    segs = ["cls", "A", "C", "B", "trig", "brest", "mean"]
    fig, axes = plt.subplots(2, len(targets), figsize=(2.6 * len(targets), 5.2), sharey=True)
    for i, (shape, sname) in enumerate(SHAPES.items()):
        for j, t in enumerate(targets):
            ax = axes[i, j]
            x = g[(g.encoder == "nli_pair") & (g.target == t) & (g["shape"] == shape)]
            M = x.pivot(index="layer", columns="segment", values="selectivity").reindex(columns=segs)
            im = ax.imshow(M.values, aspect="auto", origin="lower", cmap=SEQ, vmin=0, vmax=0.5)
            ax.set_xticks(range(len(segs)), segs, rotation=45, fontsize=7)
            ax.grid(False)
            ax.set_title(f"{sname}\n{t}", fontsize=8, loc="left", color=INK)
            if j == 0:
                ax.set_ylabel("layer")
    fig.colorbar(im, ax=axes, shrink=0.6, label="selectivity (AUC - control AUC)")
    save(fig, "fig2_selectivity_heatmaps_nli")
    # table: best cell per encoder/target/shape (segment B and mean) + layer 0
    rows = []
    for (enc, t, shape), x in g.groupby(["encoder", "target", "shape"]):
        for seg in ["B", "mean", "A", "cls"]:
            y = x[x.segment == seg]
            if y.empty:
                continue
            b = y.loc[y.auc.idxmax()]
            rows.append(dict(encoder=ENC[enc][0], target=t, shape=SHAPES[shape], segment=seg, best_layer=int(b.layer),
                             best_auc=b.auc, control_auc=b.auc_control, selectivity=b.selectivity,
                             layer0_auc=float(y[y.layer == 0].auc.iloc[0]), last_auc=float(y[y.layer == y.layer.max()].auc.iloc[0])))
    T = pd.DataFrame(rows)
    MD.append("\n### Fallacy label (best layer per segment)\n")
    MD.append(table(T[T.target == "label"].drop(columns="target")))
    MD.append("\n\n### Trigger presence, entailment (A => B), narrowing proxy\n")
    MD.append(table(T[(T.target != "label") & T.encoder.isin(["DeBERTa-NLI", "DeBERTa plain", "random init"])]))
    T.to_csv(FIG / "table_probe_best.csv", index=False)


# ------------------------------------------------------------------------------------------ Exp 2
def exp2():
    S, P = read("geometry_summary.csv"), read("geometry_pairs.csv")
    if S is None:
        return
    MD.append("\n## Experiment 2 - minimal pairs along the fallacy direction\n")
    MD.append("Probes fit on training conversations only; each held-out pair's shift along the fallacy-probe logit, "
              "in held-out sd units. cf = counterfactual (label flips, trigger kept), signed toward the new label; "
              "ts = trigger swap (label kept), absolute. Segment A has identical tokens in both members, so a shift "
              "at A is information moved from B by attention.\n")
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True, gridspec_kw=dict(wspace=0.28))
    for i, (shape, sname) in enumerate(SHAPES.items()):
        for j, seg in enumerate(["mean", "A"]):
            ax = axes[i, j]
            for enc in ["nli_pair", "plain_pair", "random_pair"]:
                x = S[(S.encoder == enc) & (S["shape"] == shape) & (S.segment == seg)].sort_values("layer")
                ax.plot(x.layer, x.fallacy_shift_signed_cf, color=ENC[enc][1], lw=2, label=f"{ENC[enc][0]}: cf (signed)")
                ax.plot(x.layer, x.fallacy_shift_abs_ts, color=ENC[enc][1], lw=1.4, ls="--", label=f"{ENC[enc][0]}: ts (|shift|)")
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_title(f"{sname}: {'all tokens' if seg == 'mean' else 'A tokens (identical)'}", fontsize=8.5, loc="left", color=INK)
            if j == 0:
                ax.set_ylabel("shift along fallacy direction (sd)")
        ax = axes[i, 2]
        for enc in ["nli_pair", "plain_pair", "random_pair"]:
            x = S[(S.encoder == enc) & (S["shape"] == shape) & (S.segment == "mean")].sort_values("layer")
            ax.plot(x.layer, x.cos_fallacy_trigger, color=ENC[enc][1], lw=2, label=ENC[enc][0])
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"{sname}: cos(fallacy, trigger)", fontsize=8.5, loc="left", color=INK)
    for ax in axes[1]:
        ax.set_xlabel("layer")
    axes[0, 0].legend(fontsize=6.5, loc="upper left")
    save(fig, "fig3_minimal_pair_geometry")
    if P is None:
        return
    df = load_all().set_index("row_id")
    P["conv"] = df.loc[P.mp_id, "conversation_id"].values
    rows = []
    for enc in ["nli_pair", "plain_pair", "random_pair"]:
        for shape in SHAPES:
            for layer in [0, 12, 18, 24]:
                x = P[(P.encoder == enc) & (P["shape"] == shape) & (P.segment == "mean") & (P.layer == layer)]
                cf = x[x.edit_type == "cf"].groupby(["mp_id", "conv"]).fallacy_shift_signed.mean().reset_index()
                ts = x[x.edit_type == "ts"].groupby(["mp_id", "conv"]).fallacy_shift_abs.mean().reset_index()
                both = pd.concat([cf.assign(v=cf.fallacy_shift_signed, k="cf"), ts.assign(v=ts.fallacy_shift_abs, k="ts")])
                g = both.conv.values

                def diff(ix, b=both):
                    s = b.iloc[ix]
                    return s[s.k == "cf"].v.mean() - s[s.k == "ts"].v.mean()
                lo, hi = cluster_boot(g, diff)
                beh = x.assign(ok=np.where(x.edit_type == "cf", (x.pred_edit == x.y_edit) & (x.pred_orig == x.y_orig),
                                           x.pred_edit == x.pred_orig)).groupby("edit_type").ok.mean()
                rows.append(dict(encoder=ENC[enc][0], shape=SHAPES[shape], layer=layer,
                                 cf_signed_shift=cf.fallacy_shift_signed.mean(), ts_abs_shift=ts.fallacy_shift_abs.mean(),
                                 diff=diff(np.arange(len(both))), ci_lo=lo, ci_hi=hi,
                                 cf_both_correct=beh.get("cf", np.nan), ts_pred_kept=beh.get("ts", np.nan)))
    MD.append("\n### cf signed shift vs ts absolute shift (all tokens), cluster-bootstrap 95% CI of the difference\n")
    MD.append(table(pd.DataFrame(rows)))


# ------------------------------------------------------------------------------------------ Exp 3
def exp3():
    found = [(e, read(f"patching_{e}_L24.csv")) for e in ["nli", "plain"]]
    found = [(e, r) for e, r in found if r is not None]
    if not found:
        return
    MD.append("\n## Experiment 3 - residual-stream activation patching\n")
    MD.append("Frozen encoder + layer-24 mean-pooled linear readout (fold-wise, conversation held out). "
              "clean = original, corrupt = edit. Recovery = fraction of the clean-corrupt readout gap restored by "
              "patching a segment's states at one layer (B: clean run with corrupt context, i.e. only clean B left). "
              "Pairs with |gap| < 0.5 readout-sd are excluded. Mean recovery is shown; 1 = segment carries the whole effect.\n")
    segs = ["cls", "A", "C", "ctx", "B"]
    fig, axes = plt.subplots(len(found), 4, figsize=(11, 3.3 * len(found)), squeeze=False, gridspec_kw=dict(hspace=0.45, wspace=0.3))
    rows = []
    for i, (enc, R) in enumerate(found):
        for j, (shape, et) in enumerate([(s, e) for s in SHAPES for e in ["cf", "ts"]]):
            ax = axes[i, j]
            x = R[(R["shape"] == shape) & (R.edit_type == et)]
            x = x.assign(recovery=x.recovery.clip(-2, 3))
            M = x.groupby(["layer", "segment"]).recovery.median().unstack().reindex(columns=[s for s in segs if s in set(x.segment)])
            im = ax.imshow(M.values, aspect="auto", origin="lower", cmap=DIV, norm=TwoSlopeNorm(0, -0.5, 1.2))
            ax.set_xticks(range(M.shape[1]), M.columns)
            ax.grid(False)
            ax.set_title(f"{ENC[enc][0]}\n{SHAPES[shape]}, {et} (n={x.mp_id.nunique()})", fontsize=8, loc="left", color=INK)
            if j == 0:
                ax.set_ylabel("layer")
            for layer in [6, 12, 18, 23]:
                for s in M.columns:
                    v = x[(x.layer == layer) & (x.segment == s)]
                    rows.append(dict(encoder=ENC[enc][0], shape=SHAPES[shape], edit=et, layer=layer, segment=s,
                                     median_recovery=v.recovery.median(), n_pairs=v.mp_id.nunique()))
    fig.colorbar(im, ax=axes, shrink=0.7, label="median recovery")
    save(fig, "fig4_patching_heatmaps")
    pr = [read(f"patching_pairs_{e}_L24.csv") for e, _ in found]
    pr = pd.concat([p for p in pr if p is not None])
    pr["kept"] = pr.gap_sd.abs() >= 0.5
    s = pr.groupby(["encoder", "shape", "edit_type"]).agg(pairs=("mp_id", "size"), mean_abs_gap_sd=("gap_sd", lambda v: v.abs().mean()),
                                                         kept=("kept", "mean")).reset_index()
    MD.append("\n### Size of the readout change per pair (|LD_clean - LD_corrupt| in readout sd)\n")
    MD.append(table(s))
    T = pd.DataFrame(rows).pivot_table(index=["encoder", "shape", "edit", "layer"], columns="segment", values="median_recovery").reset_index()
    MD.append("\n\n### Median recovery at selected layers\n")
    MD.append(table(T))


# ------------------------------------------------------------------------------------------ Exp 4
def exp4():
    E = read("erasure_probe.csv")
    F = [(e, read(f"erasure_forward_{e}.csv")) for e in ["nli", "plain"]]
    F = [(e, f) for e, f in F if f is not None]
    if E is None and not F:
        return
    MD.append("\n## Experiment 4 - concept erasure (LEACE)\n")
    col = {"none": INK2, "trigger": "#eb6834", "entail": "#2a78d6", "label": "#1baf7a", "random": "#8a8985"}
    if E is not None:
        g = E.groupby(["encoder", "shape", "segment", "layer", "erased"]).auc.mean().reset_index()
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.4), sharex=True, sharey=True)
        for i, enc in enumerate(["nli_pair", "plain_pair"]):
            for j, shape in enumerate(SHAPES):
                ax = axes[i, j]
                for er in ["none", "random", "trigger", "entail", "label"]:
                    x = g[(g.encoder == enc) & (g["shape"] == shape) & (g.segment == "mean") & (g.erased == er)]
                    if len(x):
                        ax.plot(x.layer, x.auc, color=col[er], lw=2 if er != "none" else 1.4,
                                ls="-" if er != "none" else "--", label=f"erase {er}" if er != "none" else "no erasure")
                ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
                ax.set_title(f"{ENC[enc][0]}: {SHAPES[shape]}", fontsize=8.5, loc="left", color=INK)
                if i == 1:
                    ax.set_xlabel("layer")
                if j == 0:
                    ax.set_ylabel("fallacy-label probe AUC")
        axes[0, 0].legend(fontsize=7)
        save(fig, "fig5_amnesic_probing")
        t = g[g.segment == "mean"].pivot_table(index=["encoder", "shape", "layer"], columns="erased", values="auc").reset_index()
        t = t[t.layer.isin([0, 6, 12, 18, 24])]
        MD.append("\n### (a) Amnesic probing on cached features (label-probe AUC after erasing a concept; all tokens)\n")
        MD.append(table(t))
    for enc, f in F:
        rows = []
        for (shape, er, layer), x in f.groupby(["shape", "erased", "layer"]):
            rows.append(dict(shape=SHAPES[shape], erased=er, layer=layer, auc=roc_auc_score(x.y, x.logit)))
        t = pd.DataFrame(rows).pivot_table(index=["shape", "layer"], columns="erased", values="auc").reset_index()
        MD.append(f"\n### (b) {ENC[enc][0]}: erase at layer l inside the forward pass, layer-24 readout AUC on held-out rows "
                  "(layer -1 = no erasure)\n")
        MD.append(table(t))


# ------------------------------------------------------------------------------------------ heads
def heads():
    for enc in ["nli", "plain"]:
        A, V, B = read(f"heads_atp_{enc}.csv"), read(f"heads_verify_{enc}.csv"), read(f"heads_ablation_{enc}.csv")
        if A is None:
            continue
        MD.append(f"\n## Heads and MLPs ({ENC[enc][0]})\n")
        MD.append("Attribution patching on counterfactual pairs at the context positions (B-dependent information "
                  "written into [CLS]/A/C), verified by real patching; mean-ablation of the top-10 heads on held-out real rows.\n")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        for ax, shape in zip(axes, SHAPES):
            x = A[(A["shape"] == shape) & (A.kind == "head")].groupby(["layer", "head"]).atp.mean().unstack()
            im = ax.imshow(x.T.values, aspect="auto", origin="lower", cmap=DIV, norm=TwoSlopeNorm(0, -x.abs().max().max(), x.abs().max().max()))
            ax.set_xlabel("layer"); ax.set_ylabel("head"); ax.grid(False)
            ax.set_title(f"{SHAPES[shape]}: mean AtP recovery per head", fontsize=8.5, loc="left", color=INK)
            fig.colorbar(im, ax=ax, shrink=0.8)
        save(fig, f"fig6_heads_atp_{enc}")
        if V is not None:
            v = V.groupby(["shape", "comp"]).agg(atp=("atp_mean", "first"), patched_recovery=("recovery", "median"),
                                                 pairs=("mp_id", "size")).reset_index().sort_values(["shape", "atp"], ascending=[True, False])
            MD.append("\n### Top components: AtP estimate vs real patching (median recovery)\n")
            MD.append(table(v))
        if B is not None:
            rows = []
            for (shape, cond), x in B.groupby(["shape", "condition"]):
                rows.append(dict(shape=SHAPES[shape], condition=cond, auc=roc_auc_score(x.y, x.logit),
                                 mean_logit_pos_minus_neg=x[x.y == 1].logit.mean() - x[x.y == 0].logit.mean()))
            MD.append("\n\n### Mean-ablation on held-out real rows\n")
            MD.append(table(pd.DataFrame(rows)))


if __name__ == "__main__":
    MD.append("# Interpretability results (generated by analyze.py)\n")
    for f in (exp1, exp2, exp3, exp4, heads):
        f()
    (HERE / "RESULTS.md").write_text("\n".join(MD) + "\n")
    print(f"wrote RESULTS.md and {len(list(FIG.glob('*.png')))} figures")
