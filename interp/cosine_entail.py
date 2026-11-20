"""RQ4: cosine between the fallacy-label probe direction and the entailment (A => B, frozen NLI model) probe
direction, per layer, on the same standardised features (main rows; all-data fit, mean over 5 bootstrap fits).
Usage: python cosine_entail.py -> results/cosine_fallacy_entail.csv"""
import warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import POSITIVE, RESULTS, SEGMENTS, load_all, load_feats
from probe import C_REG
warnings.filterwarnings("ignore")
df = load_all(); rows = []
for enc in ["nli_pair", "plain_pair", "random_pair"]:
    X, _ = load_feats(enc, df)
    for shape, pos in POSITIVE.items():
        m = np.where((df.type_candidate == shape) & (df.set == "main"))[0]
        y = (df.label.values[m] == pos).astype(int)
        e = (df.entail_AB.values[m] > np.median(df.entail_AB.values[m])).astype(int)
        rng = np.random.default_rng(0)
        for layer in range(X.shape[0]):
            for seg in ["mean", "B"]:
                Z = StandardScaler().fit_transform(X[layer, SEGMENTS.index(seg), m].astype(np.float32))
                cs = []
                for b in range(5):
                    ix = rng.choice(len(m), len(m))
                    wf = LogisticRegression(C=C_REG, max_iter=2000).fit(Z[ix], y[ix]).coef_[0]
                    we = LogisticRegression(C=C_REG, max_iter=2000).fit(Z[ix], e[ix]).coef_[0]
                    cs.append(wf @ we / np.linalg.norm(wf) / np.linalg.norm(we))
                rows.append(dict(encoder=enc, shape=shape, segment=seg, layer=layer, cos=np.mean(cs), cos_sd=np.std(cs),
                                 corr_labels=np.corrcoef(y, e)[0, 1]))
    print(enc, flush=True)
R = pd.DataFrame(rows); R.to_csv(RESULTS / "cosine_fallacy_entail.csv", index=False)
print(R[R.segment == "mean"].pivot_table(index="layer", columns=["encoder", "shape"], values="cos").iloc[::4].round(2))
print("label/entail correlation:", R.groupby("shape").corr_labels.first().round(3).to_dict())
