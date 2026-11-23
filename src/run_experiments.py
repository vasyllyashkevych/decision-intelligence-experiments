"""Experiments E1-E3 (see PREREGISTRATION.md). Run: python3 -I src/get_data.py && python3 -I src/run_experiments.py"""
import json, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "results"
SEED = 20261123
B_VALUES = [2, 3, 5, 10, 20]
FEATS = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", "lcom3", "loc", "dam", "moa", "mfa", "cam", "ic", "cbm", "amc", "max_cc", "avg_cc"]


def load(name):
    d = pd.read_csv(RAW / f"{name}.csv")
    d = d.loc[:, ~d.columns.duplicated(keep="last")] if d.columns.duplicated().any() else d
    cols = list(d.columns)
    cols[2] = "cls"  # third column is the class name (the first one is the project name)
    d.columns = cols
    d["y"] = (d["bug"] > 0).astype(int)
    return d


def releases():
    items = {}
    for p in sorted(RAW.glob("*.csv")):
        proj, ver = p.stem.rsplit("-", 1)
        items.setdefault(proj, []).append((tuple(int(x) for x in ver.split(".")), p.stem))
    pairs = []
    for proj, lst in items.items():
        lst.sort()
        for (_, a), (_, b) in zip(lst, lst[1:]):
            pairs.append((proj, a, b))
    return pairs


def X_of(d):
    return np.log1p(d[FEATS].clip(lower=0).to_numpy(float))


def fit_predict(kind, Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if kind == "LR":
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
    else:
        m = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=SEED, n_jobs=1).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for k in range(bins):
        m = idx == k
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def net_benefit(inspect, y, b):
    return (b * (inspect & (y == 1)).sum() - inspect.sum()) / len(y)


def oracle_nb(score, y, b):
    order = np.argsort(-score)
    tp = np.cumsum(y[order] == 1)
    k = np.arange(1, len(y) + 1)
    nb = (b * tp - k) / len(y)
    return max(0.0, nb.max())  # 0 = inspect none


# ---------------------------------------------------------------- E1
def e1():
    rows = []
    for proj, a, b_ in releases():
        tr, te = load(a), load(b_)
        Xtr, ytr, Xte, yte = X_of(tr), tr.y.to_numpy(), X_of(te), te.y.to_numpy()
        if len(set(ytr)) < 2 or len(set(yte)) < 2:
            continue
        scores = {"LR": fit_predict("LR", Xtr, ytr, Xte), "RF": fit_predict("RF", Xtr, ytr, Xte), "LOC": te["loc"].to_numpy(float)}
        shift = yte.mean() - ytr.mean()
        for m, s in scores.items():
            auc = roc_auc_score(yte, s)
            rec = dict(project=proj, train=a, test=b_, model=m, n_test=len(yte), prev_train=ytr.mean(), prev_test=yte.mean(), shift=shift, auc=auc)
            if m != "LOC":
                rec.update(brier=brier_score_loss(yte, s), ece=ece(s, yte), mean_p=s.mean())
            for b in B_VALUES:
                if m == "LOC":
                    # LOC is a ranking, not a probability: inspect the same share of classes as LR would
                    k = int((scores["LR"] >= 1 / (b + 1)).sum())
                    ins = np.zeros(len(yte), bool)
                    ins[np.argsort(-s)[:k]] = True
                else:
                    ins = s >= 1 / (b + 1)
                rec[f"nb_b{b}"] = net_benefit(ins, yte, b)
                rec[f"orc_b{b}"] = oracle_nb(s, yte, b)
                rec[f"all_b{b}"] = net_benefit(np.ones(len(yte), bool), yte, b)
            rows.append(rec)
    return pd.DataFrame(rows)


def e1_summary(df):
    s = {}
    lr = df[df.model == "LR"].reset_index(drop=True)
    rf = df[df.model == "RF"].reset_index(drop=True)
    loc = df[df.model == "LOC"].reset_index(drop=True)
    s["n_pairs"] = int(len(lr))
    s["auc_median"] = {m: float(df[df.model == m].auc.median()) for m in ["LR", "RF", "LOC"]}
    s["auc_iqr"] = {m: [float(df[df.model == m].auc.quantile(q)) for q in (.25, .75)] for m in ["LR", "RF", "LOC"]}
    for m, d in (("LR", lr), ("RF", rf)):
        r1 = stats.spearmanr(d["shift"].abs(), d["ece"])
        r2 = stats.spearmanr(d["shift"].abs(), d["brier"])
        s[f"H1a_{m}_spearman_absShift_vs_ECE"] = [float(r1.statistic), float(r1.pvalue)]
        s[f"H1a_{m}_spearman_absShift_vs_Brier"] = [float(r2.statistic), float(r2.pvalue)]
        s[f"{m}_ece_median"] = float(d.ece.median())
    # exploratory (NOT pre-registered): can the shift be seen before outcomes arrive?
    proxy = (lr.mean_p - lr.prev_train).abs()
    r3 = stats.spearmanr(proxy, lr["shift"].abs()); r4 = stats.spearmanr(proxy, lr.ece)
    s["EXPLORATORY_observable_proxy"] = dict(
        definition="|mean predicted p on r+1 - observed prevalence on r|",
        spearman_vs_abs_true_shift=[float(r3.statistic), float(r3.pvalue)],
        spearman_vs_ece=[float(r4.statistic), float(r4.pvalue)])
    # H1b: thresholded model vs best trivial policy
    h1b = {}
    for m, d in (("LR", lr), ("RF", rf)):
        for b in B_VALUES:
            triv = np.maximum(d[f"all_b{b}"], 0.0)
            gap = d[f"nb_b{b}"] - triv
            h1b[f"{m}_b{b}"] = dict(
                share_model_strictly_better=float((gap > 1e-12).mean()),
                share_model_tied_with_trivial=float((gap.abs() <= 1e-12).mean()),
                share_model_strictly_worse_than_trivial=float((gap < -1e-12).mean()),
                share_model_not_better_than_trivial=float((d[f"nb_b{b}"] <= triv + 1e-12).mean()),
                share_model_negative_nb=float((d[f"nb_b{b}"] < 0).mean()),
                median_regret_vs_oracle=float((d[f"orc_b{b}"] - d[f"nb_b{b}"]).median()))
    s["H1b"] = h1b
    # H1c: AUC order vs NB order, LR vs RF
    h1c = {}
    for b in B_VALUES:
        auc_sign = np.sign(lr.auc - rf.auc)
        nb_sign = np.sign(lr[f"nb_b{b}"] - rf[f"nb_b{b}"])
        valid = (auc_sign != 0) & (nb_sign != 0)
        h1c[f"b{b}"] = dict(discordant=float((auc_sign[valid] != nb_sign[valid]).mean()), n=int(valid.sum()))
    s["H1c_LR_vs_RF"] = h1c
    # also: model vs LOC baseline
    base = {}
    for b in B_VALUES:
        base[f"b{b}"] = dict(
            LR_beats_LOC_share=float((lr[f"nb_b{b}"] > loc[f"nb_b{b}"]).mean()),
            RF_beats_LOC_share=float((rf[f"nb_b{b}"] > loc[f"nb_b{b}"]).mean()),
            LR_AUC_beats_LOC_share=float((lr.auc > loc.auc).mean()))
    s["vs_LOC_baseline"] = base
    return s


# ---------------------------------------------------------------- E2
def pkg(n):
    return n.rsplit(".", 1)[0] if "." in n else n


def oof_lr(X, y):
    p = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for a, b in skf.split(X, y):
        p[b] = fit_predict("LR", X[a], y[a], X[b])
    return p


def e2():
    rows = []
    for proj, a, b_ in releases():
        tr, te = load(a), load(b_)
        if tr.y.sum() < 5 or (tr.y == 0).sum() < 5:
            continue
        tr = tr.assign(p=oof_lr(X_of(tr), tr.y.to_numpy()), pkg=tr.cls.map(pkg))
        te = te.assign(pkg=te.cls.map(pkg))
        g_tr = tr.groupby("pkg")
        g_te = te.groupby("pkg").agg(n_te=("y", "size"), out=("y", "mean"))
        agg = pd.DataFrame({
            "mean": g_tr.p.mean(), "max": g_tr.p.max(),
            "top3": g_tr.p.apply(lambda s: s.nlargest(3).mean()),
            "share_ge_.5": g_tr.p.apply(lambda s: (s >= .5).mean()), "n_tr": g_tr.size()})
        d = agg.join(g_te, how="inner")
        d = d[(d.n_tr >= 3) & (d.n_te >= 3)]
        if len(d) < 8 or d.out.nunique() < 2:
            continue
        rec = dict(project=proj, train=a, test=b_, n_pkg=len(d))
        for k in ["mean", "max", "top3", "share_ge_.5"]:
            rec[f"rho_{k}"] = float(stats.spearmanr(d[k], d.out).statistic)
            q = d[k].rank(ascending=False, method="first") <= max(1, int(np.ceil(len(d) / 4)))
            rec[f"topq_out_{k}"] = float(d.out[q].mean())
        rec["base_out"] = float(d.out.mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def e2_summary(df):
    rng = np.random.default_rng(SEED)
    s = {"n_pairs": int(len(df)), "n_packages_total": int(df.n_pkg.sum())}
    for k in ["mean", "max", "top3", "share_ge_.5"]:
        s[f"rho_{k}_mean"] = float(df[f"rho_{k}"].mean())
        s[f"rho_{k}_median"] = float(df[f"rho_{k}"].median())
    for k in ["max", "top3", "share_ge_.5"]:
        diff = (df[f"rho_{k}"] - df["rho_mean"]).to_numpy()
        boots = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
        try:
            w = stats.wilcoxon(diff).pvalue
        except ValueError:
            w = None
        s[f"{k}_minus_mean"] = dict(mean=float(diff.mean()), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                    wilcoxon_p=None if w is None else float(w), share_pairs_better=float((diff > 0).mean()))
    s["topq_outcome_mean"] = {k: float(df[f"topq_out_{k}"].mean()) for k in ["mean", "max", "top3", "share_ge_.5"]}
    s["topq_outcome_mean"]["all_packages"] = float(df.base_out.mean())
    return s


# ---------------------------------------------------------------- E3
QS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75]


def kendall(a, b):
    return stats.kendalltau(a, b).statistic


def e3():
    rng = np.random.default_rng(SEED)
    rows = []
    for proj, a, b_ in releases():
        tr, te = load(a), load(b_)
        if len(set(tr.y)) < 2 or len(set(te.y)) < 2:
            continue
        s = fit_predict("LR", X_of(tr), tr.y.to_numpy(), X_of(te))
        y, loc = te.y.to_numpy(), te["loc"].to_numpy(float)
        order = np.argsort(-s)
        V = []
        for q in QS:
            k = max(1, int(round(q * len(y))))
            sel = order[:k]
            rec = y[sel].sum() / max(1, y.sum())
            eff = 1 - loc[sel].sum() / loc.sum()
            prec = y[sel].mean()
            V.append([rec, eff, prec])
        V = np.array(V)
        W = rng.dirichlet([1, 1, 1], 2000)
        U = W @ V.T
        base = V.mean(1) * 0 + V @ np.array([1 / 3] * 3)
        win_eq = int(np.argmax(base))
        wins = U.argmax(1)
        # min-max normalised variant (the rubric depends on normalisation choice)
        Vn = (V - V.min(0)) / (V.max(0) - V.min(0) + 1e-12)
        wins_n = (W @ Vn.T).argmax(1)
        win_eq_n = int(np.argmax(Vn @ np.array([1 / 3] * 3)))
        # hard constraint: recall >= 0.5
        adm = V[:, 0] >= 0.5
        if adm.any():
            Uc = np.where(adm[None, :], U, -np.inf)
            wins_c = Uc.argmax(1)
            win_eq_c = int(np.argmax(np.where(adm, base, -np.inf)))
        else:
            wins_c, win_eq_c = wins, win_eq
        taus = [kendall(U[i], base) for i in range(0, 2000, 10)]
        rows.append(dict(project=proj, train=a, test=b_, n_admissible=int(adm.sum()),
                         p_winner_differs=float((wins != win_eq).mean()),
                         p_winner_differs_minmax=float((wins_n != win_eq_n).mean()),
                         p_winner_differs_constrained=float((wins_c != win_eq_c).mean()),
                         n_distinct_winners=int(len(set(wins))), n_distinct_winners_constrained=int(len(set(wins_c))),
                         mean_tau=float(np.nanmean(taus)), winner_eq=QS[win_eq]))
    return pd.DataFrame(rows)


def e3_summary(df):
    return dict(n_pairs=int(len(df)),
                median_p_winner_differs=float(df.p_winner_differs.median()),
                share_pairs_winner_differs_gt_25pct=float((df.p_winner_differs > .25).mean()),
                median_p_winner_differs_minmax=float(df.p_winner_differs_minmax.median()),
                median_p_winner_differs_constrained=float(df.p_winner_differs_constrained.median()),
                median_distinct_winners=float(df.n_distinct_winners.median()),
                median_distinct_winners_constrained=float(df.n_distinct_winners_constrained.median()),
                median_mean_tau=float(df.mean_tau.median()),
                equal_weight_winner_counts={str(k): int(v) for k, v in df.winner_eq.value_counts().items()})


if __name__ == "__main__":
    d1 = e1(); d1.to_csv(OUT / "e1_pairs.csv", index=False)
    d2 = e2(); d2.to_csv(OUT / "e2_pairs.csv", index=False)
    d3 = e3(); d3.to_csv(OUT / "e3_pairs.csv", index=False)
    summary = dict(E1=e1_summary(d1), E2=e2_summary(d2), E3=e3_summary(d3))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
