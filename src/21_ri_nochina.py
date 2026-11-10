"""Randomisation inference for the stacked difference test without China.

Same designs, seed and code path as 02_ri_placebo.py; deals with China as acquirer or
target are dropped before relabelling. Share model and scaled patents are not rerun.

usage: python 21_ri_nochina.py --check
       python 21_ri_nochina.py --design iid --reps 10000 --workers 10
       python 21_ri_nochina.py --design dyadshare --reps 2000 --workers 10
       python 21_ri_nochina.py --summary
Output: results/ri/ri_nochn_{design}.csv, results/ri_nochina.csv
"""
import argparse, multiprocessing as mp, time, warnings
import numpy as np, pandas as pd
from scipy import stats as st
from thesis_lib import *

SEED = 20261105
G = {}
STATS = ["joint_wald"] + list(REGS)


def init():
    warnings.filterwarnings("ignore")
    deals = load_deals()
    deals = deals[(deals.iso_o != "CHN") & (deals.iso_d != "CHN")].reset_index(drop=True)
    cov = add_scaled_patents(load_covariates())
    cov = cov[(cov.iso_o != "CHN") & (cov.iso_d != "CHN")]
    key, cov2 = prepare_keys(deals, cov)
    G.update(key=key, cov=cov2, val=deals.deal_value.to_numpy(), lab=deals.is_tech_target.to_numpy().astype(bool))
    dy = pd.factorize(cov2["dyad"])[0]
    G["dy"] = dy[key]; G["ndy"] = dy.max() + 1
    n = np.bincount(G["dy"], minlength=G["ndy"])
    G["sh_dy"] = np.bincount(G["dy"], weights=G["lab"], minlength=G["ndy"]) / np.maximum(n, 1)
    G["p"] = G["lab"].mean()


def draw(design, rng):
    n = len(G["lab"])
    if design == "iid":
        return rng.random(n) < G["p"]
    s = rng.permutation(G["sh_dy"])
    return rng.random(n) < s[G["dy"]]


def stats_row(lab):
    row = {}
    d = stacked_sector_lag(G["key"], G["val"], lab, G["cov"])
    for name, s in ladder_stats(d).items():
        row[f"{name}_N"], row[f"{name}_joint_wald"], row[f"{name}_joint_wald_p"] = s["N"], s["wald"], s["wald_p"]
        for v, b in s["coef"].items():
            row[f"{name}_{v}"], row[f"{name}_{v}_p"] = b, s["p"][v]
    return row


def one(args):
    i, design = args
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(i,)))
    row = {"rep": i}
    try:
        row.update(stats_row(draw(design, rng)))
    except Exception as e:
        row["error"] = str(e)[:200]
    return row


def observed():
    init()
    return stats_row(G["lab"])


def check():
    o = observed()
    for fe in ("year", "country", "pair"):
        print(fe, "N", o[f"{fe}_N"], "joint Wald", round(o[f"{fe}_joint_wald"], 2), "nominal p", round(o[f"{fe}_joint_wald_p"], 5))
    pd.Series(o).to_csv(RESULTS / "ri" / "ri_nochn_observed.csv", header=["value"])


def summary():
    o = pd.read_csv(RESULTS / "ri" / "ri_nochn_observed.csv", index_col=0)["value"]
    rows = []
    for design in ("iid", "dyadshare"):
        f = RESULTS / "ri" / f"ri_nochn_{design}.csv"
        if not f.exists():
            continue
        B = pd.read_csv(f).drop_duplicates("rep")
        print(design, len(B), "errors", int(B["error"].notna().sum()) if "error" in B else 0)
        for fe in ("year", "country", "pair"):
            for s in STATS:
                col = f"{fe}_{s}"
                if col not in B or col not in o.index:
                    continue
                x = pd.to_numeric(B[col], errors="coerce").dropna().to_numpy(); N = len(x)
                ob = float(o[col]); r = int((np.abs(x) >= abs(ob)).sum()); p = (r + 1) / (N + 1)
                rows.append(dict(design=design, fe=fe, stat=s, estimate=ob, nominal_p=float(o[col + "_p"]), N=N, r=r, ri_p=p,
                                 mc_se=np.sqrt(p * (1 - p) / N), ci_lo=st.beta.ppf(.025, r, N - r + 1) if r else 0.0,
                                 ci_hi=st.beta.ppf(.975, r + 1, N - r), placebo_mean=float(x.mean()) if s != "joint_wald" else np.nan))
    T = pd.DataFrame(rows)
    T.to_csv(RESULTS / "ri_nochina.csv", index=False)
    pd.set_option("display.width", 200)
    print(T[["design", "fe", "stat", "estimate", "nominal_p", "ri_p", "mc_se"]].to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="iid", choices=["iid", "dyadshare"])
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    if a.check:
        check()
    elif a.summary:
        summary()
    else:
        path = RESULTS / "ri" / f"ri_nochn_{a.design}.csv"
        done = set(pd.read_csv(path, usecols=["rep"])["rep"]) if path.exists() else set()
        todo = [(i, a.design) for i in range(a.reps) if i not in done]
        print(f"{a.design}: {len(done)} done, {len(todo)} to run, seed {SEED}", flush=True)
        t0 = time.time(); buf, hdr, count = [], not path.exists(), 0
        cols = list(pd.read_csv(path, nrows=0).columns) if not hdr else None
        with mp.Pool(a.workers, initializer=init) as pool:
            for r in pool.imap_unordered(one, todo, chunksize=5):
                buf.append(r); count += 1
                if len(buf) >= 200 or count == len(todo):
                    df = pd.DataFrame(buf)
                    if cols is None:
                        cols = list(df.columns)
                        if "error" not in cols:
                            cols.append("error")
                    df.reindex(columns=cols).to_csv(path, mode="a", header=hdr, index=False)
                    hdr, buf = False, []
                    print(f"{count}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
