"""Randomisation inference for the stacked difference test and the share model.

Each replication relabels every deal as pseudo-technology and refits. Replications
use independent streams derived from a fixed seed, so results do not depend on the
number of worker processes.

designs
  iid       each deal is pseudo-technology with the observed 29.55 per cent probability
  dyadshare dyads swap their observed technology shares, then deals are drawn at that
            share; this keeps the persistence of the technology share within a dyad

usage: python 02_ri_placebo.py --design iid --reps 10000 --workers 10
"""
import argparse, multiprocessing as mp, time, warnings
import numpy as np, pandas as pd
from thesis_lib import *

SEED = 20261105
G = {}


def init():
    warnings.filterwarnings("ignore")
    deals = load_deals()
    cov = add_scaled_patents(load_covariates())
    key, cov2 = prepare_keys(deals, cov)
    G.update(key=key, cov=cov2, val=deals.deal_value.to_numpy(),
             lab=deals.is_tech_target.to_numpy().astype(bool))
    dy = pd.factorize(cov2["dyad"])[0]
    G["dy"] = dy[key]
    G["ndy"] = dy.max() + 1
    n = np.bincount(G["dy"], minlength=G["ndy"])
    G["sh_dy"] = np.bincount(G["dy"], weights=G["lab"], minlength=G["ndy"]) / np.maximum(n, 1)
    G["p"] = G["lab"].mean()
    base = cov2.dropna(subset=REGS).index.to_numpy()
    G["share_idx"] = base


def draw(design, rng):
    n = len(G["lab"])
    if design == "iid":
        return rng.random(n) < G["p"]
    s = rng.permutation(G["sh_dy"])
    return rng.random(n) < s[G["dy"]]


def one(args):
    i, design = args
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(i,)))
    lab = draw(design, rng)
    row = {"rep": i}
    try:
        d = stacked_sector_lag(G["key"], G["val"], lab, G["cov"])
        for name, st in ladder_stats(d).items():
            row[f"{name}_N"], row[f"{name}_wald"], row[f"{name}_wald_p"] = st["N"], st["wald"], st["wald_p"]
            for v, b in st["coef"].items():
                row[f"{name}_{v}"] = b
                row[f"{name}_{v}_p"] = st["p"][v]
        regs = [v for v in REGS if not v.startswith("patents")] + ["pat_gdp_o_l1", "pat_gdp_d_l1"]
        m = fit_stacked(d, regs, ["year^tech"])
        row["year_pat_gdp_d"] = float(m.coef()["tech:pat_gdp_d_l1"])
        c = G["cov"].loc[G["share_idx"]].copy()
        n = np.bincount(G["key"], minlength=len(G["cov"]))
        t = np.bincount(G["key"], weights=lab, minlength=len(G["cov"]))
        c["n"] = n[G["share_idx"]]
        c["sh"] = t[G["share_idx"]] / c["n"]
        for name, (fes, regs) in SHARE_LADDER.items():
            m = fit_share(c, regs, fes)
            for v, b in m.coef().items():
                row[f"share_{name}_{v}"] = float(b)
    except Exception as e:
        row["error"] = str(e)[:200]
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="iid", choices=["iid", "dyadshare"])
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    out = RESULTS / "ri"
    out.mkdir(exist_ok=True)
    path = out / f"ri_{a.design}.csv"
    done = set(pd.read_csv(path, usecols=["rep"])["rep"]) if path.exists() else set()
    todo = [(i, a.design) for i in range(a.reps) if i not in done]
    print(f"{a.design}: {len(done)} done, {len(todo)} to run, seed {SEED}", flush=True)
    t0 = time.time()
    buf, hdr, count = [], not path.exists(), 0
    cols = None
    if not hdr:
        cols = list(pd.read_csv(path, nrows=0).columns)
    with mp.Pool(a.workers, initializer=init) as pool:
        for r in pool.imap_unordered(one, todo, chunksize=5):
            buf.append(r); count += 1
            if len(buf) >= 200 or count == len(todo):
                df = pd.DataFrame(buf)
                if cols is None:
                    cols = list(df.columns)
                    if "error" not in cols: cols.append("error")
                df = df.reindex(columns=cols)
                df.to_csv(path, mode="a", header=hdr, index=False)
                hdr, buf = False, []
                print(f"{count}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
