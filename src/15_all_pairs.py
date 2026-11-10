"""Sector differences on the full grid of 46 x 45 pairs, 2001-2021, with zeros where no deal occurred.

The thesis panels hold only cells with at least one deal. test_recovery.py shows on synthetic
data that this restriction can bend levels and create a false sectoral difference. This script
asks whether the main results survive when pair-years with no deals are kept.

Outcomes
  value      sum of disclosed value; no deals and undisclosed both give zero
  value_obs  same, dropping cells that have deals but no disclosed value
  deals      number of deals; not affected by disclosure
  any        1 if at least one deal, linear probability model (extensive margin)

usage: python 15_all_pairs.py point
       python 15_all_pairs.py ri --design iid --reps 2000 --workers 10
"""
import argparse, gc, multiprocessing as mp, os, time, warnings
import numpy as np, pandas as pd
from thesis_lib import *

warnings.filterwarnings("ignore")
SEED = 20261106
DROP = os.environ.get("ALL_PAIRS_DROP", "")  # economy removed from both sides, for example CHN; results go to all_pairs/no_CHN
OUT = RESULTS / "all_pairs" / (f"no_{DROP}" if DROP else ""); OUT.mkdir(exist_ok=True, parents=True)
KEEP = ["ln_gdp_o", "ln_gdp_d", "ln_dist", "patents_o_l1", "patents_d_l1", "rd_intensity_gross_d_l1", "rd_intensity_gov_d_l1"]
FE = {"year": "year^tech", "country": "year^tech + iso_o^tech + iso_d^tech", "pair": "year^tech + dyad^tech"}
G = {}


def build_grid():
    cov = load_covariates()
    est = sorted(set(cov.dropna(subset=REGS).iso_o))
    assert len(est) == 46, len(est)
    g = pd.read_csv(PUBLIC / "gravity_cepii_2000_2024.csv.gz", keep_default_na=False, na_values=[""],
                    usecols=["year", "isoo", "isod", "dist", "gdp_o", "gdp_d"], low_memory=False)
    g = g.rename(columns={"isoo": "iso_o", "isod": "iso_d"})
    g = g[g.iso_o.isin(est) & g.iso_d.isin(est) & (g.iso_o != g.iso_d) & g.year.between(2001, 2021)].drop_duplicates(["iso_o", "iso_d", "year"])
    g["ln_gdp_o"], g["ln_gdp_d"], g["ln_dist"] = np.log(g.gdp_o), np.log(g.gdp_d), np.log(g.dist)
    m = pd.read_csv(PUBLIC / "msti_loaded.csv", keep_default_na=False, na_values=[""])
    p = pd.read_csv(PUBLIC / "patents_loaded.csv", keep_default_na=False, na_values=[""]).rename(columns={"patent_applications": "patents"})
    cy = m.merge(p, on=["iso3", "year"], how="outer"); cy["year"] += 1
    for s, k in (("o", "iso_o"), ("d", "iso_d")):
        g = g.merge(cy.rename(columns={"iso3": k, "rd_intensity_gov": f"rd_intensity_gov_{s}_l1", "rd_intensity_gross": f"rd_intensity_gross_{s}_l1", "patents": f"patents_{s}_l1"}), on=[k, "year"], how="left")
    g["dyad"] = g.iso_o + "_" + g.iso_d
    return g.dropna(subset=REGS).reset_index(drop=True)


def stack(grid, key, value, lab):
    """Two rows per pair-year, one per sector. key is -1 for deals outside the grid."""
    n = len(grid)
    ok = key >= 0
    v = np.nan_to_num(value, nan=0.0)
    out = []
    for t, mask in ((1, lab), (0, ~lab)):
        m = ok & mask
        x = grid.copy()
        x["deals"] = np.bincount(key[m], minlength=n)
        x["total_value"] = np.bincount(key[m], weights=v[m], minlength=n)
        x["tech"] = t
        out.append(x)
    return pd.concat(out, ignore_index=True)


def fit(d, y, fe, estimator="pois"):
    regs = REGS if fe != "pair" else [v for v in REGS if v != "ln_dist"]
    fml = f"{y} ~ " + " + ".join(regs + [f"tech:{v}" for v in regs]) + " | " + FE[fe]
    if estimator == "pois":
        m = pf.fepois(fml, data=d, vcov={"CRV1": "dyad"}, store_data=False, copy_data=False, fixef_tol=1e-6)
    else:
        m = pf.feols(fml, data=d, vcov={"CRV1": "dyad"})
    s, wp, k = wald(m, [f"tech:{v}" for v in regs])
    return m, s, wp


def setup():
    warnings.filterwarnings("ignore")
    deals = load_deals()
    grid = build_grid()
    if DROP:
        deals = deals[(deals.iso_o != DROP) & (deals.iso_d != DROP)].reset_index(drop=True)
        grid = grid[(grid.iso_o != DROP) & (grid.iso_d != DROP)].reset_index(drop=True)
    ids = grid.reset_index()[["iso_o", "iso_d", "year", "index"]]
    k = deals[["iso_o", "iso_d", "year"]].merge(ids, on=["iso_o", "iso_d", "year"], how="left")["index"].fillna(-1).astype(int).to_numpy()
    G.update(grid=grid, key=k, val=deals.deal_value.to_numpy(), lab=deals.is_tech_target.to_numpy().astype(bool))
    dy = pd.factorize(deals.iso_o + "_" + deals.iso_d)[0]
    G["dy"], G["ndy"] = dy, dy.max() + 1
    n = np.bincount(dy, minlength=G["ndy"])
    G["sh"] = np.bincount(dy, weights=G["lab"], minlength=G["ndy"]) / n
    G["p"] = G["lab"].mean()


def point():
    setup()
    d = stack(G["grid"], G["key"], G["val"], G["lab"])
    print("grid pair-years", len(G["grid"]), "stacked", len(d), "cells with deals", int((d.deals > 0).sum()), "deals on grid", int((G["key"] >= 0).sum()), flush=True)
    d["any"] = (d.deals > 0).astype(float)
    rows = []
    obs = d[~((d.deals > 0) & (d.total_value == 0))]
    trans = d[d.deals > 0]
    specs = [("value, all pairs", d, "total_value", "pois"), ("value, dropping undisclosed cells", obs, "total_value", "pois"),
             ("deals, all pairs", d, "deals", "pois"), ("deals, transacting only", trans, "deals", "pois"),
             ("any deal, LPM", d, "any", "ols")]
    for name, data, y, est in specs:
        for fe in FE:
            t0 = time.time()
            m, s, wp = fit(data, y, fe, est)
            c, se, p = m.coef(), m.se(), m.pvalue()
            for v in c.index:
                rows.append(dict(spec=name, fe=fe, N=int(m._N), var=v, coef=float(c[v]), se=float(se[v]), p=float(p[v]), wald=s, wald_p=wp))
            print(name, fe, int(m._N), "Wald %.2f p=%.2g" % (s, wp), "%.1fs" % (time.time() - t0), flush=True)
    r = pd.DataFrame(rows); r.to_csv(OUT / "point.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    x = r[r["var"].isin(["tech:ln_gdp_d", "tech:ln_gdp_o", "tech:ln_dist", "tech:patents_d_l1", "tech:patents_o_l1", "ln_gdp_d", "ln_gdp_o", "ln_dist"])]
    print(x[["spec", "fe", "N", "var", "coef", "se", "p", "wald", "wald_p"]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))


def one(args):
    i, design = args
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(i,)))
    n = len(G["lab"])
    lab = rng.random(n) < G["p"] if design == "iid" else rng.random(n) < rng.permutation(G["sh"])[G["dy"]]
    row = {"rep": i}
    try:
        d = stack(G["grid"], G["key"], G["val"], lab)
        for y in ("total_value", "deals"):
            for fe in FE:
                m, s, wp = fit(d, y, fe)
                row[f"{y}_{fe}_wald"], row[f"{y}_{fe}_wald_p"] = s, wp
                c = m.coef()
                for v in ("ln_gdp_d", "ln_gdp_o", "ln_dist", "patents_d_l1", "patents_o_l1"):
                    if f"tech:{v}" in c.index:
                        row[f"{y}_{fe}_{v}"] = float(c[f"tech:{v}"])
                del m
    except Exception as e:
        row["error"] = str(e)[:150]
    gc.collect()
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["point", "ri"])
    ap.add_argument("--design", default="iid")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.mode == "point":
        point()
    else:
        path = OUT / f"ri_{a.design}.csv"
        t0, buf, count, hdr, cols = time.time(), [], 0, True, None
        with mp.Pool(a.workers, initializer=setup, maxtasksperchild=40) as pool:  # workers restart to release memory
            for r in pool.imap_unordered(one, [(i, a.design) for i in range(a.reps)], chunksize=5):
                buf.append(r); count += 1
                if len(buf) >= 100 or count == a.reps:
                    df = pd.DataFrame(buf)
                    if cols is None:
                        cols = list(df.columns) + ([] if "error" in df else ["error"])
                    df.reindex(columns=cols).to_csv(path, mode="a", header=hdr, index=False)
                    hdr, buf = False, []
                    print(count, "%.0fs" % (time.time() - t0), flush=True)
