"""Randomisation inference for the patent-quality specifications of Table 5-19.

Same designs, seed and replication streams as 02_ri_placebo.py, so replication i relabels
the deals exactly as it does for Table 5-6. Only the full sample is run.

usage: python 20_ri_patent_quality.py --check
       python 20_ri_patent_quality.py --design iid --reps 10000 --workers 10
       python 20_ri_patent_quality.py --design dyadshare --reps 2000 --workers 10
       python 20_ri_patent_quality.py --summary
Output: results/ri/ri_quality_{design}.csv, results/ri_patent_quality.csv
"""
import argparse, multiprocessing as mp, time, warnings
import numpy as np, pandas as pd
from scipy import stats as st
from thesis_lib import *

SEED = 20261105
SPECS = {"triad": "Triadic families", "log_triad": "Triadic families, log(1+x)",
         "pct": "PCT filings", "log_pct": "PCT filings, log(1+x)"}
G = {}


def swap(regs, new):
    return [v.replace("patents", new) if v.startswith("patents") else v for v in regs]


def add_logs(d):
    return d.assign(log_triad_o_l1=np.log1p(d.triad_o_l1), log_triad_d_l1=np.log1p(d.triad_d_l1),
                    log_pct_o_l1=np.log1p(d.pct_o_l1), log_pct_d_l1=np.log1p(d.pct_d_l1))


def load():
    deals = load_deals()
    cov = add_scaled_patents(load_covariates())
    n0 = len(cov)
    cy = pd.read_csv(PUBLIC / "extra_country_year.csv")
    x = cy[["iso3", "year", "triad", "pct"]].copy(); x["year"] += 1
    for s, k in (("o", "iso_o"), ("d", "iso_d")):
        cov = cov.merge(x.rename(columns={"iso3": k, "triad": f"triad_{s}_l1", "pct": f"pct_{s}_l1"}), on=[k, "year"], how="left")
    assert len(cov) == n0
    key, cov2 = prepare_keys(deals, cov)
    return deals, key, cov2


def init():
    warnings.filterwarnings("ignore")
    deals, key, cov2 = load()
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


def fit_all(d):
    """Joint Wald and the two patent interactions for every measure and FE structure."""
    row = {}
    for new in SPECS:
        for fe, (ff, regs) in LADDER.items():
            rr = swap(regs, new)
            m = fit_stacked(d, rr, ff, pair_filter=fe == "pair")
            c, p = m.coef(), m.pvalue()
            s, wp, _ = wald(m, [f"tech:{v}" for v in rr])
            row[f"{new}_{fe}_N"], row[f"{new}_{fe}_wald"], row[f"{new}_{fe}_wald_p"] = int(m._N), s, wp
            for side in ("o", "d"):
                v = f"tech:{new}_{side}_l1"
                row[f"{new}_{fe}_{side}"], row[f"{new}_{fe}_{side}_p"] = float(c[v]), float(p[v])
    return row


def one(args):
    i, design = args
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(i,)))
    lab = draw(design, rng)
    row = {"rep": i}
    try:
        row.update(fit_all(add_logs(stacked_sector_lag(G["key"], G["val"], lab, G["cov"]))))
    except Exception as e:
        row["error"] = str(e)[:200]
    return row


def check():
    """Stop unless this code reproduces Table 5-19 and the relabelling behind Table 5-6."""
    init()
    obs = fit_all(add_logs(stacked_sector_lag(G["key"], G["val"], G["lab"], G["cov"])))
    pub = pd.read_csv(RESULTS / "new_specs.csv")
    pub = pub[pub.table == "patent_quality"]
    worst = 0
    for new, nm in SPECS.items():
        for fe in LADDER:
            for side in ("o", "d"):
                r = pub[(pub.spec == f"{nm}|full") & (pub.fe == fe) & (pub["var"] == f"tech:{new}_{side}_l1")].iloc[0]
                worst = max(worst, abs(obs[f"{new}_{fe}_{side}"] / r.coef - 1), abs(obs[f"{new}_{fe}_{side}_p"] - r.p))
                assert obs[f"{new}_{fe}_N"] == r.N, (new, fe, obs[f"{new}_{fe}_N"], r.N)
    print("Table 5-19 worst relative or p difference", worst)
    assert worst < 1e-5
    for design in ("iid", "dyadshare"):
        B = pd.read_csv(RESULTS / "ri" / f"ri_{design}.csv", usecols=["rep", "year_patents_d_l1", "pair_patents_d_l1"]).set_index("rep")
        for i in (0, 7):
            lab = draw(design, np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(i,))))
            st_ = ladder_stats(stacked_sector_lag(G["key"], G["val"], lab, G["cov"]))
            a = st_["year"]["coef"]["patents_d_l1"], st_["pair"]["coef"]["patents_d_l1"]
            b = B.loc[i, "year_patents_d_l1"], B.loc[i, "pair_patents_d_l1"]
            print(design, i, a, b)
            assert abs(a[0] / b[0] - 1) < 1e-5 and abs(a[1] / b[1] - 1) < 1e-5
    print("check passed")


def summary():
    pub = pd.read_csv(RESULTS / "new_specs.csv")
    pub = pub[pub.table == "patent_quality"]
    rows = []
    for design in ("iid", "dyadshare"):
        f = RESULTS / "ri" / f"ri_quality_{design}.csv"
        if not f.exists():
            continue
        B = pd.read_csv(f).drop_duplicates("rep").sort_values("rep")
        print(design, len(B), "errors", int(B["error"].notna().sum()) if "error" in B else 0)
        for new, nm in SPECS.items():
            for fe in LADDER:
                sub = pub[(pub.spec == f"{nm}|full") & (pub.fe == fe)]
                for stat, col, ob, nom in (
                        ("joint_wald", f"{new}_{fe}_wald", sub.wald.iloc[0], sub.wald_p.iloc[0]),
                        ("tech_o", f"{new}_{fe}_o", *sub[sub["var"] == f"tech:{new}_o_l1"][["coef", "p"]].iloc[0]),
                        ("tech_d", f"{new}_{fe}_d", *sub[sub["var"] == f"tech:{new}_d_l1"][["coef", "p"]].iloc[0])):
                    x = B[col].dropna().to_numpy(); N = len(x)
                    r = int((np.abs(x) >= abs(ob)).sum()); p = (r + 1) / (N + 1)
                    pc = B[col + "_p"].dropna()
                    rows.append(dict(design=design, measure=nm, fe=fe, stat=stat, estimate=ob, nominal_p=nom, N=N, r=r, ri_p=p,
                                     mc_se=np.sqrt(p * (1 - p) / N), ci_lo=st.beta.ppf(.025, r, N - r + 1) if r else 0.0,
                                     ci_hi=st.beta.ppf(.975, r + 1, N - r), placebo_p95=float(np.quantile(np.abs(x), .95)),
                                     reject_5pct=float((pc < .05).mean())))
    T = pd.DataFrame(rows)
    T.to_csv(RESULTS / "ri_patent_quality.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print(T[["design", "measure", "fe", "stat", "estimate", "nominal_p", "ri_p", "mc_se", "reject_5pct"]].to_string(float_format=lambda v: f"{v:.5g}"))


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
        path = RESULTS / "ri" / f"ri_quality_{a.design}.csv"
        done = set(pd.read_csv(path, usecols=["rep"])["rep"]) if path.exists() else set()
        todo = [(i, a.design) for i in range(a.reps) if i not in done]
        print(f"{a.design}: {len(done)} done, {len(todo)} to run, seed {SEED}", flush=True)
        t0 = time.time(); buf, hdr, count = [], not path.exists(), 0
        cols = list(pd.read_csv(path, nrows=0).columns) if not hdr else None
        with mp.Pool(a.workers, initializer=init) as pool:
            for r in pool.imap_unordered(one, todo, chunksize=5):
                buf.append(r); count += 1
                if len(buf) >= 100 or count == len(todo):
                    df = pd.DataFrame(buf)
                    if cols is None:
                        cols = list(df.columns)
                        if "error" not in cols:
                            cols.append("error")
                    df.reindex(columns=cols).to_csv(path, mode="a", header=hdr, index=False)
                    hdr, buf = False, []
                    print(f"{count}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
