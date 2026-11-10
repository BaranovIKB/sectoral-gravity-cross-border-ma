"""Summarise the placebo draws: RI p, Monte Carlo error, size distortion.

RI p = (r + 1) / (N + 1), r = draws with |stat| >= |observed|.
Monte Carlo SE = sqrt(p (1 - p) / N); the interval is Clopper-Pearson on r of N.
"""
import warnings
import numpy as np, pandas as pd
from scipy import stats as st
from thesis_lib import *
warnings.filterwarnings("ignore")

deals = load_deals()
cov = add_scaled_patents(load_covariates())
key, cov2 = prepare_keys(deals, cov)
val = deals.deal_value.to_numpy(); lab = deals.is_tech_target.to_numpy().astype(bool)
d = stacked_sector_lag(key, val, lab, cov2)
obs = ladder_stats(d)
regs = [v for v in REGS if not v.startswith("patents")] + ["pat_gdp_o_l1", "pat_gdp_d_l1"]
m = fit_stacked(d, regs, ["year^tech"])
obs_pg = float(m.coef()["tech:pat_gdp_d_l1"]); obs_pg_p = float(m.pvalue()["tech:pat_gdp_d_l1"])
idx = cov2.dropna(subset=REGS).index.to_numpy()
c = cov2.loc[idx].copy()
n = np.bincount(key, minlength=len(cov2)); t = np.bincount(key, weights=lab, minlength=len(cov2))
c["n"] = n[idx]; c["sh"] = t[idx] / c["n"]
share = {k: fit_share(c, r, f) for k, (f, r) in SHARE_LADDER.items()}


def line(design, B, block, name, ob, nominal, col):
    x = B[col].dropna().to_numpy(); N = len(x)
    r = int((np.abs(x) >= abs(ob)).sum())
    p = (r + 1) / (N + 1)
    lo = st.beta.ppf(.025, r, N - r + 1) if r > 0 else 0.0
    hi = st.beta.ppf(.975, r + 1, N - r)
    return dict(design=design, block=block, stat=name, estimate=ob, nominal_p=nominal, N=N, r=r, ri_p=p,
                mc_se=np.sqrt(p * (1 - p) / N), ci_lo=lo, ci_hi=hi, placebo_p95=float(np.quantile(np.abs(x), .95)))


rows, size, size_coef = [], [], []
for design in ("iid", "dyadshare"):
    f = RESULTS / "ri" / f"ri_{design}.csv"
    if not f.exists():
        continue
    B = pd.read_csv(f).drop_duplicates("rep").sort_values("rep")
    print(design, len(B), "errors", int(B.get("error", pd.Series(dtype=str)).notna().sum()))
    for block in ("year", "country", "pair"):
        o = obs[block]
        rows.append(line(design, B, block, "joint_wald", o["wald"], o["wald_p"], f"{block}_wald"))
        for v, b in o["coef"].items():
            rows.append(line(design, B, block, v, b, o["p"][v], f"{block}_{v}"))
        rej = (B[f"{block}_wald_p"] < .05).mean(); k = len(B)
        size.append(dict(design=design, block=block, reject_5pct=rej, mc_se=np.sqrt(rej * (1 - rej) / k), N=k,
                         median_N=B[f"{block}_N"].median(), wald_p95=B[f"{block}_wald"].quantile(.95)))
        for v in o["coef"]:
            x = B[f"{block}_{v}_p"].dropna()
            size_coef.append(dict(design=design, block=block, stat=v, reject_5pct=(x < .05).mean(), N=len(x)))
    rows.append(line(design, B, "year", "pat_gdp_d", obs_pg, obs_pg_p, "year_pat_gdp_d"))
    for k, msh in share.items():
        co, pv = msh.coef(), msh.pvalue()
        for v in co.index:
            col = f"share_{k}_{v}"
            if col in B:
                rows.append(line(design, B, f"share_{k}", v, float(co[v]), float(pv[v]), col))
T = pd.DataFrame(rows); S = pd.DataFrame(size)
T.to_csv(RESULTS / "ri_table.csv", index=False); S.to_csv(RESULTS / "ri_size.csv", index=False)
pd.DataFrame(size_coef).to_csv(RESULTS / "ri_size_coef.csv", index=False)
pd.set_option("display.width", 200)
print(S.to_string()); print(T.to_string(float_format=lambda x: f"{x:.5g}"))
