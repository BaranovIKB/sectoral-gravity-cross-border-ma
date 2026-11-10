"""Three checks suggested by the review of 8 November 2026.

1. Baseline difference test without the largest 1 per cent of cells.
2. Difference between subsamples of Table 5-21, approximate z-test from the published estimates
   (assumes independent subsamples; they share country pairs, so the test is indicative).
3. Technology-share model (Table 5-15) with GDP in constant 2015 USD (World Bank NY.GDP.MKTP.KD)
   instead of current USD.

usage: python 22_cheap_checks.py      output: results/cheap_checks.csv
"""
import warnings
import numpy as np, pandas as pd
from scipy import stats as st
from thesis_lib import *

warnings.filterwarnings("ignore")
rows = []
deals = load_deals()
cov = add_scaled_patents(load_covariates())
key, cov2 = prepare_keys(deals, cov)
val = deals.deal_value.to_numpy(); lab = deals.is_tech_target.to_numpy().astype(bool)

# 1. without the largest 1 per cent of cells
d = stacked_sector_lag(key, val, lab, cov2)
cut = d["total_value"].quantile(0.99)
for name, dd in (("baseline", d), ("without top 1% of cells", d[d.total_value <= cut])):
    for fe, s in ladder_stats(dd).items():
        rows.append(dict(check="top1", spec=name, fe=fe, stat="joint_wald", value=s["wald"], p=s["wald_p"], N=s["N"]))
        for v in ("ln_gdp_d", "ln_dist", "patents_d_l1", "patents_o_l1"):
            if v in s["coef"]:
                rows.append(dict(check="top1", spec=name, fe=fe, stat=v, value=s["coef"][v], p=s["p"][v], N=s["N"]))
print("cut (USD thousand):", round(cut), flush=True)

# 2. subsample differences, Table 5-21
n = pd.read_csv(RESULTS / "new_specs.csv")
for a, b in (("Control deals", "Minority deals"), ("Target above median income", "Target below median income"),
             ("Acquirer above median income", "Acquirer below median income"), ("2001-2010", "2011-2021")):
    for fe in ("year", "country"):
        for v in ("tech:ln_gdp_d", "tech:ln_dist", "tech:patents_d_l1"):
            x = n[(n.spec == a) & (n.fe == fe) & (n["var"] == v)]
            y = n[(n.spec == b) & (n.fe == fe) & (n["var"] == v)]
            if x.empty or y.empty:
                continue
            x, y = x.iloc[0], y.iloc[0]
            z = (x.coef - y.coef) / np.hypot(x.se, y.se)
            rows.append(dict(check="subsample", spec=f"{a} minus {b}", fe=fe, stat=v.replace("tech:", ""), value=x.coef - y.coef,
                             p=2 * st.norm.sf(abs(z)), N=np.nan))

# 3. share model with real GDP
g = pd.read_csv(PUBLIC / "wdi_gdp_real.csv").dropna()
g = g.rename(columns={"iso3": "iso", "value": "gdp_real"})
g["ln_real"] = np.log(g.gdp_real)
c = cov2.loc[cov2.dropna(subset=REGS).index].copy()
cnt = np.bincount(key, minlength=len(cov2)); tch = np.bincount(key, weights=lab, minlength=len(cov2))
c["n"] = cnt[c.index]; c["sh"] = tch[c.index] / c["n"]
c = c[c.n > 0]
for s, k in (("o", "iso_o"), ("d", "iso_d")):
    c = c.merge(g[["iso", "year", "ln_real"]].rename(columns={"iso": k, "ln_real": f"ln_rgdp_{s}"}), on=[k, "year"], how="left")
c = c.dropna(subset=["ln_rgdp_o", "ln_rgdp_d"])
for label, swap in (("current USD", {}), ("constant 2015 USD", {"ln_gdp_o": "ln_rgdp_o", "ln_gdp_d": "ln_rgdp_d"})):
    for fe in ("year", "country", "pair"):
        fes, regs = SHARE_LADDER[fe]
        regs = [swap.get(v, v) for v in regs]
        m = fit_share(c, regs, fes)
        v = swap.get("ln_gdp_d", "ln_gdp_d")
        rows.append(dict(check="real_gdp", spec=label, fe=fe, stat="ln_gdp_d", value=float(m.coef()[v]), p=float(m.pvalue()[v]), N=int(m._N)))

T = pd.DataFrame(rows)
T.to_csv(RESULTS / "cheap_checks.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(T.to_string(float_format=lambda x: f"{x:.4g}"))
