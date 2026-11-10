"""Placebo p-values for the all-pairs regressions in 15_all_pairs.py, with and without China.

p = (r + 1) / (N + 1), r = draws at least as extreme as observed. Coefficients are compared in absolute value.
Size is the share of placebo draws that the nominal cluster-robust Wald test rejects at 5%.
Writes ri_summary.csv in all_pairs/ and all_pairs/no_CHN/.
"""
import numpy as np, pandas as pd
from pathlib import Path

from thesis_lib import RESULTS
BASE = RESULTS / "all_pairs"
SPECS = {"total_value": "value, all pairs", "deals": "deals, all pairs"}
VARS = ["ln_gdp_d", "ln_gdp_o", "ln_dist", "patents_d_l1", "patents_o_l1"]


def pvalue(extreme, n):
    p = (int(extreme) + 1) / (n + 1)
    return p, (p * (1 - p) / n) ** .5


for OUT in (BASE, BASE / "no_CHN"):
    pt = pd.read_csv(OUT / "point.csv")
    rows = []
    for design in ("iid", "dyadshare"):
        r = pd.read_csv(OUT / f"ri_{design}.csv")
        err = int(r["error"].notna().sum())
        print(OUT.name, design, "draws", len(r), "unique", r["rep"].nunique(), "errors", err)
        for y, spec in SPECS.items():
            for fe in ("year", "country", "pair"):
                p = pt[(pt.spec == spec) & (pt.fe == fe)]
                draws = r[f"{y}_{fe}_wald"].dropna().to_numpy()
                obs = float(p.wald.iloc[0])
                pv, se = pvalue((draws >= obs).sum(), len(draws))
                rows.append(dict(design=design, y=y, fe=fe, var="joint Wald", obs=obs, nominal_p=float(p.wald_p.iloc[0]), ri_p=pv, mc_se=se,
                                 N=len(draws), wald_size=float((r[f"{y}_{fe}_wald_p"].dropna() < .05).mean())))
                for v in VARS:
                    c = f"{y}_{fe}_{v}"
                    if c not in r or f"tech:{v}" not in set(p["var"]):
                        continue
                    draws = r[c].dropna().to_numpy()
                    q = p[p["var"] == f"tech:{v}"].iloc[0]
                    pv, se = pvalue((np.abs(draws) >= abs(q.coef)).sum(), len(draws))
                    rows.append(dict(design=design, y=y, fe=fe, var=v, obs=float(q.coef), nominal_p=float(q.p), ri_p=pv, mc_se=se, N=len(draws), wald_size=np.nan))
    pd.DataFrame(rows).to_csv(OUT / "ri_summary.csv", index=False)
