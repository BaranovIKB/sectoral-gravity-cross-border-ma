"""Recompute the published RI p-values from the published placebo draws. Needs no Orbis data.

Each p-value in results/thesis/ri_table.csv is rebuilt from results/thesis/ri/*.csv.gz and the
observed estimate: p = (r+1)/(N+1), r = draws with |stat| >= |observed|.
"""
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[1] / "results" / "thesis"
T = pd.read_csv(R / "ri_table.csv")
bad = checked = 0
for design in ("iid", "dyadshare"):
    B = pd.read_csv(R / "ri" / f"ri_{design}.csv.gz").drop_duplicates("rep")
    for _, row in T[T.design == design].iterrows():
        block, stat = row.block, row.stat
        if stat == "joint_wald":
            col = f"{block}_wald"
        elif stat == "pat_gdp_d":
            col = "year_pat_gdp_d"
        else:
            col = f"{block}_{stat}"
        x = B[col].dropna().to_numpy()
        p = ((np.abs(x) >= abs(row.estimate)).sum() + 1) / (len(x) + 1)
        checked += 1
        if abs(p - row.ri_p) > 1e-9:
            bad += 1
            print("MISMATCH", design, block, stat, p, row.ri_p)
    rej = {b: (B[f"{b}_wald_p"] < .05).mean() for b in ("year", "country", "pair")}
    print(design, "draws", len(B), "rejection at 5%:", {k: round(v, 4) for k, v in rej.items()})
print(f"{checked} p-values checked, {bad} mismatches")
raise SystemExit(1 if bad else 0)
