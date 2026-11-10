"""Joint test of sector differences when one economy is removed from both sides of the deal.

Runs on the thesis panels (transacting cells) and on all pairs with zeros, for every economy.
Tells whether the loss of the within-economy joint difference without China is special to China.
Writes results/all_pairs/leave_one_out.csv. Nominal cluster-robust Wald only.
"""
import importlib.util, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from thesis_lib import *

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ap", HERE / "15_all_pairs.py")
ap = importlib.util.module_from_spec(spec); sys.modules["ap"] = ap; spec.loader.exec_module(ap)

deals = load_deals()
key, cov = prepare_keys(deals, load_covariates())
thesis = stacked_sector_lag(key, deals.deal_value.to_numpy(), deals.is_tech_target.to_numpy().astype(bool), cov)
ap.setup()
G = ap.G
allp = ap.stack(G["grid"], G["key"], G["val"], G["lab"])
econ = sorted(set(G["grid"].iso_o))
rows = []


def thesis_row(name):
    d = thesis if name == "none" else thesis[(thesis.iso_o != name) & (thesis.iso_d != name)]
    for fe, s in ladder_stats(d).items():
        rows.append(dict(drop=name, panel="thesis, value", fe=fe, N=s["N"], wald=s["wald"], wald_p=s["wald_p"], gdp_d=s["coef"].get("ln_gdp_d"),
                         pat_d=s["coef"].get("patents_d_l1"), pat_o=s["coef"].get("patents_o_l1"), dist=s["coef"].get("ln_dist")))


def allp_row(name):
    d = allp if name == "none" else allp[(allp.iso_o != name) & (allp.iso_d != name)]
    for y in ("total_value", "deals"):
        for fe in ap.FE:
            m, s, wp = ap.fit(d, y, fe)
            c = m.coef()
            rows.append(dict(drop=name, panel=f"all pairs, {y}", fe=fe, N=int(m._N), wald=s, wald_p=wp, gdp_d=c.get("tech:ln_gdp_d"),
                             pat_d=c.get("tech:patents_d_l1"), pat_o=c.get("tech:patents_o_l1"), dist=c.get("tech:ln_dist")))
            del m


for e in ["none"] + econ:
    thesis_row(e); allp_row(e)
    pd.DataFrame(rows).to_csv(ap.OUT / "leave_one_out.csv", index=False)
    print(e, flush=True)
r = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_rows", 400)
for (p, fe), g in r.groupby(["panel", "fe"], sort=False):
    base = g[g["drop"] == "none"].iloc[0]
    x = g[g["drop"] != "none"].sort_values("wald")
    print(f"\n{p} | {fe} | full Wald {base.wald:.2f} p={base.wald_p:.2g} | economies with p>.05: {int((x.wald_p > .05).sum())}")
    print(x.head(5)[["drop", "N", "wald", "wald_p", "gdp_d", "pat_d", "pat_o"]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
