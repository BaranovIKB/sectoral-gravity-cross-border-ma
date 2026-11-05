"""Does the estimator recover the true parameters of the synthetic deals?

Two estimates on the same invented deals, 2001-2021:
  all pairs     every pair-year, with zero where no deal was drawn
  transacting   only pair-years with at least one deal, as in the thesis panels

The gap between them shows what restricting a panel to transacting pairs does on data where
the truth is known. It needs no Orbis data, and it says nothing about the size of the effect in Orbis.
"""
import warnings
import numpy as np, pandas as pd, pyfixest as pf
from thesis_lib import ROOT, PUBLIC, RESULTS, _READ
warnings.filterwarnings("ignore")

TRUE = {"ln_gdp_o": 0.90, "ln_gdp_d": 1.00, "ln_dist": -0.40, "patents_d_l1": -1.5e-6,
        "tech:ln_gdp_o": 0.0, "tech:ln_gdp_d": -0.35, "tech:ln_dist": 0.20, "tech:patents_d_l1": 1.5e-6}
d = pd.read_csv(ROOT / "data" / "synthetic" / "deals_synthetic.csv", usecols=["iso_o", "iso_d", "year", "deal_value", "is_tech_target"], **_READ)
d = d[(d.year >= 2001) & (d.year <= 2021)]
agg = d.groupby(["iso_o", "iso_d", "year", "is_tech_target"]).agg(total_value=("deal_value", "sum"), deals=("year", "size")).reset_index().rename(columns={"is_tech_target": "tech"})
g = pd.read_csv(PUBLIC / "gravity_cepii_2000_2024.csv.gz", **_READ).rename(columns={"isoo": "iso_o", "isod": "iso_d"})
g = g[(g.iso_o != g.iso_d) & (g.year >= 2001) & (g.year <= 2021)].drop_duplicates(["iso_o", "iso_d", "year"])
g = g.dropna(subset=["gdp_o", "gdp_d", "dist"])
g = g[(g.gdp_o > 0) & (g.gdp_d > 0) & (g.dist > 0)][["iso_o", "iso_d", "year", "gdp_o", "gdp_d", "dist"]]
g["ln_gdp_o"], g["ln_gdp_d"], g["ln_dist"] = np.log(g.gdp_o), np.log(g.gdp_d), np.log(g.dist)
p = pd.read_csv(PUBLIC / "patents_loaded.csv", **_READ).rename(columns={"iso3": "iso_d", "patent_applications": "patents_d_l1"})[["iso_d", "year", "patents_d_l1"]]
p["year"] += 1
g = g.merge(p, on=["iso_d", "year"], how="left")
g["patents_d_l1"] = g.patents_d_l1.fillna(0)
full = pd.concat([g.assign(tech=t) for t in (0, 1)], ignore_index=True).merge(agg, on=["iso_o", "iso_d", "year", "tech"], how="left")
full[["total_value", "deals"]] = full[["total_value", "deals"]].fillna(0)
regs = ["ln_gdp_o", "ln_gdp_d", "ln_dist", "patents_d_l1"]
fml = "total_value ~ " + " + ".join(regs + [f"tech:{v}" for v in regs]) + " | year^tech"
full["dyad"] = full.iso_o + "_" + full.iso_d
rows = []
for name, data in (("all pairs", full), ("transacting", full[full.deals > 0])):
    m = pf.fepois(fml, data=data, vcov={"CRV1": "dyad"}, store_data=False, copy_data=False)
    c, s = m.coef(), m.se()
    for v, t in TRUE.items():
        rows.append(dict(sample=name, var=v, true=t, estimate=float(c[v]), se=float(s[v]), z_from_true=(float(c[v]) - t) / float(s[v]), N=int(m._N)))
r = pd.DataFrame(rows)
r.to_csv(RESULTS / "recovery_test.csv", index=False)
pd.set_option("display.width", 160)
print(r.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
