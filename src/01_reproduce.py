"""Rebuild the thesis tables that had no local code and compare them with the text.

Writes results/reproduce_table_5_11.csv, reproduce_table_5_12.csv, reproduce_check.csv.
"""
import warnings
import numpy as np, pandas as pd
from thesis_lib import *
warnings.filterwarnings("ignore")

deals = load_deals()
cov = add_scaled_patents(load_covariates())
key, cov2 = prepare_keys(deals, cov)
val = deals.deal_value.to_numpy()
tech = deals.is_tech_target.to_numpy().astype(bool)
acq = deals.is_tech_acquirer.to_numpy().astype(bool)

NARROW = ["357", "48", "737"]


def has_narrow(s):
    """True when any SIC code is in 3570-3579, 4812-4899 or 7370-7374."""
    if not isinstance(s, str):
        return False
    for c in s.split("/"):
        c = c.strip()
        if c.startswith("357") or (c.isdigit() and 4812 <= int(c) <= 4899) or (c.isdigit() and 7370 <= int(c) <= 7374):
            return True
    return False


narrow = deals["TARGET_SCUSCD"].map(has_narrow).to_numpy() & tech


def run(label, lab, mask=None, regs_patent=None, exclude=None, fes=("year", "country", "pair"), cluster=None):
    k, v, l = key, val, lab
    if mask is not None:
        k, v, l = key[mask], val[mask], lab[mask]
    d = stacked_sector_lag(k, v, l, cov2)
    if exclude:
        d = d[~d.iso_o.isin(exclude) & ~d.iso_d.isin(exclude)]
    rows = []
    for name in fes:
        ff, regs = LADDER[name]
        regs = list(regs)
        pd_name = "patents_d_l1"
        if regs_patent:
            regs = [regs_patent + v[7:] if v.startswith("patents") else v for v in regs]
            pd_name = regs_patent + "_d_l1"
        m = fit_stacked(d, regs, ff, pair_filter=name == "pair")
        c, p = m.coef(), m.pvalue()
        rows.append(dict(spec=label, fe=name, N=int(m._N),
                         gdp_d=c["tech:ln_gdp_d"], gdp_d_p=p["tech:ln_gdp_d"],
                         pat_d=c[f"tech:{pd_name}"], pat_d_p=p[f"tech:{pd_name}"],
                         dist=c.get("tech:ln_dist", np.nan), dist_p=p.get("tech:ln_dist", np.nan)))
    return rows


rows = []
rows += run("Baseline", tech)
rows += run("Patents per USD bn GDP", tech, regs_patent="pat_gdp")
rows += run("Patents per USD bn GDP, excl. China", tech, regs_patent="pat_gdp", exclude=["CHN"])
rows += run("Patents per million inhabitants", tech, regs_patent="pat_pop")
rows += run("Patents per million inhabitants, excl. China", tech, regs_patent="pat_pop", exclude=["CHN"])
rows += run("Classified by acquirer industry", acq, fes=("year", "country"))
both = tech == acq
rows += run("Both parties technology or both not", tech, mask=both, fes=("year", "country"))
rows += run("Narrow technology definition", narrow, fes=("year", "country"))
for c in ["LUX", "NLD", "SGP", "CHE", "IRL"]:
    rows += run(f"Excluding {c}", tech, exclude=[c], fes=("year", "country"))
rows += run("Excluding LUX, NLD, SGP, CHE, IRL", tech, exclude=["LUX", "NLD", "SGP", "CHE", "IRL"], fes=("year", "country"))
t = pd.DataFrame(rows)
t.to_csv(RESULTS / "reproduce_table_5_11.csv", index=False)
print(t.to_string(float_format=lambda x: f"{x:.5g}"))

# Table 5-12: alternative clustering
d = stacked_sector_lag(key, val, tech, cov2)
rows = []
for cl_name, cl in (("pair", {"CRV1": "dyad"}), ("acquirer", {"CRV1": "iso_o"}), ("target", {"CRV1": "iso_d"}), ("twoway", {"CRV1": "iso_o+iso_d"})):
    for name in ("year", "country"):
        ff, regs = LADDER[name]
        sub = d.dropna(subset=regs + ["total_value"])
        fml = "total_value ~ " + " + ".join(regs + [f"tech:{v}" for v in regs]) + " | " + " + ".join(ff)
        m = pf.fepois(fml, data=sub, vcov=cl, store_data=False, copy_data=False)
        s, p = m.se(), m.pvalue()
        row = dict(cluster=cl_name, fe=name)
        for v in ("ln_gdp_d", "patents_d_l1", "ln_dist", "patents_o_l1"):
            row[v + "_se"], row[v + "_p"] = s[f"tech:{v}"], p[f"tech:{v}"]
        rows.append(row)
t = pd.DataFrame(rows)
t.to_csv(RESULTS / "reproduce_table_5_12.csv", index=False)
print(t.to_string(float_format=lambda x: f"{x:.4g}"))

# common lag grid as a sensitivity
d = panels_from_labels(key, val, tech, cov2)
r = ladder_stats(d)
print("common lag grid:", {k: (v["N"], round(v["wald"], 2), round(v["coef"]["ln_gdp_d"], 4), float(f"{v['coef']['patents_d_l1']:.4g}")) for k, v in r.items()})
