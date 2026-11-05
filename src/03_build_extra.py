"""Assemble country-year and dyad-year extras from revisions/data and CEPII."""
import numpy as np, pandas as pd
from thesis_lib import ROOT, PUBLIC as D
m = pd.read_csv(D / "oecd_msti_all.csv", low_memory=False)
M, U, R, Y = "MEASURE: Measure", "UNIT_MEASURE: Unit of measure", "REF_AREA: Reference area", "TIME_PERIOD: Time period"
m["iso3"] = m[R].str[:3]; m["mc"] = m[M].str.split(":").str[0]; m["uc"] = m[U].str.split(":").str[0]
want = {("P_TRIAD", "PATN"): "triad", ("P_PCT", "PATN"): "pct", ("G_FG", "PT_B1GQ"): "rd_fund",
        ("H", "PT_B1GQ"): "herd", ("GV", "PT_B1GQ"): "goverd"}
cy = None
for (a, b), n in want.items():
    x = m[(m.mc == a) & (m.uc == b)].groupby(["iso3", Y]).OBS_VALUE.first().rename(n).reset_index().rename(columns={Y: "year"})
    cy = x if cy is None else cy.merge(x, on=["iso3", "year"], how="outer")
for n in ("wgi_pv", "wgi_rq", "wgi_rl", "wdi_gdp", "wdi_pop", "wdi_rd", "wdi_gdpcap"):
    x = pd.read_csv(D / f"{n}.csv").dropna(subset=["iso3"]); x = x[x.iso3.str.len() == 3]
    x = x.rename(columns={"value": n})[["iso3", "year", n]]
    cy = cy.merge(x, on=["iso3", "year"], how="outer")
cy = cy.sort_values(["iso3", "year"])
# WGI is biennial before 2002: carry the previous value forward by at most one year
for n in ("wgi_pv", "wgi_rq", "wgi_rl"):
    cy[n] = cy.groupby("iso3")[n].ffill(limit=1)
cy["public_perf"] = cy["goverd"] + cy["herd"]
r = pd.read_csv(D / "oecd_fdirri_scores.csv", low_memory=False)
PC, A, T = "POL_CAT: Policy category", "ACTIVITY: Economic activity", "TIME_PERIOD: Time period"
r["iso3"] = r["REF_AREA: Reference area"].str[:3]
r = r[r[A].str.startswith("_T") & (r[T] == 2018)]
rri = pd.DataFrame({"rri_screen": r[r[PC].str.startswith("CLAS1_2")].set_index("iso3").OBS_VALUE,
                    "rri_total": r[r[PC].str.startswith("_T")].set_index("iso3").OBS_VALUE}).reset_index()
cy = cy.merge(rri, on="iso3", how="left")
cy.to_csv(D / "extra_country_year.csv", index=False)
print(cy.shape, cy.notna().sum().to_dict())
cols = ["year", "iso3_o", "iso3_d", "fta_wto", "diplo_disagreement", "entry_cost_d", "entry_tp_d", "col45", "comcol"]
parts = []
for ch in pd.read_csv(ROOT / "data" / "raw" / "Gravity_V202211.csv", usecols=cols, chunksize=2_000_000, low_memory=False):
    parts.append(ch[(ch.year >= 2000) & (ch.iso3_o != ch.iso3_d) & (ch.fta_wto.notna() | ch.diplo_disagreement.notna())])
g = pd.concat(parts).rename(columns={"iso3_o": "iso_o", "iso3_d": "iso_d"}).drop_duplicates(["iso_o", "iso_d", "year"])
g.to_csv(D / "extra_dyad_year.csv.gz", index=False)
print(g.shape, g.notna().mean().round(3).to_dict())
