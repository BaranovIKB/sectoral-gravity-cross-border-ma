"""Generate a synthetic deal table with the same columns as the Orbis extract.

No Orbis record, value, count, country pair or share enters this script. Its only inputs
are the public CEPII and WIPO files and the true parameters below, which are round values
chosen to resemble the published estimates in the thesis. Every deal is invented.

The purpose is to let anyone run the whole pipeline and to test the code: the estimators
should recover TRUE up to sampling noise. Results on these data say nothing about real M&A.
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "data" / "public"
OUT = ROOT / "data" / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261105
TARGET_DEALS = 120_000
TECH_SIC = ["3571", "3572", "4813", "4899", "7371", "7372", "7373", "7374", "3559", "3561"]
OTHER_SIC = ["1000", "2000", "2834", "3714", "5000", "6021", "7011", "8000"]
TRUE = {  # log points per unit: non-technology, technology
    "ln_gdp_o": (0.90, 0.90), "ln_gdp_d": (1.00, 0.65), "ln_dist": (-0.40, -0.20),
    "patents_d": (-1.5e-6, 0.0),
}

rng = np.random.default_rng(SEED)
g = pd.read_csv(PUBLIC / "gravity_cepii_2000_2024.csv.gz", keep_default_na=False, na_values=[""], low_memory=False)
g = g[g.isoo != g.isod].drop_duplicates(["isoo", "isod", "year"])
g = g[["year", "isoo", "isod", "dist", "gdp_o", "gdp_d"]].sort_values(["isoo", "isod", "year"])
# CEPII GDP stops in 2021: hold the last value so that 2022-2024 can also carry deals
g[["gdp_o", "gdp_d", "dist"]] = g.groupby(["isoo", "isod"])[["gdp_o", "gdp_d", "dist"]].ffill()
g = g.dropna(subset=["gdp_o", "gdp_d", "dist"])
g = g[(g.gdp_o > 0) & (g.gdp_d > 0) & (g.dist > 0)]
p = pd.read_csv(PUBLIC / "patents_loaded.csv", keep_default_na=False, na_values=[""])
p = p.rename(columns={"iso3": "isod", "patent_applications": "patents_d"})[["isod", "year", "patents_d"]]
p["year"] += 1
g = g.merge(p, on=["isod", "year"], how="left")
g["patents_d"] = g.patents_d.fillna(0)
dy = g[["isoo", "isod"]].drop_duplicates()
dy["u"] = rng.normal(0, 1.0, len(dy))  # persistent pair effect
g = g.merge(dy, on=["isoo", "isod"])

out = []
for tech in (0, 1):
    b = {k: v[tech] for k, v in TRUE.items()}
    eta = (b["ln_gdp_o"] * np.log(g.gdp_o) + b["ln_gdp_d"] * np.log(g.gdp_d) + b["ln_dist"] * np.log(g.dist)
           + b["patents_d"] * g.patents_d + g.u).to_numpy()
    lam = np.exp(eta - eta.max())
    lam *= TARGET_DEALS * (0.30 if tech else 0.70) / lam.sum()
    n = rng.poisson(lam)
    idx = np.repeat(np.arange(len(g)), n)
    k = len(idx)
    x = g.iloc[idx][["isoo", "isod", "year"]].rename(columns={"isoo": "iso_o", "isod": "iso_d"}).reset_index(drop=True)
    x["is_tech_target"] = tech
    x["deal_value"] = np.where(rng.random(k) < 0.655, rng.lognormal(9.0, 1.8, k), np.nan)
    x["is_tech_acquirer"] = (rng.random(k) < (0.5 if tech else 0.1)).astype(int)
    x["TARGET_SCUSCD"] = rng.choice(TECH_SIC if tech else OTHER_SIC, k)
    x["BIDDER_SCUSCD"] = np.where(x.is_tech_acquirer == 1, rng.choice(TECH_SIC, k), rng.choice(OTHER_SIC, k))
    minority = rng.random(k) < 0.5
    x["FINLSTAK"] = np.where(minority, "20.00000", "100.00000")
    x["STAKE"] = x["FINLSTAK"]
    x["DEALTYPE"] = np.where(minority, "Minority stake", "Acquisition")
    x["BIDDER_STRATEGIC_CTRYC"] = ""
    out.append(x)
d = pd.concat(out, ignore_index=True)
d.to_csv(OUT / "deals_synthetic.csv", index=False)
print(f"{len(d):,} synthetic deals, {d.is_tech_target.mean():.3f} technology, {d.deal_value.notna().mean():.3f} disclosed, seed {SEED}")
