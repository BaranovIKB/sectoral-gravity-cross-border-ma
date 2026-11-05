"""Shared helpers for the replication package.

The deal table is the only licensed input. It is read from
  data/synthetic/deals_synthetic.csv   THESIS_DATA=synthetic (default, generated here)
  data/orbis/deals_min.csv             THESIS_DATA=orbis     (your own Orbis extract)
All other inputs are public. Results go to results/<THESIS_DATA>/.
"""
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "data" / "public"
DATA_TAG = os.environ.get("THESIS_DATA", "synthetic")
DEALS_FILE = Path(os.environ.get("THESIS_DEALS", ROOT / "data" / ("synthetic/deals_synthetic.csv" if DATA_TAG == "synthetic" else "orbis/deals_min.csv")))
FAST = os.environ.get("THESIS_FAST", "0") == "1"
RESULTS = ROOT / "results" / DATA_TAG
RESULTS.mkdir(parents=True, exist_ok=True)

YEAR_MIN, YEAR_MAX = 2000, 2024
GRAV = ["ln_gdp_o", "ln_gdp_d", "ln_dist"]
INNOV = ["rd_intensity_gov_o_l1", "rd_intensity_gov_d_l1",
         "rd_intensity_gross_o_l1", "rd_intensity_gross_d_l1",
         "patents_o_l1", "patents_d_l1"]
REGS = GRAV + INNOV
LADDER = {
    "year": (["year^tech"], REGS),
    "country": (["year^tech", "iso_o^tech", "iso_d^tech"], REGS),
    "pair": (["year^tech", "dyad^tech"], [v for v in REGS if v != "ln_dist"]),
}
_READ = dict(keep_default_na=False, na_values=[""], low_memory=False)  # "NA" is Namibia


def load_deals():
    """Cross-border deals in the 2000-2024 analysis window."""
    if not DEALS_FILE.exists():
        raise FileNotFoundError(f"{DEALS_FILE} not found. Run make_synthetic_deals.py, or see data/orbis/README.md")
    d = pd.read_csv(DEALS_FILE, usecols=["iso_o", "iso_d", "year", "deal_value", "is_tech_target", "is_tech_acquirer",
                                         "TARGET_SCUSCD", "BIDDER_SCUSCD", "BIDDER_STRATEGIC_CTRYC", "STAKE", "FINLSTAK", "DEALTYPE"],
                    dtype={c: str for c in ("TARGET_SCUSCD", "BIDDER_SCUSCD", "BIDDER_STRATEGIC_CTRYC", "STAKE", "FINLSTAK", "DEALTYPE")}, **_READ)
    d = d[(d.year >= YEAR_MIN) & (d.year <= YEAR_MAX) & (d.iso_o != d.iso_d)].copy()
    d["deal_value"] = pd.to_numeric(d["deal_value"], errors="coerce")
    return d.reset_index(drop=True)


def load_gravity():
    g = pd.read_csv(PUBLIC / "gravity_cepii_2000_2024.csv.gz", **_READ)
    return g.rename(columns={"isoo": "iso_o", "isod": "iso_d"}).drop_duplicates(["iso_o", "iso_d", "year"])


def load_covariates(extra=()):
    """One row per dyad-year with at least one deal: gravity variables and lagged innovation regressors."""
    deals = load_deals()
    c = deals[["iso_o", "iso_d", "year"]].drop_duplicates()
    g = load_gravity()
    c = c.merge(g, on=["iso_o", "iso_d", "year"], how="left")
    c["ln_gdp_o"], c["ln_gdp_d"], c["ln_dist"] = np.log(c.gdp_o), np.log(c.gdp_d), np.log(c.dist)
    m = pd.read_csv(PUBLIC / "msti_loaded.csv", **_READ)
    p = pd.read_csv(PUBLIC / "patents_loaded.csv", **_READ)
    cy = m.merge(p, on=["iso3", "year"], how="outer").rename(columns={"patent_applications": "patents"})
    for v in ("rd_intensity_gov", "rd_intensity_gross", "patents"):
        cy[v] = cy[v].clip(lower=0)
    cy["year"] += 1
    for s, k in (("o", "iso_o"), ("d", "iso_d")):
        x = cy.rename(columns={"iso3": k, "rd_intensity_gov": f"rd_intensity_gov_{s}_l1",
                               "rd_intensity_gross": f"rd_intensity_gross_{s}_l1", "patents": f"patents_{s}_l1"})
        c = c.merge(x, on=[k, "year"], how="left")
    # as in the thesis panels, a lag exists only if the country appears on that side in t-1
    c = sector_grid_lag(c)
    return c[["iso_o", "iso_d", "year"] + REGS + list(extra)].reset_index(drop=True)


def panels_from_labels(deals_key, value, label, cov):
    """Rebuild the stacked panel from one tech label per deal."""
    v = np.nan_to_num(value, nan=0.0)
    n = len(cov)
    out = []
    for tech, mask in ((1, label), (0, ~label)):
        cnt = np.bincount(deals_key[mask], minlength=n)
        val = np.bincount(deals_key[mask], weights=v[mask], minlength=n)
        keep = cnt > 0
        p = cov.loc[keep].copy()
        p["total_value"] = val[keep]
        p["tech"] = tech
        out.append(p)
    return pd.concat(out, ignore_index=True)


def prepare_keys(deals, cov):
    """Integer dyad-year code per deal, aligned with cov's row order."""
    cov = cov.reset_index(drop=True).copy()
    cov["dyad"] = cov["iso_o"].astype(str) + "_" + cov["iso_d"].astype(str)
    cov["_k"] = np.arange(len(cov))
    m = deals[["iso_o", "iso_d", "year"]].merge(cov[["iso_o", "iso_d", "year", "_k"]], on=["iso_o", "iso_d", "year"], how="left")
    if m["_k"].isna().any():
        raise ValueError(f"{m['_k'].isna().sum()} deals without a covariate row")
    return m["_k"].to_numpy(dtype=np.int64), cov.drop(columns="_k")


def fit_stacked(d, regs, fes, pair_filter=False, dv="total_value"):
    sub = d.dropna(subset=regs + [dv])
    if pair_filter:
        sub = sub[sub.groupby(["dyad", "tech"])[dv].transform("size") >= 3]
    rhs = regs + [f"tech:{v}" for v in regs]
    fml = f"{dv} ~ " + " + ".join(rhs) + " | " + " + ".join(fes)
    return pf.fepois(fml, data=sub, vcov={"CRV1": "dyad"}, store_data=False, copy_data=False, lean=False)


def wald(m, names):
    c = m.coef()
    idx = [list(c.index).index(n) for n in names if n in c.index]
    b = c.to_numpy()[idx]
    V = np.asarray(m._vcov)[np.ix_(idx, idx)]
    s = float(b @ np.linalg.pinv(V) @ b)
    return s, float(stats.chi2.sf(s, len(idx))), len(idx)


def ladder_stats(d, ladder=None):
    """Interaction coefficients, nominal p and joint Wald for each FE structure."""
    out = {}
    for name, (fes, regs) in (ladder or LADDER).items():
        m = fit_stacked(d, regs, fes, pair_filter=name == "pair")
        inter = [f"tech:{v}" for v in regs]
        c, p = m.coef(), m.pvalue()
        s, wp, k = wald(m, inter)
        out[name] = dict(N=int(m._N), wald=s, wald_p=wp, df=k,
                         coef={v: float(c[f"tech:{v}"]) for v in regs if f"tech:{v}" in c.index},
                         p={v: float(p[f"tech:{v}"]) for v in regs if f"tech:{v}" in p.index})
    return out


def sector_grid_lag(p):
    """Mask lags whose (country, t-1) cell is absent from this sector's own panel.

    This reproduces how the thesis panels were built: lags ran on each sectoral panel
    separately, so an economy with no deals in that sector in t-1 has no lag. One common
    grid instead gives N=18,792 rather than 18,672 and changes no conclusion.
    """
    p = p.copy()
    for side, key in (("o", "iso_o"), ("d", "iso_d")):
        code = pd.factorize(p[key])[0] * 10000 + p["year"].to_numpy()
        ok = np.isin(code - 1, code)
        for v in p.columns:
            if v.endswith(f"_{side}_l1"):
                p.loc[~ok, v] = np.nan
    return p


def stacked_sector_lag(key, value, label, cov):
    d = panels_from_labels(key, value, label, cov)
    return pd.concat([sector_grid_lag(d[d.tech == t]) for t in (1, 0)], ignore_index=True)


def country_year_gdp_pop():
    g = pd.read_csv(PUBLIC / "gravity_cepii_2000_2024.csv.gz", usecols=["year", "isoo", "gdp_o", "pop_o"], **_READ)
    return g.groupby(["isoo", "year"], as_index=False)[["gdp_o", "pop_o"]].first()


def add_scaled_patents(cov):
    """Patents(t-1) per USD bn of GDP(t-1) and per million inhabitants(t-1)."""
    cy = country_year_gdp_pop()
    cy["year"] = cy["year"] + 1
    c = cov.copy()
    for s, k in (("o", "iso_o"), ("d", "iso_d")):
        x = cy.rename(columns={"isoo": k, "gdp_o": f"_g{s}", "pop_o": f"_p{s}"})
        c = c.merge(x, on=[k, "year"], how="left")
        c[f"pat_gdp_{s}_l1"] = c[f"patents_{s}_l1"] / (c[f"_g{s}"] / 1e6)
        c[f"pat_pop_{s}_l1"] = c[f"patents_{s}_l1"] / (c[f"_p{s}"] / 1e3)
        c = c.drop(columns=[f"_g{s}", f"_p{s}"])
    return c


def fit_share(d, regs, fes):
    """Poisson with offset log(total deals), written as a weighted share model."""
    fml = "sh ~ " + " + ".join(regs) + " | " + fes
    return pf.fepois(fml, data=d, weights="n", vcov={"CRV1": "dyad"}, store_data=False, copy_data=False)


SHARE_LADDER = {"year": ("year", REGS), "country": ("year+iso_o+iso_d", REGS),
                "pair": ("year+dyad", [v for v in REGS if v != "ln_dist"])}
