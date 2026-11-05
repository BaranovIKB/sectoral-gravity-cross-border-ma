"""
================================================================================
revision_specs.py  --  Revision specifications for the SDUFE thesis
Companion module to Main.py.  Place in the same folder and run:

    python revision_specs.py

Requires the panels built by Main.py. The quickest wiring is to import the
panel CSVs Main.py already writes, or to paste the SPEC_* calls at the end of
Main.py's __main__ block (see WIRING at the bottom of this file).

Every spec writes both a full statsmodels summary (.txt) and a tidy row set
into output/revision/coefficients.csv so the results can be tabulated directly.
================================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

TECH_VARS_L1 = [
    "rd_intensity_gov_o_l1", "rd_intensity_gov_d_l1",
    "rd_intensity_gross_o_l1", "rd_intensity_gross_d_l1",
    "patents_o_l1", "patents_d_l1",
]
GRAV_VARS = ["ln_gdp_o", "ln_gdp_d", "ln_dist"]
LAG_BASE = [
    "rd_intensity_gov_o", "rd_intensity_gov_d",
    "rd_intensity_gross_o", "rd_intensity_gross_d",
    "patents_o", "patents_d",
]

_ROWS = []


# ==============================================================================
# FIX 0 -- CORRECT LAG CONSTRUCTION      (replaces create_lagged_variables)
# ==============================================================================
# BUG in Main.py, create_lagged_variables():
#   df.sort_values(['iso_o','year']); df.groupby('iso_o')[var].shift(1)
# The panel is dyad-year, so each origin-year appears once per destination.
# shift(1) moves back one ROW, not one YEAR. With ~50-150 destinations per
# origin-year, ~99% of the "lag" is the CONTEMPORANEOUS value of the same year.
# Consequence: the one-year-lag identification strategy does not exist, and the
# IV first stage regresses a variable on (almost) itself -- which is why the
# first-stage coefficients in Table 13 are 0.9999 with se 0.0005.
# ==============================================================================

def make_lags(panel, lag_vars=None, lags=(1, 2)):
    """Correct country-year lags: shift on the unique country-year grid, then merge back."""
    if lag_vars is None:
        lag_vars = LAG_BASE
    df = panel.copy()

    for side, key in (("_o", "iso_o"), ("_d", "iso_d")):
        vars_side = [v for v in lag_vars if v.endswith(side) and v in df.columns]
        if not vars_side:
            continue
        # one row per country-year
        cy = (df[[key, "year"] + vars_side]
              .drop_duplicates(subset=[key, "year"])
              .sort_values([key, "year"])
              .reset_index(drop=True))
        # reindex onto a dense year grid so gaps do not silently become lags
        full = []
        for c, g in cy.groupby(key):
            g = g.set_index("year").reindex(range(int(cy["year"].min()),
                                                  int(cy["year"].max()) + 1))
            g[key] = c
            g.index.name = "year"
            full.append(g.reset_index())
        cy = pd.concat(full, ignore_index=True)

        for L in lags:
            for v in vars_side:
                cy[f"{v}_l{L}"] = cy.groupby(key)[v].shift(L)

        keep = [key, "year"] + [f"{v}_l{L}" for L in lags for v in vars_side]
        df = df.drop(columns=[c for c in keep if c in df.columns and c not in (key, "year")],
                     errors="ignore")
        df = df.merge(cy[keep], on=[key, "year"], how="left")

    return df


def audit_lags(panel, var="rd_intensity_gross_o"):
    """Prove the fix: share of rows where the lag equals the contemporaneous value."""
    out = {}
    for suffix in ("_l1",):
        col = var + suffix
        if col not in panel.columns:
            continue
        m = panel[[var, col]].dropna()
        out[col] = float((m[var] == m[col]).mean())
    return out


# ==============================================================================
# ESTIMATION HELPERS
# ==============================================================================

def _dyad(df):
    return df["iso_o"].astype(str) + "_" + df["iso_d"].astype(str)


def _formula_vars(formula, data):
    """Exact identifier extraction. Substring matching is wrong here: 'patents_o'
    is a substring of 'patents_o_l1', which would drop rows on the wrong column."""
    toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
    return [c for c in data.columns if c in toks and c != "_dyad"]


def ppml(formula, data, cluster=True, maxiter=200):
    """PPML with dyad-clustered SE (falls back to HC0)."""
    d = data.copy()
    d["_dyad"] = _dyad(d)
    used = _formula_vars(formula, d)
    d = d.dropna(subset=used)
    model = smf.glm(formula=formula, data=d, family=sm.families.Poisson())
    if cluster:
        res = model.fit(cov_type="cluster",
                        cov_kwds={"groups": d.loc[model.data.row_labels, "_dyad"]},
                        maxiter=maxiter)
    else:
        res = model.fit(cov_type="HC0", maxiter=maxiter)
    return res, d


def record(spec_id, panel_label, res, fe="none", note="", keep_prefixes=None):
    """Append tidy coefficient rows; skip fixed-effect dummies by default."""
    if res is None:
        _ROWS.append(dict(spec_id=spec_id, panel=panel_label, variable="FAILED",
                          coef=np.nan, se=np.nan, z=np.nan, p=np.nan,
                          n_obs=np.nan, fe=fe, note=note))
        return
    for name in res.params.index:
        if name.startswith("C(") and keep_prefixes is None:
            continue
        if keep_prefixes and not any(name.startswith(p) for p in keep_prefixes):
            if not name.startswith("Intercept"):
                continue
        _ROWS.append(dict(
            spec_id=spec_id, panel=panel_label, variable=name,
            coef=float(res.params[name]), se=float(res.bse[name]),
            z=float(res.tvalues[name]), p=float(res.pvalues[name]),
            n_obs=int(res.nobs), fe=fe, note=note))


def dump(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "coefficients.csv")
    pd.DataFrame(_ROWS).to_csv(path, index=False)
    print(f"[revision] wrote {path} ({len(_ROWS)} rows)")
    return path


def _write(output_dir, name, res):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
        f.write(res.summary().as_text() if res is not None else "FAILED")


# ==============================================================================
# SPEC 1  -- YEAR FE, THEN COUNTRY FE           (закрывает замечание 2.1)
# Priority 1. Section 1.4 claims FE that Tables 4-2/4-3 report as absent.
# Run the ladder and report all three columns side by side.
#   1a  year FE only                        <- minimum the reviewer demands
#   1b  year FE + origin FE + destination FE <- Anderson-van Wincoop proxy
#   1c  year FE + dyad FE                    <- ln_dist drops (time-invariant)
# ==============================================================================

def _record_pf(spec_id, panel_label, model, fe, note=""):
    t = model.tidy()
    for name, row in t.iterrows():
        _ROWS.append(dict(
            spec_id=spec_id, panel=panel_label, variable=str(name),
            coef=float(row["Estimate"]), se=float(row["Std. Error"]),
            z=float(row["t value"]), p=float(row["Pr(>|t|)"]),
            n_obs=int(model._N), fe=fe, note=note))


def ppml_fe(rhs, fe_list, data, label, spec_id, output_dir, fname,
            depvar="total_value", note=""):
    """PPML with absorbed fixed effects.

    Preferred path is pyfixest.fepois (absorbs FE, no dense dummy matrix).
    Fallback is statsmodels with C() dummies, but only when the implied dummy
    count is small enough not to blow memory -- a dyad-FE model on this panel
    would need a 14k x 5k dense design matrix otherwise.
    """
    d = data.copy()
    d["dyad"] = _dyad(d)
    need = [v for v in rhs if v in d.columns] + [depvar] + \
           [f for f in fe_list if f in d.columns]
    d = d.dropna(subset=list(dict.fromkeys(need)))
    if len(d) < 100:
        print(f"[{spec_id}] {label}: too few observations ({len(d)})")
        record(spec_id, label, None, fe="+".join(fe_list), note="n<100")
        return None

    try:
        import pyfixest as pf
        fml = f"{depvar} ~ " + " + ".join(rhs) + " | " + " + ".join(fe_list)
        m = pf.fepois(fml, data=d, vcov={"CRV1": "dyad"})
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, fname), "w", encoding="utf-8") as f:
            f.write(f"{fml}\nN = {m._N}\nSE: clustered on dyad\n\n")
            f.write(m.tidy().to_string())
        _record_pf(spec_id, label, m, fe="+".join(fe_list), note=note or "pyfixest")
        print(f"[{spec_id}] {label}: n={m._N} (pyfixest, FE absorbed)")
        return m
    except ImportError:
        pass
    except Exception as e:
        print(f"[{spec_id}] {label} pyfixest failed ({str(e)[:120]}), trying dense dummies")

    n_dummies = sum(d[f].nunique() for f in fe_list if f in d.columns)
    if n_dummies > 1200:
        msg = (f"{n_dummies} dummies -- skipped to avoid memory blow-up. "
               f"Install pyfixest (pip install pyfixest) and rerun.")
        print(f"[{spec_id}] {label}: {msg}")
        record(spec_id, label, None, fe="+".join(fe_list), note=msg)
        return None
    try:
        f_str = f"{depvar} ~ " + " + ".join(rhs) + " + " + \
                " + ".join(f"C({f})" for f in fe_list)
        res, _ = ppml(f_str, d)
        _write(output_dir, fname, res)
        record(spec_id, label, res, fe="+".join(fe_list),
               note=note or "statsmodels dense dummies")
        print(f"[{spec_id}] {label}: n={res.nobs:.0f} (dense dummies)")
        return res
    except Exception as e:
        print(f"[{spec_id}] {label} FAILED: {str(e)[:200]}")
        record(spec_id, label, None, fe="+".join(fe_list), note=str(e)[:200])
        return None


def spec1_year_fe(panel, label, output_dir):
    """FE ladder. All three columns go into the thesis side by side."""
    rhs = GRAV_VARS + TECH_VARS_L1
    rhs_nodist = [v for v in rhs if v != "ln_dist"]

    # 1a -- year FE only: the minimum the reviewer demands
    ppml_fe(rhs, ["year"], panel, label,
            "1a_yearFE", output_dir, f"spec1a_yearfe_{label}.txt")

    # 1b -- year + origin + destination FE: multilateral-resistance proxy
    ppml_fe(rhs, ["year", "iso_o", "iso_d"], panel, label,
            "1b_countryYearFE", output_dir, f"spec1b_countryyearfe_{label}.txt")

    # 1c -- year + dyad FE: ln_dist is absorbed (time-invariant within dyad).
    # This is the real answer to "correlated unobservables at the country-pair
    # level" that Section 5.4 raises -- stronger than any lag length.
    d = panel.copy()
    d["dyad"] = _dyad(d)
    d = d[d.groupby("dyad")["total_value"].transform("size") >= 3]
    ppml_fe(rhs_nodist, ["year", "dyad"], d, label,
            "1c_dyadYearFE", output_dir, f"spec1c_dyadfe_{label}.txt",
            note="ln_dist absorbed by dyad FE; dyads with >=3 obs")


# ==============================================================================
# SPEC 2  -- STRUCTURAL BREAK / TIME SPLIT      (закрывает замечание 1.1)
# Decides whether the title survives. Two forms; report both.
#   2a  split-sample estimation 2000-2010 vs 2011-2024
#   2b  pooled + post2010 interactions + joint Wald test on all interactions
# Robustness: rerun 2b with break at 2008 and 2012.
# ==============================================================================

def spec2_time_split(panel, label, output_dir, break_year=2010):
    early = panel[panel["year"] <= break_year]
    late = panel[panel["year"] > break_year]
    base = " + ".join(GRAV_VARS + TECH_VARS_L1)

    for tag, sub in (("early", early), ("late", late)):
        try:
            res, _ = ppml(f"total_value ~ {base} + C(year)", sub)
            _write(output_dir, f"spec2a_{tag}{break_year}_{label}.txt", res)
            record(f"2a_{tag}_{break_year}", label, res, fe="year")
            print(f"[2a-{tag}] {label}: n={res.nobs:.0f}")
        except Exception as e:
            print(f"[2a-{tag}] {label} FAILED: {e}")

    d = panel.copy()
    d["post"] = (d["year"] > break_year).astype(int)
    inter_vars = ["ln_gdp_d", "ln_gdp_o", "ln_dist",
                  "patents_o_l1", "patents_d_l1",
                  "rd_intensity_gov_d_l1", "rd_intensity_gross_d_l1"]
    inter_terms = [f"post:{v}" for v in inter_vars]
    f = f"total_value ~ {base} + post + " + " + ".join(inter_terms) + " + C(year)"
    try:
        res, _ = ppml(f, d)
        _write(output_dir, f"spec2b_post{break_year}_{label}.txt", res)
        record(f"2b_post{break_year}", label, res, fe="year")
        present = [t for t in inter_terms if t in res.params.index]
        w = res.wald_test(" = 0, ".join(present) + " = 0", scalar=True)
        print(f"[2b] {label} joint Wald on interactions: "
              f"stat={float(w.statistic):.3f}, p={float(w.pvalue):.5f}")
        _ROWS.append(dict(spec_id=f"2b_post{break_year}_WALD", panel=label,
                          variable="joint_interactions", coef=np.nan, se=np.nan,
                          z=float(w.statistic), p=float(w.pvalue),
                          n_obs=int(res.nobs), fe="year",
                          note="H0: no structural break"))
    except Exception as e:
        print(f"[2b] {label} FAILED: {e}")


# ==============================================================================
# SPEC 3  -- FORMAL TECH vs NON-TECH DIFFERENCE  (закрывает замечание 3.3)
# The thesis interprets rd_gov_d as "insignificant in tech, -2.208 in non-tech"
# without ever testing whether the two coefficients differ. Stack the panels,
# interact everything with a tech dummy, cluster on dyad (each dyad-year can
# appear in both panels), and test each interaction.
# ==============================================================================

def spec3_sector_interaction(panel_tech, panel_nontech, output_dir):
    a = panel_tech.copy(); a["tech"] = 1
    b = panel_nontech.copy(); b["tech"] = 0
    cols = ["iso_o", "iso_d", "year", "total_value", "tech"] + GRAV_VARS + TECH_VARS_L1
    cols = [c for c in cols if c in a.columns and c in b.columns]
    d = pd.concat([a[cols], b[cols]], ignore_index=True)

    regs = GRAV_VARS + TECH_VARS_L1
    base = " + ".join(regs)
    inter = " + ".join([f"tech:{v}" for v in regs])
    f = f"total_value ~ {base} + tech + {inter} + C(year)"
    try:
        res, _ = ppml(f, d)
        _write(output_dir, "spec3_tech_interaction.txt", res)
        record("3_techInteraction", "stacked", res, fe="year",
               note="tech:X = tech minus non-tech difference")
        print("\n[3] Coefficient differences (tech - non-tech):")
        for v in regs:
            t = f"tech:{v}"
            if t in res.params.index:
                print(f"    {v:28s} diff={res.params[t]: .4g}  p={res.pvalues[t]:.4f}")
    except Exception as e:
        print(f"[3] FAILED: {e}")


# ==============================================================================
# SPEC 4  -- log(1+patents) UNDER PPML
# Main.py runs this check as OLS only (run_robustness_checks, log-patents block),
# and Table 20 contradicts the baseline: log_patents_o = 0.0020 (p=0.903),
# log_patents_d = +0.0390 (p=0.040). The thesis text in 4.4 states the opposite.
# Rerun like-for-like under PPML before writing anything about this check.
# ==============================================================================

def spec4_log_patents(panel, label, output_dir):
    d = panel.copy()
    for s in ("o", "d"):
        src = f"patents_{s}_l1"
        if src in d.columns:
            d[f"log_patents_{s}_l1"] = np.log1p(d[src])
    rd = [v for v in TECH_VARS_L1 if v.startswith("rd_")]
    base = " + ".join(GRAV_VARS + rd + ["log_patents_o_l1", "log_patents_d_l1"])
    for tag, extra in (("noFE", ""), ("yearFE", " + C(year)")):
        try:
            res, _ = ppml(f"total_value ~ {base}{extra}", d)
            _write(output_dir, f"spec4_logpatents_{tag}_{label}.txt", res)
            record(f"4_logPatents_{tag}", label, res,
                   fe=("year" if extra else "none"))
            print(f"[4-{tag}] {label}: n={res.nobs:.0f}")
        except Exception as e:
            print(f"[4-{tag}] {label} FAILED: {e}")


# ==============================================================================
# SPEC 5  -- GDP PER CAPITA / INCOME CONTROL     (закрывает замечание 3.2)
# Reviewer: the negative origin-patent coefficient may just proxy for income.
# Use ln_gdp + ln_pop rather than ln_gdp + ln_gdppc (algebraically the same
# reparametrisation, but avoids the mechanical collinearity of gdp and gdppc).
# REQUIRES: add "pop_o","pop_d" to the column list in load_gravity_data().
# ==============================================================================

def spec5_income_control(panel, label, output_dir):
    d = panel.copy()
    have = [c for c in ("pop_o", "pop_d") if c in d.columns]
    if len(have) < 2:
        print("[5] SKIPPED: add pop_o/pop_d to load_gravity_data() column list")
        record("5_incomeControl", label, None, note="pop_o/pop_d missing")
        return
    d["ln_pop_o"] = np.log(d["pop_o"])
    d["ln_pop_d"] = np.log(d["pop_d"])
    base = " + ".join(GRAV_VARS + ["ln_pop_o", "ln_pop_d"] + TECH_VARS_L1)
    try:
        res, _ = ppml(f"total_value ~ {base} + C(year)", d)
        _write(output_dir, f"spec5_income_{label}.txt", res)
        record("5_incomeControl", label, res, fe="year")
        print(f"[5] {label}: n={res.nobs:.0f}  patents_o_l1="
              f"{res.params.get('patents_o_l1', np.nan):.4g} "
              f"(p={res.pvalues.get('patents_o_l1', np.nan):.4f})")
    except Exception as e:
        print(f"[5] {label} FAILED: {e}")


# ==============================================================================
# SPEC 6  -- MISSING DEAL VALUES                 (закрывает замечание 2.3)
#
# Main.py aggregates with total_value=("deal_value","sum"), which skips NaN,
# then fillna(0). A dyad-year where every deal has an undisclosed value becomes
# a hard zero, indistinguishable from "no deals happened".
#
# BUG FIXED IN ROUND 2. The first version passed the FULL deal table together
# with a SECTORAL panel, so disclosure was computed across all deals in the
# dyad-year and then merged onto the technology (or non-technology) panel. A
# technology cell counted as "disclosed" whenever some unrelated non-technology
# deal in the same dyad-year reported a value, which understated censoring in
# both sectoral panels. Disclosure has to be measured inside the same subset of
# deals that generated the dependent variable.
#
#   6a  share of dyad-years with deals but zero disclosed value, within sector
#   6b  re-estimate keeping only dyad-years with at least one disclosed value
# ==============================================================================

def _sector_deals(deals, sector):
    """Restrict the deal table to the subset that generated the sectoral panel."""
    if sector is None:
        return deals
    if "is_tech_target" not in deals.columns:
        raise KeyError("spec6_disclosure(sector=...) needs 'is_tech_target' in deals")
    return deals[deals["is_tech_target"] == (1 if sector == "tech" else 0)]


def spec6_disclosure(deals, panel, label, output_dir, sector="infer"):
    """Censoring audit. `sector` must match the panel: 'tech', 'nontech' or None.

    sector='infer' reads it off `label`, which is what the wiring relies on.
    Passing sector=None against a sectoral panel reproduces the round-1 bug and
    is allowed only so the old numbers can be regenerated on purpose.
    """
    if sector == "infer":
        sector = {"tech": "tech", "nontech": "nontech"}.get(label, None)
    dv = _sector_deals(deals, sector).copy()
    print(f"[6] {label}: disclosure measured on {len(dv)} deals "
          f"(sector={sector!r}; full deal table = {len(deals)})")
    dv["_has_value"] = dv["deal_value"].notna().astype(int)
    disc = dv.groupby(["iso_o", "iso_d", "year"], as_index=False).agg(
        n_deals=("_has_value", "size"), n_disclosed=("_has_value", "sum"))
    # same window as every other spec, so this and spec6_from_panel -- two
    # independent routes to the same statistic -- are comparable
    d = prep(panel, label, verbose=False).merge(
        disc, on=["iso_o", "iso_d", "year"], how="left")
    d["n_deals"] = d["n_deals"].fillna(0)
    d["n_disclosed"] = d["n_disclosed"].fillna(0)

    with_deals = d[d["n_deals"] > 0]
    share = float((with_deals["n_disclosed"] == 0).mean()) if len(with_deals) else np.nan
    deal_share = (float(with_deals["n_disclosed"].sum() / with_deals["n_deals"].sum())
                  if len(with_deals) else np.nan)
    print(f"[6a] {label}: dyad-years with deals = {len(with_deals)}; "
          f"share with ZERO disclosed value = {share:.3%}; "
          f"deal-level disclosure rate = {deal_share:.3%}")
    _ROWS.append(dict(spec_id="6a_disclosure", panel=label,
                      variable="share_dyadyears_no_disclosed_value",
                      coef=share, se=np.nan, z=np.nan, p=np.nan,
                      n_obs=len(with_deals), fe="none",
                      note=f"sector={sector}; "
                           f"deal_level_disclosure_rate={deal_share:.4f}"))

    sub = d[(d["n_deals"] == 0) | (d["n_disclosed"] > 0)]
    base = " + ".join(GRAV_VARS + TECH_VARS_L1)
    try:
        res, _ = ppml(f"total_value ~ {base} + C(year)", sub)
        _write(output_dir, f"spec6b_disclosedonly_{label}.txt", res)
        record("6b_disclosedOnly", label, res, fe="year",
               note=f"sector={sector}; dyad-years with deals but no "
                    f"disclosed value dropped")
        print(f"[6b] {label}: n={res.nobs:.0f}")
    except Exception as e:
        print(f"[6b] {label} FAILED: {e}")


# ==============================================================================
# SPEC 7  -- TWO-YEAR LAG SENSITIVITY            (закрывает замечание 2.2)
# Meaningful only after FIX 0. Run alongside spec1a for a like-for-like column.
# ==============================================================================

def spec7_two_year_lag(panel, label, output_dir):
    l2 = [v.replace("_l1", "_l2") for v in TECH_VARS_L1]
    missing = [v for v in l2 if v not in panel.columns]
    if missing:
        print(f"[7] SKIPPED: run make_lags(..., lags=(1,2)) first; missing {missing}")
        return
    base = " + ".join(GRAV_VARS + l2)
    try:
        res, _ = ppml(f"total_value ~ {base} + C(year)", panel)
        _write(output_dir, f"spec7_lag2_{label}.txt", res)
        record("7_lag2", label, res, fe="year")
        print(f"[7] {label}: n={res.nobs:.0f}")
    except Exception as e:
        print(f"[7] {label} FAILED: {e}")


# ==============================================================================
# SPEC 10 -- DOES THE STRUCTURAL BREAK ITSELF DIFFER BY SECTOR?
#
# Motivation. Section 5.3 rejects parameter stability for non-technology deals
# (p=0.019) and fails to reject it for technology deals (p=0.174). Reading that
# pair as "the break differs by sector" is the difference-in-significance error:
# the technology panel is 40% smaller, so unequal power alone can produce it.
#
# The claim "non-technology gravity structure moved while technology gravity
# structure held still" is a statement about a DIFFERENCE IN DIFFERENCES and has
# to be tested as one. Stack both panels, and fit
#
#   Y ~ X + tech + tech:X + post:X + tech:post:X + C(year)
#
# The tech:post:X terms are the sector difference in the size of the break.
# A joint Wald test on that set is the actual test. `post` on its own is absorbed
# by the year effects and is omitted.
# ==============================================================================

def spec10_break_by_sector(panel_tech, panel_nontech, output_dir, break_year=2010):
    a = panel_tech.copy(); a["tech"] = 1
    b = panel_nontech.copy(); b["tech"] = 0
    cols = ["iso_o", "iso_d", "year", "total_value", "tech"] + GRAV_VARS + TECH_VARS_L1
    cols = [c for c in cols if c in a.columns and c in b.columns]
    d = pd.concat([a[cols], b[cols]], ignore_index=True)
    d["post"] = (d["year"] > break_year).astype(int)

    regs = GRAV_VARS + TECH_VARS_L1
    base = " + ".join(regs)
    t2 = " + ".join(f"tech:{v}" for v in regs)
    p2 = " + ".join(f"post:{v}" for v in regs)
    t3 = [f"tech:post:{v}" for v in regs]
    f = (f"total_value ~ {base} + tech + {t2} + {p2} + " + " + ".join(t3) + " + C(year)")
    try:
        res, _ = ppml(f, d)
        _write(output_dir, f"spec10_break_by_sector_{break_year}.txt", res)
        record(f"10_breakBySector_{break_year}", "stacked", res, fe="year",
               note="tech:post:X = sector difference in the size of the break")
        present = [t for t in t3 if t in res.params.index]
        w = res.wald_test(" = 0, ".join(present) + " = 0", scalar=True)
        print(f"\n[10] Does the break differ by sector? break={break_year}")
        print(f"     joint Wald on tech:post:X  stat={float(w.statistic):.3f}  "
              f"p={float(w.pvalue):.5f}   (H0: break is the same in both panels)")
        for t in present:
            print(f"       {t:38s}{res.params[t]: .4g}  p={res.pvalues[t]:.4f}")
        _ROWS.append(dict(spec_id=f"10_breakBySector_{break_year}_WALD", panel="stacked",
                          variable="joint_triple_interactions", coef=np.nan, se=np.nan,
                          z=float(w.statistic), p=float(w.pvalue), n_obs=int(res.nobs),
                          fe="year", note="H0: break identical across sectors"))
    except Exception as e:
        print(f"[10] FAILED: {str(e)[:200]}")
        record(f"10_breakBySector_{break_year}", "stacked", None, note=str(e)[:200])


# ==============================================================================
# SPEC 8  -- DESCRIPTIVE STATISTICS TABLE
# The thesis has no descriptive statistics table at all. Table 4-1 is used twice
# (panel summary in 4.1, variable summary in 4.2.3). This produces the missing one.
# ==============================================================================

def spec8_descriptives(panel, label, output_dir):
    cols = ["total_value", "total_deals"] + GRAV_VARS + TECH_VARS_L1
    cols = [c for c in cols if c in panel.columns]
    t = panel[cols].describe(percentiles=[.25, .5, .75]).T
    t["n_missing"] = panel[cols].isna().sum()
    os.makedirs(output_dir, exist_ok=True)
    p = os.path.join(output_dir, f"spec8_descriptives_{label}.csv")
    t.to_csv(p)
    print(f"[8] {label}: wrote {p}")
    return t


# ==============================================================================
# SPEC 9  -- TRUE ESTIMATION-SAMPLE COVERAGE
# The thesis advertises 192 acquirer / 212 target countries, but PPML runs on
# 14,362 of 47,019 observations because OECD MSTI covers ~40 economies. Report
# what the ESTIMATION sample actually contains -- a blind reviewer will check.
# ==============================================================================

def spec9_coverage(panel, label, output_dir):
    need = GRAV_VARS + TECH_VARS_L1 + ["total_value"]
    need = [c for c in need if c in panel.columns]
    est = panel.dropna(subset=need)
    info = {
        "panel_obs": len(panel),
        "estimation_obs": len(est),
        "retained_share": round(len(est) / max(len(panel), 1), 4),
        "n_origin_panel": panel["iso_o"].nunique(),
        "n_origin_estimation": est["iso_o"].nunique(),
        "n_dest_panel": panel["iso_d"].nunique(),
        "n_dest_estimation": est["iso_d"].nunique(),
        "n_dyads_estimation": len(est[["iso_o", "iso_d"]].drop_duplicates()),
        "years_estimation": f"{est['year'].min():.0f}-{est['year'].max():.0f}" if len(est) else "",
    }
    os.makedirs(output_dir, exist_ok=True)
    pd.Series(info).to_csv(os.path.join(output_dir, f"spec9_coverage_{label}.csv"))
    pd.Series(sorted(est["iso_o"].unique())).to_csv(
        os.path.join(output_dir, f"spec9_origin_countries_{label}.csv"), index=False)
    pd.Series(sorted(est["iso_d"].unique())).to_csv(
        os.path.join(output_dir, f"spec9_dest_countries_{label}.csv"), index=False)
    print(f"[9] {label}: {info}")
    return info


# ##############################################################################
#
#  ROUND 2  --  specifications added after the independent referee report
#
#  What the referee found, and which spec answers it:
#
#   R1  The stacked tech / non-tech difference test (spec 3) was never run with
#       fixed effects. Under country effects the destination-GDP point estimates
#       reverse order (0.774 tech vs 0.611 non-technology), which would invert
#       H2. Until spec 11 is run, the sign of the main surviving result is not
#       established.                                             -> spec11
#
#   R2  The composition model (tech_deals in Main.run_mechanism_tests) is
#       pooled Poisson with HC0 errors, contemporaneous regressors and no
#       fixed effects. Under country effects all three reported coefficients
#       lose significance. The thesis calls this block the most robust one.
#                                                                -> spec12
#
#   R3  H3 (destination patent stock) is carried by China. Dropping China from
#       both sides moves the stacked difference from p=0.0001 to p=0.27. The
#       thesis reports no leave-one-out evidence at all.          -> spec13
#
#   R4  The same difference is null when patents enter in logs (p=0.43), while
#       section 5.2 reads a significant-here / insignificant-there pair as a
#       difference -- the exact inference section 5.3 rejects.    -> spec11(log)
#
#   R5  The PPML justification in the methodology says the estimator "retains
#       zero-flow dyad-years". There are no zero-flow dyad-years: min(total_deals)
#       is 1. The zeros in the dependent variable are undisclosed values, which
#       is a censoring problem, not an excess-zeros problem.      -> spec15
#
#   R6  Estimation ran on the unfiltered panel (1991-2026) while the thesis
#       describes a 2000-2024 window, and domestic pairs were only removed
#       late.                                                     -> prep()
#
# Every round-2 spec writes into the same _ROWS store, so coefficients.csv
# carries round 1 and round 2 side by side and the robustness table can be
# assembled from one file.
# ##############################################################################

from scipy import stats as _st

YEAR_MIN, YEAR_MAX = 2000, 2024
STACK_KEY = ["iso_o", "iso_d", "year", "total_value", "total_deals"]


def prep(df, label="", ymin=YEAR_MIN, ymax=YEAR_MAX, drop_domestic=True, verbose=True):
    """Sample window used by every round-2 spec. Loud on purpose.

    The thesis describes 2000-2024 cross-border deals. Round-1 estimation ran on
    whatever the panel contained. Both filters are applied here so that the code
    and the text describe the same sample.
    """
    d = df.copy()
    n0 = len(d)
    if drop_domestic and {"iso_o", "iso_d"}.issubset(d.columns):
        d = d[d["iso_o"] != d["iso_d"]]
    n1 = len(d)
    if "year" in d.columns:
        d = d[(d["year"] >= ymin) & (d["year"] <= ymax)]
    n2 = len(d)
    if verbose:
        print(f"[prep] {label or 'panel'}: {n0} -> {n1} (domestic pairs dropped: "
              f"{n0 - n1}) -> {n2} (outside {ymin}-{ymax}: {n1 - n2})")
    return d.reset_index(drop=True)


def dump_named(output_dir, fname="coefficients.csv"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, fname)
    pd.DataFrame(_ROWS).to_csv(path, index=False)
    print(f"[revision] wrote {path} ({len(_ROWS)} rows)")
    return path


def _wald(coef, vcov, index, names):
    """Joint Wald test on a named subset. Works for pyfixest and statsmodels."""
    index = list(index)
    keep = [n for n in names if n in index]
    if not keep:
        return np.nan, np.nan, 0
    idx = [index.index(n) for n in keep]
    b = np.asarray(coef, dtype=float)[idx]
    V = np.asarray(vcov, dtype=float)[np.ix_(idx, idx)]
    stat = float(b @ np.linalg.pinv(V) @ b)
    dfree = len(keep)
    return stat, float(_st.chi2.sf(stat, dfree)), dfree


def _pf_parts(m):
    c = m.coef()
    return c, m._vcov, list(c.index)


def _record_pf_rows(spec_id, panel_label, m, fe, note=""):
    t = m.tidy()
    for name, row in t.iterrows():
        _ROWS.append(dict(
            spec_id=spec_id, panel=panel_label, variable=str(name),
            coef=float(row["Estimate"]), se=float(row["Std. Error"]),
            z=float(row["t value"]), p=float(row["Pr(>|t|)"]),
            n_obs=int(m._N), fe=fe, note=note))


# ==============================================================================
# SPEC 11 -- THE STACKED DIFFERENCE TEST, WITH FIXED EFFECTS    (referee R1, R4)
#
# spec3 tested the tech / non-tech difference with year effects only. That is
# the weakest column in the FE ladder the thesis itself builds for the level
# equations, so the difference test was held to a lower standard than the
# results it is used to interpret.
#
# The fixed effects here are SECTOR-SPECIFIC (iso_o^tech, not iso_o). A common
# country effect across the two stacked panels would force technology and
# non-technology deals from the same origin to share one intercept, which is
# not what the sectoral regressions do. Interacting each effect with the sector
# dummy makes the stacked regression numerically equivalent to estimating both
# panels separately, while the tech:X coefficients still deliver the difference
# with one dyad-clustered covariance matrix. That is the whole point of stacking.
#
#   11a  year^tech FE                       (reproduces spec3, like-for-like)
#   11b  year^tech + iso_o^tech + iso_d^tech
#   11c  year^tech + dyad^tech              (ln_dist and tech:ln_dist absorbed)
#
# log_patents=True reruns the identical ladder with log(1+patents) in place of
# patent counts. If H3 survives in levels but not in logs, the thesis has to say
# so; a functional form chosen after seeing which one is significant is exactly
# what a blind reviewer looks for.
# ==============================================================================

def _stack(panel_tech, panel_nontech, log_patents=False, exclude=None):
    a = prep(panel_tech, "tech", verbose=False).copy()
    a["tech"] = 1
    b = prep(panel_nontech, "nontech", verbose=False).copy()
    b["tech"] = 0
    cols = STACK_KEY + ["tech"] + GRAV_VARS + TECH_VARS_L1
    cols = [c for c in cols if c in a.columns and c in b.columns]
    d = pd.concat([a[cols], b[cols]], ignore_index=True)
    if exclude:
        excl = [exclude] if isinstance(exclude, str) else list(exclude)
        d = d[~d["iso_o"].isin(excl) & ~d["iso_d"].isin(excl)]
    d["dyad"] = _dyad(d)
    regs = list(GRAV_VARS + TECH_VARS_L1)
    if log_patents:
        for s in ("o", "d"):
            src, dst = f"patents_{s}_l1", f"log_patents_{s}_l1"
            if src in d.columns:
                d[dst] = np.log1p(d[src])
                regs[regs.index(src)] = dst
    return d.reset_index(drop=True), regs


def spec11_stacked_fe(panel_tech, panel_nontech, output_dir,
                      log_patents=False, exclude=None, tag=None):
    """Difference test across the FE ladder. Returns a tidy DataFrame."""
    try:
        import pyfixest as pf
    except ImportError:
        print("[11] SKIPPED: pip install pyfixest")
        record("11_stackedFE", "stacked", None, note="pyfixest missing")
        return None

    if tag is None:
        tag = "log" if log_patents else "lvl"
        if exclude:
            tag += "_no" + (exclude if isinstance(exclude, str) else "".join(exclude))

    d, regs = _stack(panel_tech, panel_nontech, log_patents, exclude)
    ladder = [
        ("11a_yearFE",    ["year^tech"], regs),
        ("11b_countryFE", ["year^tech", "iso_o^tech", "iso_d^tech"], regs),
        ("11c_dyadFE",    ["year^tech", "dyad^tech"],
         [v for v in regs if v != "ln_dist"]),
    ]
    out = []
    for spec_id, fes, rr in ladder:
        rhs = rr + [f"tech:{v}" for v in rr]
        sub = d.dropna(subset=rr + ["total_value"]).copy()
        if spec_id.endswith("dyadFE"):
            # a dyad observed once contributes nothing once its own effect is
            # removed; dropping singletons keeps the reported N honest
            sub = sub[sub.groupby(["dyad", "tech"])["total_value"]
                      .transform("size") >= 3]
        fml = "total_value ~ " + " + ".join(rhs) + " | " + " + ".join(fes)
        try:
            m = pf.fepois(fml, data=sub, vcov={"CRV1": "dyad"})
        except Exception as e:
            print(f"[{spec_id}/{tag}] FAILED: {str(e)[:160]}")
            record(spec_id, f"stacked_{tag}", None, fe="+".join(fes), note=str(e)[:160])
            continue

        _record_pf_rows(spec_id, f"stacked_{tag}", m, "+".join(fes),
                        note=f"tech:X = tech minus non-tech; {tag}")
        coef, V, idx = _pf_parts(m)
        inter = [f"tech:{v}" for v in rr]
        stat, p, k = _wald(coef, V, idx, inter)
        _ROWS.append(dict(spec_id=spec_id + "_WALD", panel=f"stacked_{tag}",
                          variable="joint_tech_interactions", coef=np.nan, se=np.nan,
                          z=stat, p=p, n_obs=int(m._N), fe="+".join(fes),
                          note=f"H0: no sectoral difference in any slope (df={k})"))
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"spec{spec_id}_{tag}.txt"), "w",
                  encoding="utf-8") as f:
            f.write(f"{fml}\nN = {m._N}\nSE: clustered on dyad\n\n")
            f.write(m.tidy().to_string())
            f.write(f"\n\nJoint Wald on tech:X  chi2({k}) = {stat:.4f}  p = {p:.5f}\n")

        print(f"\n[{spec_id}] {tag}: N={m._N}   joint Wald chi2({k})={stat:.3f} p={p:.5f}")
        t = m.tidy()
        for v in rr:
            n = f"tech:{v}"
            if n in t.index:
                out.append(dict(spec=spec_id, tag=tag, var=v,
                                diff=float(t.loc[n, "Estimate"]),
                                se=float(t.loc[n, "Std. Error"]),
                                p=float(t.loc[n, "Pr(>|t|)"]), n=int(m._N)))
                print(f"    tech-minus-nontech  {v:26s}{t.loc[n, 'Estimate']: .4g}"
                      f"   p={t.loc[n, 'Pr(>|t|)']:.4f}")
        # sectoral levels too, so the thesis reports both slopes and not just the gap
        for v in rr:
            if v in t.index and f"tech:{v}" in t.index:
                lo = float(t.loc[v, "Estimate"])
                print(f"        non-tech {v:22s}{lo: .4g}    tech "
                      f"{lo + float(t.loc[f'tech:{v}', 'Estimate']): .4g}")
    res = pd.DataFrame(out)
    if len(res):
        res.to_csv(os.path.join(output_dir, f"spec11_differences_{tag}.csv"), index=False)
    return res


# ==============================================================================
# SPEC 12 -- COMPOSITION MODEL, DONE PROPERLY                     (referee R2)
#
# Main.run_mechanism_tests fits  tech_deals ~ gravity + technology  as pooled
# Poisson with HC0 errors and contemporaneous regressors. Three problems:
#   - no offset, so the coefficients describe the COUNT of technology deals,
#     not the technology SHARE, while the text reads them as a share;
#   - no fixed effects, so any cross-country correlation between R&D intensity
#     and deal composition is attributed to R&D;
#   - HC0 rather than dyad clustering, in a panel where each dyad recurs.
#
# Here: offset log(total_deals) makes it a share model, regressors are lagged to
# match the level equations, errors are clustered on the dyad, and the FE ladder
# is the same one used for the level equations. The free-regressor variant lets
# log(total_deals) take its own coefficient, which is the standard check on
# whether the offset restriction (coefficient exactly 1) is doing the work.
# ==============================================================================

def _fepois_takes_offset():
    try:
        import inspect
        import pyfixest as pf
        return "offset" in inspect.signature(pf.fepois).parameters
    except Exception:
        return False


def _offset_poisson_sm(d, rhs, fes, off_col, label, spec_id, output_dir,
                       depvar="tech_deals", max_dummies=400):
    """Share model with a fixed offset, via statsmodels, for older pyfixest."""
    fes = list(fes or [])
    n_dummies = sum(d[f].nunique() for f in fes if f in d.columns)
    if n_dummies > max_dummies:
        msg = (f"{n_dummies} dummies -- offset form skipped on this pyfixest "
               f"version; upgrade with 'pip install -U pyfixest' to get it")
        print(f"[{spec_id}] {label}: {msg}")
        record(spec_id, label, None, fe="+".join(fes or ["none"]), note=msg)
        return None
    sub = d.dropna(subset=list(rhs) + [depvar, off_col]).copy()
    fml = f"{depvar} ~ " + " + ".join(rhs)
    if fes:
        fml += " + " + " + ".join(f"C({f})" for f in fes)
    try:
        model = smf.glm(formula=fml, data=sub, family=sm.families.Poisson(),
                        offset=sub[off_col].to_numpy())
        res = model.fit(cov_type="cluster",
                        cov_kwds={"groups": sub.loc[model.data.row_labels, "dyad"]},
                        maxiter=200)
    except Exception as e:
        print(f"[{spec_id}] {label} FAILED: {str(e)[:160]}")
        record(spec_id, label, None, fe="+".join(fes or ["none"]), note=str(e)[:160])
        return None
    _write(output_dir, f"spec{spec_id}_{label}.txt", res)
    record(spec_id, label, res, fe="+".join(fes or ["none"]),
           note="share model, fixed offset log(total_deals); statsmodels fallback")
    keys = [v for v in ("rd_intensity_gross_d_l1", "rd_intensity_gov_d_l1",
                        "patents_d_l1") if v in res.params.index]
    print(f"[{spec_id}] N={res.nobs:.0f}  " + "  ".join(
        f"{k.replace('_intensity', '').replace('_l1', ''):16s}="
        f"{res.params[k]: .4g}(p={res.pvalues[k]:.4f})" for k in keys))
    return res


def spec12_composition(panel, output_dir, exclude=None, tag="", lagged=True):
    try:
        import pyfixest as pf
    except ImportError:
        print("[12] SKIPPED: pip install pyfixest")
        record("12_composition", "full", None, note="pyfixest missing")
        return None
    if "tech_deals" not in panel.columns:
        print("[12] SKIPPED: 'tech_deals' only exists on the full panel")
        record("12_composition", "full", None, note="tech_deals missing")
        return None

    d = prep(panel, "full (composition)")
    if exclude:
        excl = [exclude] if isinstance(exclude, str) else list(exclude)
        d = d[~d["iso_o"].isin(excl) & ~d["iso_d"].isin(excl)]
        tag = tag or ("no" + "".join(excl))
    d = d[d["total_deals"] > 0].copy()
    d["ln_total_deals"] = np.log(d["total_deals"])
    d["dyad"] = _dyad(d)

    tech_vars = TECH_VARS_L1 if lagged else LAG_BASE
    rhs = GRAV_VARS + list(tech_vars)
    d = d.dropna(subset=rhs + ["tech_deals", "ln_total_deals"])
    print(f"[12] estimation sample: {len(d)} dyad-years, "
          f"{d['tech_deals'].sum():.0f} technology deals of {d['total_deals'].sum():.0f}")

    ladder = [
        ("12_none",    None,                       rhs),
        ("12_year",    ["year"],                   rhs),
        ("12_country", ["year", "iso_o", "iso_d"], rhs),
        ("12_dyad",    ["year", "dyad"], [v for v in rhs if v != "ln_dist"]),
    ]
    label = f"full{('_' + tag) if tag else ''}"
    for spec_id, fes, rr in ladder:
        for form, extra, off in (("offset", [], "ln_total_deals"),
                                 ("free", ["ln_total_deals"], None)):
            fml = "tech_deals ~ " + " + ".join(rr + extra)
            if fes:
                fml += " | " + " + ".join(fes)
            if off and not _fepois_takes_offset():
                # pyfixest gained the offset argument only in recent versions.
                # Fall back to statsmodels, which has always had it, as long as
                # the fixed effects fit in a dense dummy matrix (country FE is
                # ~113 dummies; dyad FE is thousands and is skipped -- the free
                # variant below covers that rung).
                _offset_poisson_sm(d, rr, fes, off, label, f"{spec_id}_offset",
                                   output_dir)
                continue
            try:
                m = pf.fepois(fml, data=d, vcov={"CRV1": "dyad"},
                              **({"offset": off} if off else {}))
            except Exception as e:
                print(f"[{spec_id}/{form}] FAILED: {str(e)[:160]}")
                record(f"{spec_id}_{form}", label, None,
                       fe="+".join(fes or ["none"]), note=str(e)[:160])
                continue
            _record_pf_rows(f"{spec_id}_{form}", label, m,
                            "+".join(fes or ["none"]),
                            note=f"share model, {form} log(total_deals)")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f"spec{spec_id}_{form}_{label}.txt"),
                      "w", encoding="utf-8") as f:
                f.write(f"{fml}\noffset={off}\nN = {m._N}\nSE: clustered on dyad\n\n")
                f.write(m.tidy().to_string())
            t = m.tidy()
            keys = [v for v in ("rd_intensity_gross_d_l1", "rd_intensity_gov_d_l1",
                                "patents_d_l1") if v in t.index]
            print(f"[{spec_id}/{form}] N={m._N}  " + "  ".join(
                f"{k.replace('_intensity', '').replace('_l1', ''):16s}="
                f"{t.loc[k, 'Estimate']: .4g}(p={t.loc[k, 'Pr(>|t|)']:.4f})"
                for k in keys))
    return True


# ==============================================================================
# SPEC 13 -- LEAVE ONE COUNTRY OUT                                (referee R3)
#
# "The destination patent effect is driven by China" is a claim that has to be
# either refuted or reported. Refuting it needs the whole leave-one-out
# distribution, not a single China re-run: if the coefficient is fragile to
# China alone it may be fragile to the United States too, and a reviewer who
# asks the question will not stop at one country.
#
# Each iteration drops one economy from BOTH sides of every dyad and refits the
# year-FE stacked difference model. The output is one row per excluded country
# for the coefficients the hypotheses rest on, which goes straight into an
# appendix table.
# ==============================================================================

def spec13_leave_one_out(panel_tech, panel_nontech, output_dir, countries=None,
                         top=15, focus=("ln_gdp_d", "patents_d_l1", "ln_dist",
                                        "ln_gdp_o")):
    try:
        import pyfixest as pf
    except ImportError:
        print("[13] SKIPPED: pip install pyfixest")
        return None

    d, regs = _stack(panel_tech, panel_nontech)
    d = d.dropna(subset=regs + ["total_value"])
    if countries is None:
        vol = (pd.concat([d.groupby("iso_o")["total_value"].sum(),
                          d.groupby("iso_d")["total_value"].sum()])
               .groupby(level=0).sum().sort_values(ascending=False))
        countries = list(dict.fromkeys(list(vol.head(top).index) + ["CHN", "USA"]))
    rhs = regs + [f"tech:{v}" for v in regs]
    fml = "total_value ~ " + " + ".join(rhs) + " | year^tech"

    rows = []
    base = pf.fepois(fml, data=d, vcov={"CRV1": "dyad"}).tidy()
    for v in focus:
        n = f"tech:{v}"
        if n in base.index:
            rows.append(dict(excluded="(none)", var=v,
                             diff=float(base.loc[n, "Estimate"]),
                             p=float(base.loc[n, "Pr(>|t|)"]), n=int(len(d))))
    for c in countries:
        sub = d[(d["iso_o"] != c) & (d["iso_d"] != c)]
        if len(sub) < 500:
            continue
        try:
            t = pf.fepois(fml, data=sub, vcov={"CRV1": "dyad"}).tidy()
        except Exception as e:
            print(f"[13] {c}: FAILED {str(e)[:100]}")
            continue
        for v in focus:
            n = f"tech:{v}"
            if n in t.index:
                rows.append(dict(excluded=c, var=v, diff=float(t.loc[n, "Estimate"]),
                                 p=float(t.loc[n, "Pr(>|t|)"]), n=int(len(sub))))
        shown = {v: (float(t.loc[f"tech:{v}", "Estimate"]),
                     float(t.loc[f"tech:{v}", "Pr(>|t|)"]))
                 for v in focus if f"tech:{v}" in t.index}
        print(f"[13] drop {c}: " + "  ".join(
            f"{v}={b: .4g}(p={p:.3f})" for v, (b, p) in shown.items()))

    out = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    out.to_csv(os.path.join(output_dir, "spec13_leave_one_out.csv"), index=False)
    for _, r in out.iterrows():
        _ROWS.append(dict(spec_id="13_LOO", panel=f"drop_{r['excluded']}",
                          variable=f"tech:{r['var']}", coef=r["diff"], se=np.nan,
                          z=np.nan, p=r["p"], n_obs=r["n"], fe="year^tech",
                          note="leave-one-country-out on the stacked difference"))
    for v in focus:
        s = out[(out["var"] == v) & (out["excluded"] != "(none)")]
        if len(s):
            print(f"[13] {v}: significant at 5% in {int((s['p'] < 0.05).sum())} of "
                  f"{len(s)} leave-one-out samples")
    return out


# ==============================================================================
# SPEC 15 -- WHAT THE ZEROS ACTUALLY ARE                          (referee R5)
#
# The methodology chapter justifies PPML by saying it "retains zero-flow
# dyad-years that a log-linear specification would discard". The panel is built
# by grouping the deal table, so a dyad-year exists only if at least one deal
# exists: min(total_deals) is 1 and there is not a single zero-flow cell. The
# zeros in total_value come from fillna(0) on undisclosed transaction values.
#
# Those are opposite problems. Excess zeros in a count of flows is what PPML is
# advertised for; systematically missing values on observed flows is censoring,
# and PPML does not fix it -- spec 6b is the check that does.
#
# This spec produces the numbers needed to rewrite that paragraph truthfully and
# to state the extensive-margin selection explicitly: how many of the potential
# country-pair-years the estimation sample actually uses.
# ==============================================================================

def spec6_from_panel(panel, label, output_dir):
    """spec6 without the deal table.

    total_value is the sum of DISCLOSED deal values inside the panel's own sector,
    and every cell in the panel has at least one deal (spec15). So a cell with
    total_value == 0 is exactly a cell where no deal disclosed a value -- the 6a
    statistic, measured inside the right sector by construction, with no merge to
    get wrong. That removes the dependency on the deal table, which only exists
    inside Main.py, and makes the censoring numbers reproducible from the saved
    panels alone.

    What this cannot give is the deal-LEVEL disclosure rate; for that the deal
    table is still needed. The cell-level rate is what the thesis reports.
    """
    d = prep(panel, f"{label} (censoring)", verbose=False)
    need = [c for c in GRAV_VARS + TECH_VARS_L1 + ["total_value"] if c in d.columns]
    est = d.dropna(subset=need)
    share_panel = float((d["total_value"] == 0).mean())
    share_est = float((est["total_value"] == 0).mean())
    print(f"[6a] {label}: censored cells = {int((d['total_value'] == 0).sum())}/{len(d)} "
          f"({share_panel:.2%}) in the panel, "
          f"{int((est['total_value'] == 0).sum())}/{len(est)} ({share_est:.2%}) "
          f"in the estimation sample")
    _ROWS.append(dict(spec_id="6a_disclosure_panel", panel=label,
                      variable="share_cells_no_disclosed_value",
                      coef=share_est, se=np.nan, z=np.nan, p=np.nan,
                      n_obs=len(est), fe="none",
                      note=f"cell-level censoring, within sector; "
                           f"whole panel={share_panel:.4f}"))

    try:
        import pyfixest as pf
    except ImportError:
        return share_est
    sub = est[est["total_value"] > 0]
    sub = sub.copy()
    sub["dyad"] = _dyad(sub)
    rhs = GRAV_VARS + TECH_VARS_L1
    for spec_id, fes in (("6b_disclosedOnly_year", ["year"]),
                         ("6b_disclosedOnly_country", ["year", "iso_o", "iso_d"])):
        fml = "total_value ~ " + " + ".join(rhs) + " | " + " + ".join(fes)
        try:
            m = pf.fepois(fml, data=sub, vcov={"CRV1": "dyad"})
        except Exception as e:
            print(f"[{spec_id}] {label} FAILED: {str(e)[:140]}")
            continue
        _record_pf_rows(spec_id, label, m, "+".join(fes),
                        note="cells with deals but no disclosed value dropped")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"spec{spec_id}_{label}.txt"), "w",
                  encoding="utf-8") as f:
            f.write(f"{fml}\nN = {m._N}\nSE: clustered on dyad\n\n")
            f.write(m.tidy().to_string())
        print(f"[{spec_id}] {label}: N={m._N}")
    return share_est


def spec15_extensive_margin(panel, panel_tech, panel_nontech, output_dir):
    rows = {}
    for lbl, p in (("full", panel), ("tech", panel_tech), ("nontech", panel_nontech)):
        d = prep(p, lbl, verbose=False)
        need = [c for c in GRAV_VARS + TECH_VARS_L1 + ["total_value"] if c in d.columns]
        est = d.dropna(subset=need)
        n_o, n_d = est["iso_o"].nunique(), est["iso_d"].nunique()
        yrs = est["year"].nunique()
        # potential cells among the economies the estimation sample can identify,
        # excluding own-country pairs
        both = len(set(est["iso_o"]).intersection(set(est["iso_d"])))
        pot = (n_o * n_d - both) * yrs
        td = est["total_deals"] if "total_deals" in est.columns else pd.Series(dtype=float)
        rows[lbl] = {
            "panel_cells": len(d),
            "estimation_cells": len(est),
            "origins_estimation": n_o,
            "destinations_estimation": n_d,
            "years_estimation": yrs,
            "potential_cells": pot,
            "share_of_potential_observed": round(len(est) / pot, 4) if pot else np.nan,
            "min_total_deals": float(td.min()) if len(td) else np.nan,
            "cells_with_zero_deals": int((td == 0).sum()) if len(td) else 0,
            "cells_with_zero_value": int((est["total_value"] == 0).sum()),
            "share_cells_zero_value": round(float((est["total_value"] == 0).mean()), 4),
        }
        print(f"[15] {lbl}: {rows[lbl]}")
    out = pd.DataFrame(rows).T
    os.makedirs(output_dir, exist_ok=True)
    out.to_csv(os.path.join(output_dir, "spec15_extensive_margin.csv"))
    for lbl, r in rows.items():
        _ROWS.append(dict(spec_id="15_extensiveMargin", panel=lbl,
                          variable="share_of_potential_cells_observed",
                          coef=r["share_of_potential_observed"], se=np.nan, z=np.nan,
                          p=np.nan, n_obs=r["estimation_cells"], fe="none",
                          note=(f"zero-deal cells={r['cells_with_zero_deals']}; "
                                f"zero-value cells={r['cells_with_zero_value']} "
                                f"({r['share_cells_zero_value']:.1%}) = undisclosed, "
                                f"not absent flows")))
    return out


# ==============================================================================
# SPEC 16 -- ROBUSTNESS TABLE
# One table, one row per claim, one column per stress test. This is the table a
# blind reviewer looks for and the thesis currently does not have.
# ==============================================================================

def spec16_robustness_table(output_dir, fname="spec16_robustness.csv"):
    df = pd.DataFrame(_ROWS)
    if df.empty:
        return None
    v = df["variable"].astype(str)
    keep = df[v.str.startswith("tech:") | (v == "joint_tech_interactions")].copy()
    if keep.empty:
        return None
    piv = keep.pivot_table(index="variable", columns=["spec_id", "panel"],
                           values=["coef", "p"], aggfunc="first")
    os.makedirs(output_dir, exist_ok=True)
    piv.to_csv(os.path.join(output_dir, fname))
    print(f"[16] wrote {os.path.join(output_dir, fname)}")
    return piv


# ==============================================================================
# WIRING -- append to the end of Main.py's __main__ block
# ==============================================================================
WIRING = r'''
# --- REVISION SPECS -----------------------------------------------------------
import revision_specs as rv

out_rev = os.path.join(output_dir, "revision")
os.makedirs(out_rev, exist_ok=True)

# FIX 0: correct lags (replaces create_lagged_variables everywhere)
panel         = rv.make_lags(panel,         lags=(1, 2))
panel_tech    = rv.make_lags(panel_tech,    lags=(1, 2))
panel_nontech = rv.make_lags(panel_nontech, lags=(1, 2))
print("LAG AUDIT (share of rows where lag == contemporaneous; must be ~0):",
      rv.audit_lags(panel))

# so round 2 can be rerun without rebuilding the panels from the Orbis extract
panel.to_csv(os.path.join(output_dir, "panel_full.csv"), index=False)
panel_tech.to_csv(os.path.join(output_dir, "panel_tech.csv"), index=False)
panel_nontech.to_csv(os.path.join(output_dir, "panel_nontech.csv"), index=False)
deals[["iso_o", "iso_d", "year", "deal_value", "is_tech_target"]].to_csv(
    os.path.join(output_dir, "deals_min.csv"), index=False)

# ---- ROUND 1 -----------------------------------------------------------------
for lbl, pnl in (("full", panel), ("tech", panel_tech), ("nontech", panel_nontech)):
    rv.spec9_coverage(pnl, lbl, out_rev)
    rv.spec8_descriptives(pnl, lbl, out_rev)
    rv.spec1_year_fe(pnl, lbl, out_rev)          # PRIORITY 1
    rv.spec4_log_patents(pnl, lbl, out_rev)      # PRIORITY 2
    rv.spec2_time_split(pnl, lbl, out_rev, 2010) # PRIORITY 3
    rv.spec7_two_year_lag(pnl, lbl, out_rev)
    rv.spec5_income_control(pnl, lbl, out_rev)

rv.spec2_time_split(panel_tech, "tech", out_rev, 2008)
rv.spec2_time_split(panel_tech, "tech", out_rev, 2012)
rv.spec3_sector_interaction(panel_tech, panel_nontech, out_rev)
rv.spec10_break_by_sector(panel_tech, panel_nontech, out_rev, 2010)

# spec6 infers the sector from the label and measures disclosure inside that
# sector's own deals. Passing the full deal table with a sectoral panel was the
# round-1 bug; the sectoral subset is now taken inside the function.
rv.spec6_disclosure(deals, panel_tech,    "tech",    out_rev)
rv.spec6_disclosure(deals, panel_nontech, "nontech", out_rev)

# ---- ROUND 2 (referee report) ------------------------------------------------
rv.run_round2(panel, panel_tech, panel_nontech, out_rev)

rv.dump(out_rev)
# --- END REVISION SPECS -------------------------------------------------------
'''


# ==============================================================================
# ROUND 2 DRIVER
# Runs as part of Main.py, or standalone from the saved panel CSVs:
#
#     python revision_specs.py --panels output --out output/revision
#
# Standalone mode exists so the referee checks can be rerun in about a minute
# without touching the Orbis extract. It rebuilds the lags itself, because the
# saved CSVs hold the base series, not the lagged ones.
# ==============================================================================

def run_round2(panel, panel_tech, panel_nontech, output_dir, drop_country="CHN"):
    print("\n" + "=" * 78)
    print("ROUND 2 -- referee report specifications")
    print("=" * 78)

    spec15_extensive_margin(panel, panel_tech, panel_nontech, output_dir)

    # corrected censoring audit, measured inside each sector's own panel
    for lbl, pnl in (("tech", panel_tech), ("nontech", panel_nontech)):
        spec6_from_panel(pnl, lbl, output_dir)

    # R1 / R4: the difference test under the FE ladder, levels and logs
    spec11_stacked_fe(panel_tech, panel_nontech, output_dir, log_patents=False)
    spec11_stacked_fe(panel_tech, panel_nontech, output_dir, log_patents=True)

    # R3: same ladder without the economy that may be carrying H3
    if drop_country:
        spec11_stacked_fe(panel_tech, panel_nontech, output_dir, exclude=drop_country)
        spec13_leave_one_out(panel_tech, panel_nontech, output_dir)

    # R2: composition model with an offset, lags, clustering and fixed effects
    spec12_composition(panel, output_dir)
    if drop_country:
        spec12_composition(panel, output_dir, exclude=drop_country)

    spec16_robustness_table(output_dir)
    print("\nROUND 2 complete.\n")


def _load_panels(panel_dir):
    """Read the CSVs Main.py writes and rebuild the lags."""
    def rd(name, alts=()):
        for n in (name,) + tuple(alts):
            p = os.path.join(panel_dir, n)
            if os.path.exists(p):
                print(f"[load] {p}")
                return pd.read_csv(p)
        raise FileNotFoundError(f"{name} not found in {panel_dir}")

    panel = rd("panel_full.csv", ("mna_gravity_panel.csv",))
    ptech = rd("panel_tech.csv")
    pnon = rd("panel_nontech.csv")
    out = []
    for d in (panel, ptech, pnon):
        if "patents_o_l1" not in d.columns:
            d = make_lags(d, lags=(1, 2))
        out.append(d)
    print("LAG AUDIT (must be ~0):", audit_lags(out[0]))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Revision specs for the SDUFE thesis")
    ap.add_argument("--panels", help="folder holding panel_*.csv written by Main.py")
    ap.add_argument("--out", default=None,
                    help="output folder (default <panels>/revision)")
    ap.add_argument("--drop-country", default="CHN",
                    help="economy removed in the sensitivity run ('' to skip)")
    a = ap.parse_args()

    if not a.panels:
        print(__doc__)
        print("Append this to the end of Main.py's __main__ block:")
        print(WIRING)
        raise SystemExit(0)

    out_dir = a.out or os.path.join(a.panels, "revision")
    pnl, ptech, pnon = _load_panels(a.panels)
    run_round2(pnl, ptech, pnon, out_dir, drop_country=(a.drop_country or None))

    # deals_min.csv gives the deal-LEVEL disclosure rate, which the panel alone
    # cannot: it is the only statistic in spec 6 that needs the deal table.
    dmin = os.path.join(a.panels, "deals_min.csv")
    if os.path.exists(dmin):
        print(f"[load] {dmin}")
        deals = pd.read_csv(dmin)
        spec6_disclosure(deals, ptech, "tech", out_dir)
        spec6_disclosure(deals, pnon, "nontech", out_dir)
    else:
        print("[6] deals_min.csv not found -- cell-level censoring only")

    dump_named(out_dir, "coefficients_round2.csv")
