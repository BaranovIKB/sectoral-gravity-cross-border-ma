"""New specifications answering the post-review remarks. Output: results/new_specs.csv.

patent_quality  B4     triadic families and PCT filings in place of raw applications
rd_measure      H1     funding-based and performance-based R&D, replacing GOVERD
institutions    B5     WGI, RTA, diplomatic distance, entry costs, FDI screening
ddd, event      A4     triple differences around FIRRMA, GDPR and EU screening
iv              A6     control function with distance-weighted peer instruments
hetero          A5     control vs minority, income groups of the target, sub-periods
wide            A1,C1  sample without the OECD R&D constraint
window          A1     46 economies extended to 2024 with World Bank GDP
"""
import sys, warnings
import numpy as np, pandas as pd
from thesis_lib import *
warnings.filterwarnings("ignore")

ROWS = []
_prev = RESULTS / "new_specs.csv"
if len(sys.argv) > 1 and _prev.exists():  # rerunning some parts keeps the others
    _p = pd.read_csv(_prev)
    _drop = {"ddd": ["ddd", "event"]}
    _skip = [t for a in sys.argv[1:] for t in _drop.get(a, [a])]
    ROWS = _p[~_p.table.isin(_skip)].to_dict("records")
D = PUBLIC


def rec(table, spec, fe, m, keep=None, **extra):
    c, s, p = m.coef(), m.se(), m.pvalue()
    for v in c.index:
        if keep is None or any(k in v for k in keep):
            ROWS.append(dict(table=table, spec=spec, fe=fe, N=int(m._N), var=v, coef=float(c[v]),
                             se=float(s[v]), p=float(p[v]), **extra))


def save():
    pd.DataFrame(ROWS).to_csv(RESULTS / "new_specs.csv", index=False)


deals = load_deals()
cov0 = add_scaled_patents(load_covariates())
cy = pd.read_csv(D / "extra_country_year.csv")
dy = pd.read_csv(D / "extra_dyad_year.csv.gz")
EST = sorted(set(cov0.dropna(subset=REGS).iso_o))
print("estimation economies:", len(EST), flush=True)

# lagged country-year extras on both sides
LAGGED = ["triad", "pct", "rd_fund", "herd", "public_perf", "wgi_pv", "wgi_rq", "wgi_rl"]
x = cy[["iso3", "year"] + LAGGED].copy(); x["year"] += 1
cov = cov0.copy()
for s, k in (("o", "iso_o"), ("d", "iso_d")):
    cov = cov.merge(x.rename(columns={"iso3": k, **{v: f"{v}_{s}_l1" for v in LAGGED}}), on=[k, "year"], how="left")
rri = cy[["iso3", "rri_screen", "rri_total"]].drop_duplicates("iso3").rename(columns={"iso3": "iso_d"})
cov = cov.merge(rri, on="iso_d", how="left")
cov = cov.merge(dy[["iso_o", "iso_d", "year", "fta_wto", "diplo_disagreement", "entry_cost_d", "col45"]], on=["iso_o", "iso_d", "year"], how="left")
key, cov2 = prepare_keys(deals, cov)
val = deals.deal_value.to_numpy()
tech = deals.is_tech_target.to_numpy().astype(bool)
base = stacked_sector_lag(key, val, tech, cov2)
base = base.assign(log_triad_o_l1=np.log1p(base.triad_o_l1), log_triad_d_l1=np.log1p(base.triad_d_l1),
                   log_pct_o_l1=np.log1p(base.pct_o_l1), log_pct_d_l1=np.log1p(base.pct_d_l1))
print("base", base.shape, flush=True)


def swap(regs, old, new):
    return [v.replace(old, new) if v.startswith(old) else v for v in regs]


def part_patent_quality():
    for nm, new in (("Triadic families", "triad"), ("Triadic families, log(1+x)", "log_triad"),
                    ("PCT filings", "pct"), ("PCT filings, log(1+x)", "log_pct")):
        for ex_lbl, d in (("full", base), ("noCHN", base[(base.iso_o != "CHN") & (base.iso_d != "CHN")])):
            for fe, (ff, regs) in LADDER.items():
                rr = swap(regs, "patents", new)
                m = fit_stacked(d, rr, ff, pair_filter=fe == "pair")
                inter = [f"tech:{v}" for v in rr]
                s, wp, k = wald(m, inter)
                rec("patent_quality", f"{nm}|{ex_lbl}", fe, m, keep=[f"{new}_d_l1", f"{new}_o_l1", "tech:ln_gdp_d"], wald=s, wald_p=wp)
    save()


def part_rd_measure():
    for nm, new in (("GOVERD (baseline variable)", None), ("Government-financed GERD (G_FG)", "rd_fund"),
                    ("GOVERD + HERD", "public_perf")):
        for fe, (ff, regs) in LADDER.items():
            rr = list(regs)
            if new:
                rr = [v.replace("rd_intensity_gov", new) if v.startswith("rd_intensity_gov") else v for v in rr]
            m = fit_stacked(base, rr, ff, pair_filter=fe == "pair")
            rec("rd_measure", nm, fe, m, keep=[(new or "rd_intensity_gov") + "_d_l1", "rd_intensity_gross_d_l1"])
    save()


def part_institutions():
    A = ["wgi_pv_d_l1", "wgi_rq_d_l1", "wgi_pv_o_l1", "wgi_rq_o_l1", "fta_wto", "diplo_disagreement"]
    sets = {"WGI, RTA, diplomatic distance": A, "plus entry costs (2003-2019)": A + ["entry_cost_d"]}
    for nm, ex in sets.items():
        for fe, (ff, regs) in LADDER.items():
            rr = regs + ex
            m = fit_stacked(base, rr, ff, pair_filter=fe == "pair")
            rec("institutions", nm, fe, m, keep=["ln_gdp_d", "patents_d_l1", "ln_dist", "wgi", "fta", "diplo", "entry"])
    # the 2018 FDI screening score does not vary over time: identified under year effects only
    for v in ("rri_screen", "rri_total"):
        m = fit_stacked(base, REGS + [v], ["year^tech"])
        rec("institutions", f"FDI index 2018: {v}", "year", m, keep=["ln_gdp_d", "patents_d_l1", v])
    save()


EU27 = ["AUT", "BEL", "BGR", "HRV", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU",
        "LUX", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"]


def ddd_panel(ymin=2005):
    d = stacked_sector_lag(key, val, tech, cov2)
    d = d[d.iso_o.isin(EST) & d.iso_d.isin(EST) & (d.year >= ymin) & (d.year <= 2024)].copy()
    n = deals.assign(t=tech.astype(int))
    c = n.groupby(["iso_o", "iso_d", "year", "t"]).size().rename("deals").reset_index().rename(columns={"t": "tech"})
    return d.merge(c, on=["iso_o", "iso_d", "year", "tech"], how="left")


DDD_FE = "dyad^tech + year^tech + iso_d^year + iso_o^year"


def fit_ddd(d, y, treat):
    return pf.fepois(f"{y} ~ {treat} | {DDD_FE}", data=d, vcov={"CRV1": "dyad"}, store_data=False, copy_data=False,
                     fixef_tol=1e-5)


def part_ddd():
    d = ddd_panel()
    d["usa"] = (d.iso_d == "USA").astype(int)
    d["chn_o"] = (d.iso_o == "CHN").astype(int)
    d["eu"] = d.iso_d.isin(EU27 + ["GBR"]).astype(int)
    d["eu27"] = d.iso_d.isin(EU27).astype(int)
    d["screen"] = d.rri_screen.fillna(0)
    shocks = {
        "FIRRMA: tech x USA x post2019": lambda x: x.tech * x.usa * (x.year >= 2019),
        "FIRRMA alt: tech x USA x post2018": lambda x: x.tech * x.usa * (x.year >= 2018),
        "GDPR: tech x EU+UK x post2018": lambda x: x.tech * x.eu * (x.year >= 2018),
        "EU screening: tech x EU27 x post2021": lambda x: x.tech * x.eu27 * (x.year >= 2021),
        "Screening score: tech x score2018 x post2019": lambda x: x.tech * x.screen * (x.year >= 2019),
    }
    for nm, f in shocks.items():
        d["T"] = f(d).astype(float)
        for y in ("total_value", "deals"):
            m = fit_ddd(d, y, "T")
            rec("ddd", f"{nm}|{y}", "dyad^tech+year^tech+dest^year+orig^year", m, keep=["T"])
    # direct exposure: Chinese acquirers into the United States
    d["T"] = (d.tech * d.usa * d.chn_o * (d.year >= 2019)).astype(float)
    d["A1"] = (d.tech * d.usa * (d.year >= 2019)).astype(float)
    d["A2"] = (d.tech * d.chn_o * (d.year >= 2019)).astype(float)
    for y in ("total_value", "deals"):
        m = fit_ddd(d, y, "T + A1 + A2")
        rec("ddd", f"FIRRMA China: tech x USA x CHN x post2019|{y}", "same", m)
    # event study for FIRRMA and GDPR, base year 2017
    for nm, grp in (("FIRRMA", "usa"), ("GDPR", "eu")):
        names = []
        for yr in range(2010, 2025):
            if yr == 2017:
                continue
            n = f"ev{yr}"
            d[n] = (d.tech * d[grp] * (d.year == yr)).astype(float)
            names.append(n)
        for y in ("total_value", "deals"):
            m = fit_ddd(d, y, " + ".join(names))
            pre = [n for n in names if int(n[2:]) < 2017]
            c = m.coef(); V = np.asarray(m._vcov); idx = [list(c.index).index(n) for n in pre]
            b = c.to_numpy()[idx]; s = float(b @ np.linalg.pinv(V[np.ix_(idx, idx)]) @ b)
            rec("event", f"{nm}|{y}", "same", m, pretrend_wald=s, pretrend_p=float(stats.chi2.sf(s, len(idx))))
    # placebo: assign the treatment to each other destination in turn
    rows = []
    for y in ("total_value", "deals"):
        for c in EST:
            d["T"] = (d.tech * (d.iso_d == c) * (d.year >= 2019)).astype(float)
            try:
                m = fit_ddd(d, y, "T")
                rows.append(dict(y=y, dest=c, coef=float(m.coef()["T"]), p=float(m.pvalue()["T"])))
            except Exception as e:
                rows.append(dict(y=y, dest=c, coef=np.nan, p=np.nan))
    pd.DataFrame(rows).to_csv(RESULTS / "ddd_placebo_destinations.csv", index=False)
    # placebo for EU groups: random groups of equal size
    rng = np.random.default_rng(20261105)
    rows = []
    for y in ("total_value", "deals"):
        for r in range(20 if FAST else 200):
            g = rng.choice(EST, size=len(set(EU27 + ["GBR"]) & set(EST)), replace=False)
            d["T"] = (d.tech * d.iso_d.isin(g) * (d.year >= 2018)).astype(float)
            try:
                rows.append(dict(y=y, rep=r, coef=float(fit_ddd(d, y, "T").coef()["T"])))
            except Exception:
                pass
    pd.DataFrame(rows).to_csv(RESULTS / "ddd_placebo_groups.csv", index=False)
    save()


def part_iv():
    """Control function: peer instruments, first stage with country and year effects.

    z_it is the inverse-distance weighted mean of x_jt across the other sample economies.
    The exclusion restriction is an assumption: peers' research may reach bilateral deals
    through channels other than the destination's own research.
    """
    g = pd.read_csv(PUBLIC / "gravity_cepii_2000_2024.csv.gz",
                    usecols=["isoo", "isod", "dist"]).drop_duplicates(["isoo", "isod"])
    g = g[g.isoo.isin(EST) & g.isod.isin(EST) & (g.isoo != g.isod)]
    w = g.assign(w=1 / g.dist)
    base46 = base.dropna(subset=REGS)
    grid = base46[["iso_d", "year", "rd_intensity_gross_d_l1", "rd_intensity_gov_d_l1", "patents_d_l1", "ln_gdp_d"]].drop_duplicates(["iso_d", "year"])
    X = ["rd_intensity_gross_d_l1", "rd_intensity_gov_d_l1", "patents_d_l1"]
    long = grid.merge(w.rename(columns={"isoo": "iso_d", "isod": "peer"})[["iso_d", "peer", "w"]], on="iso_d")
    peer = grid[["iso_d", "year"] + X].rename(columns={"iso_d": "peer", **{x: f"p_{x}" for x in X}})
    long = long.merge(peer, on=["peer", "year"], how="left")
    z = {}
    for x in X:
        v = long.dropna(subset=[f"p_{x}"])
        z[x] = (v.assign(a=v[f"p_{x}"] * v.w).groupby(["iso_d", "year"]).a.sum() / v.groupby(["iso_d", "year"]).w.sum()).rename(f"z_{x}")
    fs = grid.copy()
    for x in X:
        fs = fs.merge(z[x].reset_index(), on=["iso_d", "year"], how="left")
    Z = [f"z_{x}" for x in X]
    fs = fs.dropna(subset=X + Z + ["ln_gdp_d"])

    def first(data):
        res = data[["iso_d", "year"]].copy(); info = {}
        for x in X:
            m = pf.feols(f"{x} ~ {' + '.join(Z)} + ln_gdp_d | iso_d + year", data=data, vcov={"CRV1": "iso_d"})
            c = m.coef(); V = np.asarray(m._vcov); idx = [list(c.index).index(a) for a in Z]
            b = c.to_numpy()[idx]; F = float(b @ np.linalg.pinv(V[np.ix_(idx, idx)]) @ b) / len(Z)
            info[x] = F
            res["v_" + x] = data[x].to_numpy() - np.asarray(m.predict())
        return res, info

    res, info = first(fs)
    pd.DataFrame([{"var": k, "first_stage_F": v} for k, v in info.items()]).to_csv(RESULTS / "iv_first_stage.csv", index=False)
    print("first stage F", info, flush=True)
    V = ["v_" + x for x in X]

    def second(data):
        d = data.merge(res, on=["iso_d", "year"], how="inner")
        regs = REGS + V
        return fit_stacked(d, regs, ["year^tech", "iso_o^tech", "iso_d^tech"]), d

    m, d = second(base46)
    rec("iv", "Control function, country FE", "country", m,
        keep=["ln_gdp_d", "patents_d_l1", "rd_intensity_gross_d_l1", "rd_intensity_gov_d_l1", "v_"])
    # destination block bootstrap of both stages
    rng = np.random.default_rng(20261105)
    names = list(m.coef().index); boot = []
    dests = np.array(sorted(fs.iso_d.unique()))
    for b in range(30 if FAST else 300):
        pick = rng.choice(dests, size=len(dests), replace=True)
        parts_fs, parts_d = [], []
        for j, c in enumerate(pick):
            parts_fs.append(fs[fs.iso_d == c].assign(iso_d=f"{c}_{j}"))
            parts_d.append(base46[base46.iso_d == c].assign(iso_d=f"{c}_{j}"))
        try:
            fb = pd.concat(parts_fs, ignore_index=True); db = pd.concat(parts_d, ignore_index=True)
            db["dyad"] = db.iso_o + "_" + db.iso_d
            rb, _ = first(fb)
            dd = db.merge(rb, on=["iso_d", "year"], how="inner")
            mb = fit_stacked(dd, REGS + V, ["year^tech", "iso_o^tech", "iso_d^tech"])
            boot.append(mb.coef().reindex(names))
        except Exception:
            pass
    B = pd.DataFrame(boot)
    B.to_csv(RESULTS / "iv_bootstrap.csv", index=False)
    print("bootstrap draws", len(B), flush=True)
    save()


def part_hetero():
    fs = pd.to_numeric(deals.FINLSTAK, errors="coerce")
    dt = deals.DEALTYPE.fillna("")
    minority = ((fs < 50) | dt.str.startswith("Minority") | deals.FINLSTAK.fillna("").astype(str).str.contains("minority")).to_numpy()
    control = (((fs >= 50) | dt.str.startswith("Acquisition")) & ~pd.Series(minority)).to_numpy()
    print("minority", minority.mean(), "control", control.mean(), flush=True)
    keep = ["tech:ln_gdp_d", "tech:patents_d_l1", "tech:ln_dist", "tech:patents_o_l1"]
    for nm, mask in (("Control deals", control), ("Minority deals", minority)):
        d = stacked_sector_lag(key[mask], val[mask], tech[mask], cov2)
        for fe in ("year", "country"):
            ff, regs = LADDER[fe]
            rec("hetero", nm, fe, fit_stacked(d, regs, ff), keep=keep)
    gc = cy[(cy.year >= 2001) & (cy.year <= 2021) & cy.iso3.isin(EST)].groupby("iso3").wdi_gdpcap.mean()
    high = set(gc[gc >= gc.median()].index); low = set(EST) - high
    pd.DataFrame({"iso3": gc.index, "gdpcap_mean_2001_2021": gc.values, "group": ["high" if i in high else "lower" for i in gc.index]}).to_csv(RESULTS / "income_groups.csv", index=False)
    for nm, f in (("Target above median income", lambda d: d.iso_d.isin(high)),
                  ("Target below median income", lambda d: d.iso_d.isin(low)),
                  ("Acquirer above median income", lambda d: d.iso_o.isin(high)),
                  ("Acquirer below median income", lambda d: d.iso_o.isin(low)),
                  ("2001-2010", lambda d: d.year <= 2010), ("2011-2021", lambda d: d.year >= 2011)):
        d = base[f(base)]
        for fe in ("year", "country"):
            ff, regs = LADDER[fe]
            rec("hetero", nm, fe, fit_stacked(d, regs, ff), keep=keep)
    save()


def part_wide():
    regs = ["ln_gdp_o", "ln_gdp_d", "ln_dist", "patents_o_l1", "patents_d_l1"]
    lad = {"year": (["year^tech"], regs), "country": (["year^tech", "iso_o^tech", "iso_d^tech"], regs),
           "pair": (["year^tech", "dyad^tech"], [v for v in regs if v != "ln_dist"])}
    in46 = base.iso_o.isin(EST) & base.iso_d.isin(EST)
    log = base.assign(lp_o=np.log1p(base.patents_o_l1), lp_d=np.log1p(base.patents_d_l1))
    for nm, d, rr in (("46 economies, no R&D", base[in46], regs), ("All economies, no R&D", base, regs),
                      ("All economies, no R&D, excl. China", base[(base.iso_o != "CHN") & (base.iso_d != "CHN")], regs),
                      ("All economies, log patents", log, ["ln_gdp_o", "ln_gdp_d", "ln_dist", "lp_o", "lp_d"])):
        for fe, (ff, _) in lad.items():
            use = [v for v in rr if not (fe == "pair" and v == "ln_dist")]
            sub = d
            m = fit_stacked(sub, use, ff, pair_filter=fe == "pair")
            used = sub.dropna(subset=use)
            rec("wide", nm, fe, m, keep=["tech:", ], n_o=used.iso_o.nunique(), n_d=used.iso_d.nunique())
    # wider R&D: gross R&D from the World Bank (UNESCO UIS) with no government term
    x = cy[["iso3", "year", "wdi_rd"]].copy(); x["year"] += 1
    w = base.copy()
    for s, k in (("o", "iso_o"), ("d", "iso_d")):
        w = w.merge(x.rename(columns={"iso3": k, "wdi_rd": f"rdw_{s}"}), on=[k, "year"], how="left")
    rr = regs + ["rdw_o", "rdw_d"]
    for fe, (ff, _) in lad.items():
        use = [v for v in rr if not (fe == "pair" and v == "ln_dist")]
        m = fit_stacked(w, use, ff, pair_filter=fe == "pair")
        used = w.dropna(subset=use)
        rec("wide", "UNESCO gross R&D (via World Bank)", fe, m, keep=["rdw_d", "tech:"], n_o=used.iso_o.nunique(), n_d=used.iso_d.nunique())
    save()


def part_window():
    g = cy[["iso3", "year", "wdi_gdp"]].rename(columns={"wdi_gdp": "g"})
    c = cov2.copy()
    dist = c.groupby("dyad").ln_dist.transform("max")
    c["ln_dist"] = dist
    for s, k in (("o", "iso_o"), ("d", "iso_d")):
        c = c.merge(g.rename(columns={"iso3": k, "g": f"g_{s}"}), on=[k, "year"], how="left")
        c[f"ln_gdp_{s}"] = np.log(c[f"g_{s}"])
    d = stacked_sector_lag(key, val, tech, c)
    d = d[d.iso_o.isin(EST) & d.iso_d.isin(EST)]
    for nm, f in (("World Bank GDP, 2001-2021", d.year <= 2021), ("World Bank GDP, 2001-2024", d.year <= 2024)):
        for fe, (ff, regs) in LADDER.items():
            m = fit_stacked(d[f], regs, ff, pair_filter=fe == "pair")
            inter = [f"tech:{v}" for v in regs]; s, wp, k = wald(m, inter)
            rec("window", nm, fe, m, keep=["tech:"], wald=s, wald_p=wp)
    save()


PARTS = dict(patent_quality=part_patent_quality, rd_measure=part_rd_measure, institutions=part_institutions,
             hetero=part_hetero, wide=part_wide, window=part_window, ddd=part_ddd, iv=part_iv)
for name in (sys.argv[1:] or PARTS):
    print("==", name, flush=True)
    PARTS[name]()
save()
print("done", len(ROWS), flush=True)
