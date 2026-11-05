"""Redraw Figure 3-7 (correct variable label) and Figure 5-3 (10,000 replications)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy import stats
from thesis_lib import ROOT, RESULTS, PUBLIC

out = RESULTS / "figs"; out.mkdir(exist_ok=True)

# Figure 3-7: GOVERD, same data, colours and coverage rule as the original figure
m = pd.read_csv(PUBLIC / "msti_loaded.csv", keep_default_na=False, na_values=[""])
v = m.dropna(subset=["rd_intensity_gov"])
cov = v.groupby("year").iso3.nunique()
avg = v.groupby("year").rd_intensity_gov.mean()[cov >= 0.6 * cov.max()]
fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=200)
series = [("China", v[v.iso3 == "CHN"].set_index("year").rd_intensity_gov, "#0072B2", "-"),
          ("Russia", v[v.iso3 == "RUS"].set_index("year").rd_intensity_gov, "#D55E00", "--"),
          ("Reporting average", avg, "#009E73", ":")]
for n, s, c, ls in series:
    s = s[s.index <= 2023]
    ax.plot(s.index, s.values, color=c, ls=ls, lw=2.2, label=n)
    ax.annotate(n, (s.index[-1], s.values[-1]), xytext=(6, -2), textcoords="offset points", color=c, fontsize=8.5)
ax.set_xlim(2000, 2029); ax.set_xlabel("Year"); ax.set_ylabel("Government-performed R&D, % of GDP")
ax.grid(color="#dddddd", lw=.8); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(out / "figure_3_7.png"); plt.close(fig)

# Figure 5-3
B = pd.read_csv(RESULTS / "ri" / "ri_iid.csv").drop_duplicates("rep")
T = pd.read_csv(RESULTS / "ri_table.csv"); T = T[(T.design == "iid") & (T.stat == "joint_wald")].set_index("block").estimate
plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]
crit = stats.chi2.ppf(.95, 9)
fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.55), dpi=200, sharey=True)
for ax, col, title, ob in ((axes[0], "year_wald", "Year effects", T["year"]), (axes[1], "country_wald", "+ Country effects", T["country"])):
    ax.hist(B[col].dropna().clip(upper=70), bins=np.arange(0, 72, 2.4), color="#2B78D6", edgecolor="white", lw=.8)
    ax.axvline(crit, color="#666666", ls="--", lw=1.4)
    ax.axvline(ob, color="#EE6530", lw=2.6)
    top = np.histogram(B[col].clip(upper=70), bins=np.arange(0, 72, 2.4))[0].max() * 1.25
    ax.text(crit + 1, top * .90, f"nominal 5% critical\nvalue, χ²(9) = {crit:.1f}", fontsize=9.5, va="center", color="#333333")
    ax.text(ob + 1, top * .90, f"observed\n{ob:.2f}", fontsize=9.5, va="center", color="#333333")
    ax.set_title(title, fontsize=12); ax.set_xlim(-3.5, 71); ax.set_ylim(0, top)
    ax.set_xlabel("Joint Wald statistic, placebo relabelling", fontsize=11)
    ax.grid(axis="y", color="#e2e2e2", lw=.8); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Replications", fontsize=11)
axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout(); fig.savefig(out / "figure_5_3.png"); plt.close(fig)
print("max", B[["year_wald", "country_wald"]].max().to_dict(), "tall", np.histogram(B.year_wald.clip(upper=70), bins=np.arange(0, 72, 2.4))[0].max(), np.histogram(B.country_wald.clip(upper=70), bins=np.arange(0, 72, 2.4))[0].max())
