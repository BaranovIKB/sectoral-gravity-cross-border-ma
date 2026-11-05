"""Compare a run on a licensed Orbis extract with the numbers printed in the thesis (Table 5-2, Table 5-15)."""
import warnings
import numpy as np, pandas as pd
from thesis_lib import *
warnings.filterwarnings("ignore")

EXPECT = {"year": (18672, 52.153, -0.36055, 1.24846e-6), "country": (18666, 44.843, 0.16350, 1.30720e-6), "pair": (17379, 32.055, 0.10504, 1.43689e-6)}
EXPECT_SHARE = {"country": (13047, 0.35340), "pair": (12021, 0.33388)}
deals = load_deals()
cov = load_covariates()
key, c2 = prepare_keys(deals, cov)
print(f"data: {DATA_TAG}, {len(deals):,} deals, {len(c2):,} dyad-years (thesis: 342,473 and 41,142)")
d = stacked_sector_lag(key, deals.deal_value.to_numpy(), deals.is_tech_target.to_numpy().astype(bool), c2)
ok = True
for k, st in ladder_stats(d).items():
    e = EXPECT[k]
    got = (st["N"], st["wald"], st["coef"]["ln_gdp_d"], st["coef"]["patents_d_l1"])
    same = got[0] == e[0] and abs(got[1] - e[1]) < 0.01 and abs(got[2] - e[2]) < 1e-4 and abs(got[3] - e[3]) < 1e-9
    ok &= same
    print(k, "N", got[0], "Wald %.3f" % got[1], "GDP_d %.5f" % got[2], "Patents_d %.5e" % got[3], "OK" if same else f"expected {e}")
idx = c2.dropna(subset=REGS).index.to_numpy()
c = c2.loc[idx].copy()
n = np.bincount(key, minlength=len(c2)); t = np.bincount(key, weights=deals.is_tech_target.to_numpy(), minlength=len(c2))
c["n"] = n[idx]; c["sh"] = t[idx] / c["n"]
for k, (N, b) in EXPECT_SHARE.items():
    m = fit_share(c, SHARE_LADDER[k][1], SHARE_LADDER[k][0])
    same = int(m._N) == N and abs(float(m.coef()["ln_gdp_d"]) - b) < 1e-4
    ok &= same
    print("share", k, int(m._N), "%.5f" % float(m.coef()["ln_gdp_d"]), "OK" if same else f"expected {N} {b}")
print("MATCHES THE THESIS" if ok else "DOES NOT MATCH THE THESIS")
