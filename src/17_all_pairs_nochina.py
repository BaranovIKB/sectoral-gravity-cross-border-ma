"""All-pairs estimates without China (as acquirer, then in any role), and a side-by-side table with the thesis.

Point estimates only: the thesis found tech x Patents_o vanishes without China, and this asks whether
the same holds with zeros kept.
"""
import warnings
import numpy as np, pandas as pd
import importlib.util, sys
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ap", HERE / "15_all_pairs.py")
ap = importlib.util.module_from_spec(spec); sys.modules["ap"] = ap; spec.loader.exec_module(ap)

ap.setup()
G = ap.G
d = ap.stack(G["grid"], G["key"], G["val"], G["lab"])
rows = []
SAMPLES = {"all": d, "no CHN acquirer": d[d.iso_o != "CHN"], "no CHN either side": d[(d.iso_o != "CHN") & (d.iso_d != "CHN")]}
for name, data in SAMPLES.items():
    for y in ("total_value", "deals"):
        for fe in ap.FE:
            m, s, wp = ap.fit(data, y, fe)
            c, p = m.coef(), m.pvalue()
            for v in ("patents_o_l1", "patents_d_l1", "ln_gdp_d", "ln_gdp_o", "ln_dist"):
                if f"tech:{v}" in c.index:
                    rows.append(dict(sample=name, y=y, fe=fe, N=int(m._N), var=v, coef=float(c[f"tech:{v}"]), p=float(p[f"tech:{v}"]), wald=s, wald_p=wp))
            print(name, y, fe, "Wald %.2f" % s, flush=True)
r = pd.DataFrame(rows)
r.to_csv(ap.OUT / "no_china.csv", index=False)
pd.set_option("display.width", 200, "display.max_rows", 300)
print(r.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
