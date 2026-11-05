"""Download the public series added in the post-review revision into revisions/data.

Each file's URL, download time and size is logged to revisions/data/SOURCES.csv.
"""
import csv, datetime as dt, json, time
from pathlib import Path
import requests

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
DATA.mkdir(exist_ok=True)
HDR = {"User-Agent": "thesis-replication-script/1.0"}
LOG = []


def get(url, name, accept=None, timeout=240):
    h = dict(HDR)
    if accept:
        h["Accept"] = accept
    try:
        r = requests.get(url, headers=h, timeout=timeout)
    except Exception as e:
        print("FAIL", name, str(e)[:160]); return None
    print(r.status_code, name, len(r.content), url[:110])
    if r.status_code != 200:
        print("   ", r.text[:200].replace("\n", " "))
        return None
    (DATA / name).write_bytes(r.content)
    LOG.append(dict(file=name, url=url, bytes=len(r.content),
                    retrieved=dt.datetime.now().isoformat(timespec="seconds")))
    return r


OECD = "https://sdmx.oecd.org/public/rest"
CSV = "application/vnd.sdmx.data+csv; charset=utf-8; labels=both"

# all annual MSTI measures: triadic patent families, GOVERD and HERD are in this flow
if not (DATA / "oecd_msti_all.csv").exists():
  get(f"{OECD}/data/OECD.STI.STP,DSD_MSTI@DF_MSTI,/.A......?startPeriod=2000&dimensionAtObservation=AllDimensions",
    "oecd_msti_all.csv", CSV)
# FDI Regulatory Restrictiveness Index
if not (DATA / "oecd_fdirri_scores.csv").exists():
  get(f"{OECD}/data/OECD.DAF.INV,DSD_FDIRRI_SCORES@DF_FDIRRI_SCORES,/?startPeriod=1997",
    "oecd_fdirri_scores.csv", CSV)

WB = "https://api.worldbank.org/v2/country/all/indicator"
SERIES = {
    "wgi_pv": ("GOV_WGI_PV.EST", 3),      # political stability and absence of violence
    "wgi_rq": ("GOV_WGI_RQ.EST", 3),      # regulatory quality
    "wgi_rl": ("GOV_WGI_RL.EST", 3),      # rule of law
    "wdi_gdp": ("NY.GDP.MKTP.CD", 2),
    "wdi_pop": ("SP.POP.TOTL", 2),
    "wdi_rd": ("GB.XPD.RSDV.GD.ZS", 2),  # R&D % GDP, UNESCO UIS via the World Bank
    "wdi_gdpcap": ("NY.GDP.PCAP.CD", 2),
}
for name, (code, src) in SERIES.items():
    if (DATA / f"{name}.csv").exists():
        continue
    rows, page = [], 1
    while True:
        url = f"{WB}/{code}?source={src}&format=json&per_page=20000&date=1999:2024&page={page}"
        try:
            r = requests.get(url, headers=HDR, timeout=120)
            js = r.json()
        except Exception as e:
            print("FAIL", name, str(e)[:160]); break
        if not isinstance(js, list) or len(js) < 2 or js[1] is None:
            print("EMPTY", name, str(js)[:200]); break
        rows += js[1]
        if page >= js[0]["pages"]:
            break
        page += 1
        time.sleep(0.3)
    if rows:
        out = DATA / f"{name}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["iso3", "year", "value"])
            for x in rows:
                if x.get("countryiso3code"):
                    w.writerow([x["countryiso3code"], x["date"], x["value"]])
        print(200, name, len(rows))
        LOG.append(dict(file=out.name, url=f"{WB}/{code}?source={src}", bytes=out.stat().st_size,
                        retrieved=dt.datetime.now().isoformat(timespec="seconds")))

with open(DATA / "SOURCES.csv", "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, ["file", "url", "bytes", "retrieved"])
    if f.tell() == 0:
        w.writeheader()
    w.writerows(LOG)
