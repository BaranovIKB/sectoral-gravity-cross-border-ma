# Sectoral differences in the gravity of cross-border M&A: replication package

Code and public data for a master's thesis on technology-intensive and other cross-border acquisitions in 46 economies, 2001-2021.

The deal data are licensed from Bureau van Dijk (Orbis M&A) and **are not in this repository**. This package offers three ways to check the work without them.

## What can be checked, and how

| Level | What you need | What it shows |
|---|---|---|
| 1. Published inference | Nothing beyond this repository | `verify_published_ri.py` recomputes all RI p-values in Table 5-6 from the published placebo draws |
| 2. Code on synthetic deals | Python | The whole pipeline runs, and the estimators recover known true parameters. The numbers say nothing about real M&A |
| 3. Full replication | Your own Orbis M&A licence | Rebuilds the thesis estimates; `check_against_thesis.py` compares them with Tables 5-2 and 5-15 |

Levels 1 and 2 do not reproduce the thesis coefficients from raw data. That requires level 3.

## Quick start

```bash
pip install -r requirements.txt
python src/verify_published_ri.py
python run_all.py
```

`run_all.py` first generates synthetic deals, then runs every estimate into `results/synthetic/`. The default uses 200 placebo draws.

## Synthetic data

`src/make_synthetic_deals.py` invents about 120,000 deals from public CEPII GDP and distance, WIPO patents and a fixed seed. No Orbis record, value, count, country pair or share enters it. True parameters are round values set in the script, such as a destination GDP elasticity of 1.00 for other deals and 0.65 for technology deals. Comparing them with `results/synthetic/ri_table.csv` tests the code.

## Contents

```
src/                 analysis scripts
  make_synthetic_deals.py   invented deals with the Orbis extract's columns
  00_download_data.py       public OECD and World Bank series
  01_reproduce.py           robustness tables (Tables 5-11, 5-12)
  02_ri_placebo.py          randomisation inference, seed 20261105
  03_build_extra.py         assembles extra variables; needs the full CEPII file in data/raw/
  04_new_specs.py           extended tests (Tables 5-18 to 5-24)
  05_ri_summary.py          RI p-values, Monte Carlo errors, size distortion (Table 5-6)
  06_figures.py             Figures 3-7 and 5-3
  verify_published_ri.py    level 1 check
  check_against_thesis.py   level 3 check
data/public/         public series, with sources in SOURCES.csv
data/orbis/          empty; column list for licensed users
results/thesis/      statistics reported in the thesis and placebo draws; no deal-level or cell-level data
legacy/              Main.py and revision_specs.py, which built the original panels from the Orbis XML
```

`legacy/Main.py` still holds old 2SLS and Heckman routines that the thesis does not use. It is kept unchanged as a record of how the panels were built. It is not needed for levels 1 and 2.

## Data sources

CEPII Gravity V202211; OECD Main Science and Technology Indicators; OECD FDI Regulatory Restrictiveness Index; WIPO IP Statistics Data Center; World Bank World Development Indicators and Worldwide Governance Indicators. Each provider's terms apply to its data. `data/public/SOURCES.csv` records URLs and download dates for the series fetched by script.

## Software

Python 3.14, pyfixest 0.50.1. Exact versions are in `requirements.txt`. The placebo draws use independent streams derived from seed 20261105, so results do not depend on the number of processes.
