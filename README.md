# Sectoral differences in the gravity of cross-border M&A: replication package

Code and public data for a master's thesis on technology-intensive and other cross-border acquisitions in 46 economies, 2001-2021.

The deal data are licensed from Bureau van Dijk (Orbis M&A) and **are not in this repository**. This package offers three ways to check the work without them.

## What can be checked, and how

| Level | What you need | What it shows |
|---|---|---|
| 1. Published inference | Nothing beyond this repository | `verify_published_ri.py` recomputes all RI p-values in Table 5-6 from the published placebo draws |
| 2. Code on synthetic deals | Python | The whole pipeline runs, and the estimator recovers known true parameters on all pairs. The numbers say nothing about real M&A |
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

`src/make_synthetic_deals.py` invents about 120,000 deals from public CEPII GDP and distance, WIPO patents and a fixed seed. No Orbis record, value, count, country pair or share enters it. True parameters are round values set in the script, such as a destination GDP elasticity of 1.00 for other deals and 0.65 for technology deals.

`src/test_recovery.py` compares estimates with those values in `results/synthetic/recovery_test.csv`:

- **On all pairs, with zeros where no deal was drawn,** every parameter is recovered within two standard errors. This is the test of the estimator.
- **On transacting pairs only, as in the thesis panels,** level elasticities shrink (destination GDP 0.61 against a true 1.00) and a false sectoral difference appears (tech × ln(GDP_o) -0.24 against a true 0). The size depends on how sparse the invented data are, so these numbers cannot be carried over to Orbis. They illustrate why the thesis describes its results as intensive-margin only.

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
  20_ri_patent_quality.py   randomisation inference for Table 5-19; run_all.py runs it only with --reps 2000 or more
  21_ri_nochina.py          randomisation inference without China (Sections 5.2, 6.1)
  22_cheap_checks.py        largest-cells, subsample-difference and constant-price GDP checks
  15_all_pairs.py to 18_leave_one_out_joint.py   full country-pair grid with zeros and its placebo (Section 5.4)
  test_recovery.py          known-parameter test on the synthetic deals
  verify_published_ri.py    level 1 check
  check_against_thesis.py   level 3 check
data/public/         public series, with sources in SOURCES.csv
data/orbis/          empty; column list for licensed users
results/thesis/      statistics reported in the thesis and placebo draws; no deal-level or cell-level data
legacy/              Main.py and revision_specs.py, which built the original panels from the Orbis XML
tools/               licensed-data check and its pre-commit hook installer
```

`legacy/Main.py` still holds old 2SLS and Heckman routines that the thesis does not use. It is kept unchanged as a record of how the panels were built. It is not needed for levels 1 and 2.

## Licensed data: what is not published

The Orbis M&A licence (Bureau van Dijk, Moody's) does not allow redistribution of its records or of files built from them. None of the following is in this repository or its history:

- the deal extract (`deals_min.csv`, `deals_raw_full.csv.gz`) and the Orbis XML;
- the pair-year panels built from it (`panel_*.csv`), including cell-level deal values and counts;
- any deal identifier, company name or deal-level field.

What is published instead: public country-level and pair-level series (`data/public/`), estimated coefficients, standard errors, p-values and sample sizes (`results/thesis/`), and the placebo draws, which hold one coefficient vector per replication and no data. These are the statistics the thesis itself reports. `.gitignore` excludes the licensed files, and `tools/check_no_licensed_data.py` refuses a commit that contains a deal-extract or panel file name, an Orbis field name in a table header, or any file in `data/orbis/` other than its README. Run `python tools/install_hook.py` once after cloning to make it a pre-commit hook.

## Data sources

CEPII Gravity V202211; OECD Main Science and Technology Indicators; OECD FDI Regulatory Restrictiveness Index; WIPO IP Statistics Data Center; World Bank World Development Indicators and Worldwide Governance Indicators. Each provider's terms apply to its data. `data/public/SOURCES.csv` records URLs and download dates for the series fetched by script.

## Licence

The code is under the MIT licence (`LICENSE`). The public data keep the terms of their publishers listed above. No Orbis data are included or licensed.

## Software

Python 3.14, pyfixest 0.50.1. Exact versions are in `requirements.txt`. The placebo draws use independent streams derived from seed 20261105, so results do not depend on the number of processes.
