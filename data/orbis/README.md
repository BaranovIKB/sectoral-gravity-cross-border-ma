# Where the licensed data go

This folder is empty on purpose. Orbis M&A is licensed by Bureau van Dijk (Moody's), and no deal record, aggregate of deals, or file built from them is in this repository.

If your institution has its own Orbis M&A licence, export your deals and save `deals_min.csv` here. It needs these columns:

| Column | Content |
|---|---|
| `iso_o`, `iso_d` | Acquirer and target country, ISO 3166-1 alpha-3, from `BIDDER_CTRYC` and `TARGET_CTRYC` |
| `year` | Year of `COMPLETION_DATE` |
| `deal_value` | `ALLDLVALUE`, USD thousand, empty if undisclosed |
| `is_tech_target`, `is_tech_acquirer` | 1 if any target or acquirer SIC code is in `TECH_SECTOR_CODES` in `legacy/Main.py` |
| `TARGET_SCUSCD`, `BIDDER_SCUSCD` | US SIC codes as exported, separated by ` / ` |
| `BIDDER_STRATEGIC_CTRYC`, `STAKE`, `FINLSTAK`, `DEALTYPE` | As exported |

Then run:

```bash
THESIS_DATA=orbis python run_all.py --reps 10000 --block-reps 2000
```

`src/check_against_thesis.py` then compares your run with the thesis (Tables 5-2 and 5-15). The thesis extract gives 342,473 deals in the 2000-2024 window. A newer download may differ, since Orbis revises its records, so exact equality is not guaranteed.

Read your files with `keep_default_na=False`: "NA" is the ISO code of Namibia.
