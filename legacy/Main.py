"""
================================================================================
Main.py - Gravity Model for M&A Analysis (Master's Thesis)
================================================================================
This script implements gravity models for analyzing cross-border M&A deals
with a focus on technological determinants (R&D intensity and patents).

STRUCTURE FOR THESIS:
---------------------
1. BASELINE MODELS (Sections 4.2-4.3)
   - PPML and OLS gravity models on full, tech, and non-tech panels
   - Output: ppml_results.txt, ols_results.txt

2. ENDOGENEITY TESTS (Section 5.1)
   - Lagged variables to address reverse causality
   - Output: ppml_lagged_results.txt, ols_lagged_results.txt

3. IV/2SLS ESTIMATION (Section 5.2)
   - Using lagged R&D/patents as internal instruments
   - Durbin-Wu-Hausman tests
   - Output: iv_2sls_results.txt, iv_hausman_tests.txt

4. FIXED EFFECTS (Section 5.3)
   - Origin, destination, and year fixed effects
   - Output: ppml_fe_results.txt, ols_fe_results.txt

5. SAMPLE SELECTION (Section 5.4)
   - Diagnostics comparing included vs excluded observations
   - Heckman two-step selection model
   - Output: selection_diagnostics.csv, heckman_results.txt

6. ROBUSTNESS CHECKS (Section 6.1)
   - Alternative dependent variables, R&D specifications
   - Output: robustness_*.txt, robustness_summary.csv

7. MECHANISM TESTS (Section 6.2)
   - Tech vs non-tech interaction effects
   - Output: mechanism_results.txt

================================================================================
"""

import os
import pandas as pd
import numpy as np
from lxml import etree
import re
import warnings
import traceback
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from scipy import stats
from statsmodels.iolib.summary2 import summary_col

import revision_specs as rv

# ==============================================================================
# EXPORT LAYER (v14)
# One run of this script must leave behind everything any later check could
# need, so that no further run is ever required. Exports are written BEFORE any
# estimation, and every export is fail-safe: an error is reported loudly but
# never stops the pipeline. EXPORT_MANIFEST.txt lists what was written, with
# row counts, columns and MD5 checksums, and a validation block at the end says
# in one line whether the export is complete.
# ==============================================================================
EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "export_v14")
_EXPORT_LOG = []
_EXPORT_REQUIRED = {
    "deals_raw_full.csv.gz": ["RECORD_ID", "COMPLETION_DATE", "BIDDER_CTRYC", "TARGET_CTRYC",
                              "BIDDER_STRATEGIC_CTRYC", "ALLDLVALUE", "TRGTVAL",
                              "TARGET_SCUSCD", "BIDDER_SCUSCD", "STAKE", "FINLSTAK"],
    "patents_loaded.csv": ["iso3", "year", "patent_applications"],
    "msti_loaded.csv": ["iso3", "year", "rd_intensity_gov", "rd_intensity_gross"],
    "gravity_cepii_2000_2024.csv.gz": ["isoo", "isod", "year", "gdp_o", "gdp_d", "dist"],
    "deals_min.csv": ["iso_o", "iso_d", "year", "deal_value", "is_tech_target", "is_tech_acquirer"],
}


def _md5(path, chunk=1 << 20):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def export_df(df, name):
    """Write one export. Never raises: a failed export is logged and reported."""
    try:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        path = os.path.join(EXPORT_DIR, name)
        comp = "gzip" if name.endswith(".gz") else None
        df.to_csv(path, index=False, compression=comp, encoding="utf-8")
        size = os.path.getsize(path)
        rec = dict(name=name, ok=True, rows=len(df), cols=list(df.columns),
                   bytes=size, md5=_md5(path))
        print(f"[EXPORT OK] {name}: {len(df):,} rows x {df.shape[1]} cols, {size/1e6:.1f} MB")
    except Exception as e:
        rec = dict(name=name, ok=False, error=f"{type(e).__name__}: {e}")
        print(f"[EXPORT FAILED] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
    _EXPORT_LOG[:] = [r for r in _EXPORT_LOG if r["name"] != name] + [rec]
    _write_manifest()
    return rec.get("ok", False)


_README = """HOW TO READ THESE FILES -- read this first

Every file that carries two-letter ISO country codes (deals_raw_full.csv.gz and
the *_CTRYC columns of deals_min.csv) must be read with

    pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])

'NA' is Namibia. With default settings pandas turns it into a missing value and
the Namibian deals disappear without any warning.

deals_raw_full.csv.gz   every record of the Orbis XML, every field, before any
                        filter (year, country mapping, cross-border, value).
                        The whole pipeline can be rebuilt from it plus the three
                        loaded series below.
deals_min.csv           the filtered cross-border deal table as used by the
                        pipeline, with is_tech_target, is_tech_acquirer and the
                        raw SIC / stake / strategic-country fields.
msti_loaded.csv         OECD MSTI R&D intensities after loading.
patents_loaded.csv      WIPO patent applications by origin after loading.
gravity_cepii_2000_2024.csv.gz   CEPII gravity subset after loading.
EXPORT_MANIFEST.txt     rows, columns and MD5 of each file.
"""


def _write_manifest():
    try:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        with open(os.path.join(EXPORT_DIR, "README_READ_FIRST.txt"), "w", encoding="utf-8") as f:
            f.write(_README)
        with open(os.path.join(EXPORT_DIR, "EXPORT_MANIFEST.txt"), "w", encoding="utf-8") as f:
            for r in _EXPORT_LOG:
                if r["ok"]:
                    f.write(f"{r['name']}\trows={r['rows']}\tbytes={r['bytes']}\tmd5={r['md5']}\n")
                    f.write(f"    columns: {', '.join(map(str, r['cols']))}\n")
                else:
                    f.write(f"{r['name']}\tFAILED\t{r['error']}\n")
    except Exception as e:
        print(f"[EXPORT] manifest could not be written: {e}")


def validate_exports():
    """One-line verdict on whether the export is complete. Printed twice: after
    the last export, and again as the final line of the run."""
    done = {r["name"]: r for r in _EXPORT_LOG}
    problems = []
    for name, need in _EXPORT_REQUIRED.items():
        r = done.get(name)
        if r is None:
            problems.append(f"{name}: not written")
        elif not r["ok"]:
            problems.append(f"{name}: FAILED ({r['error']})")
        else:
            miss = [c for c in need if c not in r["cols"]]
            if miss:
                problems.append(f"{name}: missing columns {miss}")
            if r["rows"] == 0:
                problems.append(f"{name}: zero rows")
    print("\n" + "=" * 78)
    if problems:
        print("EXPORT VALIDATION: INCOMPLETE -- send this block for review")
        for pr in problems:
            print("   - " + pr)
    else:
        print(f"EXPORT VALIDATION: COMPLETE -- all {len(_EXPORT_REQUIRED)} exports written "
              f"with required columns. Folder: {EXPORT_DIR}")
    print("=" * 78 + "\n")
    return not problems


sns.set(style="whitegrid")

# ================================================================================
# DATA DIRECTORIES AND FILE PATHS
# ================================================================================

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

MNA_XML_FILE = os.path.join(DATA_DIR, "M&A", "merged_deals.xml")
OECD_MSTI_GV = os.path.join(DATA_DIR, "Government R&D", "OECD.STI.STP,DSD_MSTI@DF_MSTI,+.A.GV.PT_B1GQ...csv")
OECD_MSTI_GDP = os.path.join(DATA_DIR, "R&D per GDP", "OECD.STI.STP,DSD_MSTI@DF_MSTI,+.A.G.PT_B1GQ...csv")
PATENTS_FILE = os.path.join(DATA_DIR, "Patents", "countryprofiles_1 - Country profiles_Total count by applicant's origin_2000_2024.csv")

DIST_FILE = None

# ================================================================================
# CONSTANTS AND MAPPINGS
# ================================================================================

ISO2_TO_ISO3 = {
'AF': 'AFG', 'AX': 'ALA', 'AL': 'ALB', 'DZ': 'DZA', 'AS': 'ASM', 'AD': 'AND', 'AO': 'AGO',
'AI': 'AIA', 'AQ': 'ATA', 'AG': 'ATG', 'AR': 'ARG', 'AM': 'ARM', 'AW': 'ABW', 'AU': 'AUS',
'AT': 'AUT', 'AZ': 'AZE', 'BS': 'BHS', 'BH': 'BHR', 'BD': 'BGD', 'BB': 'BRB', 'BY': 'BLR',
'BE': 'BEL', 'BZ': 'BLZ', 'BJ': 'BEN', 'BM': 'BMU', 'BT': 'BTN', 'BO': 'BOL', 'BA': 'BIH',
'BW': 'BWA', 'BV': 'BVT', 'BR': 'BRA', 'VG': 'VGB', 'IO': 'IOT', 'BN': 'BRN', 'BG': 'BGR',
'BF': 'BFA', 'BI': 'BDI', 'KH': 'KHM', 'CM': 'CMR', 'CA': 'CAN', 'CV': 'CPV', 'KY': 'CYM',
'CF': 'CAF', 'TD': 'TCD', 'CL': 'CHL', 'CN': 'CHN', 'HK': 'HKG', 'MO': 'MAC', 'CX': 'CXR',
'CC': 'CCK', 'CO': 'COL', 'KM': 'COM', 'CG': 'COG', 'CD': 'COD', 'CK': 'COK', 'CR': 'CRI',
'CI': 'CIV', 'HR': 'HRV', 'CU': 'CUB', 'CY': 'CYP', 'CZ': 'CZE', 'DK': 'DNK', 'DJ': 'DJI',
'DM': 'DMA', 'DO': 'DOM', 'EC': 'ECU', 'EG': 'EGY', 'SV': 'SLV', 'GQ': 'GNQ', 'ER': 'ERI',
'EE': 'EST', 'ET': 'ETH', 'FK': 'FLK', 'FO': 'FRO', 'FJ': 'FJI', 'FI': 'FIN', 'FR': 'FRA',
'GF': 'GUF', 'PF': 'PYF', 'TF': 'ATF', 'GA': 'GAB', 'GM': 'GMB', 'GE': 'GEO', 'DE': 'DEU',
'GH': 'GHA', 'GI': 'GIB', 'GR': 'GRC', 'GL': 'GRL', 'GD': 'GRD', 'GP': 'GLP', 'GU': 'GUM',
'GT': 'GTM', 'GG': 'GGY', 'GN': 'GIN', 'GW': 'GNB', 'GY': 'GUY', 'HT': 'HTI', 'HM': 'HMD',
'VA': 'VAT', 'HN': 'HND', 'HU': 'HUN', 'IS': 'ISL', 'IN': 'IND', 'ID': 'IDN', 'IR': 'IRN',
'IQ': 'IRQ', 'IE': 'IRL', 'IM': 'IMN', 'IL': 'ISR', 'IT': 'ITA', 'JM': 'JAM', 'JP': 'JPN',
'JE': 'JEY', 'JO': 'JOR', 'KZ': 'KAZ', 'KE': 'KEN', 'KI': 'KIR', 'KP': 'PRK', 'KR': 'KOR',
'KW': 'KWT', 'KG': 'KGZ', 'LA': 'LAO', 'LV': 'LVA', 'LB': 'LBN', 'LS': 'LSO', 'LR': 'LBR',
'LY': 'LBY', 'LI': 'LIE', 'LT': 'LTU', 'LU': 'LUX', 'MK': 'MKD', 'MG': 'MDG', 'MW': 'MWI',
'MY': 'MYS', 'MV': 'MDV', 'ML': 'MLI', 'MT': 'MLT', 'MH': 'MHL', 'MQ': 'MTQ', 'MR': 'MRT',
'MU': 'MUS', 'YT': 'MYT', 'MX': 'MEX', 'FM': 'FSM', 'MD': 'MDA', 'MC': 'MCO', 'MN': 'MNG',
'ME': 'MNE', 'MS': 'MSR', 'MA': 'MAR', 'MZ': 'MOZ', 'MM': 'MMR', 'NA': 'NAM', 'NR': 'NRU',
'NP': 'NPL', 'NL': 'NLD', 'AN': 'ANT', 'NC': 'NCL', 'NZ': 'NZL', 'NI': 'NIC', 'NE': 'NER',
'NG': 'NGA', 'NU': 'NIU', 'NF': 'NFK', 'MP': 'MNP', 'NO': 'NOR', 'OM': 'OMN', 'PK': 'PAK',
'PW': 'PLW', 'PS': 'PSE', 'PA': 'PAN', 'PG': 'PNG', 'PY': 'PRY', 'PE': 'PER', 'PH': 'PHL',
'PN': 'PCN', 'PL': 'POL', 'PT': 'PRT', 'PR': 'PRI', 'QA': 'QAT', 'RE': 'REU', 'RO': 'ROU',
'RU': 'RUS', 'RW': 'RWA', 'BL': 'BLM', 'SH': 'SHN', 'KN': 'KNA', 'LC': 'LCA', 'MF': 'MAF',
'PM': 'SPM', 'VC': 'VCT', 'WS': 'WSM', 'SM': 'SMR', 'ST': 'STP', 'SA': 'SAU', 'SN': 'SEN',
'RS': 'SRB', 'SC': 'SYC', 'SL': 'SLE', 'SG': 'SGP', 'SK': 'SVK', 'SI': 'SVN', 'SB': 'SLB',
'SO': 'SOM', 'ZA': 'ZAF', 'GS': 'SGS', 'SS': 'SSD', 'ES': 'ESP', 'LK': 'LKA', 'SD': 'SDN',
'SR': 'SUR', 'SJ': 'SJM', 'SZ': 'SWZ', 'SE': 'SWE', 'CH': 'CHE', 'SY': 'SYR', 'TW': 'TWN',
'TJ': 'TJK', 'TZ': 'TZA', 'TH': 'THA', 'TL': 'TLS', 'TG': 'TGO', 'TK': 'TKL', 'TO': 'TON',
'TT': 'TTO', 'TN': 'TUN', 'TR': 'TUR', 'TM': 'TKM', 'TC': 'TCA', 'TV': 'TUV', 'UG': 'UGA',
'UA': 'UKR', 'AE': 'ARE', 'GB': 'GBR', 'US': 'USA', 'UM': 'UMI', 'UY': 'URY', 'UZ': 'UZB',
'VU': 'VUT', 'VE': 'VEN', 'VN': 'VNM', 'VI': 'VIR', 'WF': 'WLF', 'EH': 'ESH', 'YE': 'YEM',
'ZM': 'ZMB', 'ZW': 'ZWE', 'XK': 'XKX',
}

# Tech sector SIC codes for high-tech industries
TECH_SECTOR_CODES = [
    3510, 3523, 3524, 3530, 3531, 3532, 3537, 3540, 3541,
    3550, 3555, 3559, 3560, 3561, 3562, 3564, 3567, 3569,
    3570, 3571, 3572, 3575, 3576, 3577, 3578, 3579,
    3580, 3585, 3590, 4812, 4813, 4822, 4832, 4833,
    4841, 4899, 7370, 7371, 7372, 7373, 7374,
]

# ================================================================================
# DATA LOADING FUNCTIONS
# ================================================================================

def get_single_text(item):
    """Extract text from XML element."""
    if item is None:
        return None
    if item.text is not None and item.text.strip():
        return item.text.strip()
    for child in item.iter():
        if child.text is not None and child.tag.endswith("childItem"):
            return child.text.strip()
    return None

def parse_mna_xml(xml_path, max_records=None):
    """Parse Orbis XML file of M&A deals."""
    ns = {'ns': 'http://www.bvdep.com/schemas/RemoteAccessDataResults.xsd'}
    records = []
    context = etree.iterparse(xml_path, tag="{http://www.bvdep.com/schemas/RemoteAccessDataResults.xsd}record")
    for i, (event, elem) in enumerate(context):
        if max_records is not None and i >= max_records:
            break
        record = {}
        for item in elem.findall("ns:item", ns):
            field = item.get("field")
            if field is None:
                continue
            text = get_single_text(item)
            record[field] = text
        if len(record) == 0:
            for item in elem.findall("item"):
                field = item.get("field")
                if field is None:
                    continue
                text = get_single_text(item)
                record[field] = text
        if len(record) > 0:
            records.append(record)
        elem.clear()
    df = pd.DataFrame(records)
    return df

def load_msti_series(path, expected_measure, value_col):
    """Load OECD MSTI R&D intensity data."""
    df = pd.read_csv(path)
    print(f"Loading {path}: {len(df)} rows")
    if "MEASURE" in df.columns:
        df = df[df["MEASURE"] == expected_measure].copy()
        print(f"After MEASURE={expected_measure} filter: {len(df)} rows")
    value_cols = ["OBS_VALUE"]
    numeric_col = None
    for col in value_cols:
        if col in df.columns:
            if not df[col].isna().all():
                numeric_col = col
                break
    if numeric_col is None:
        print(f"ERROR: No numeric data found in {value_cols}")
        return pd.DataFrame()
    print(f"Using numeric column '{numeric_col}'")
    df = df.rename(columns={numeric_col: "OBS_VALUE"})
    df = df.rename(columns={"REF_AREA": "iso3", "TIME_PERIOD": "year", "OBS_VALUE": value_col})
    df = df[["iso3", "year", value_col]].drop_duplicates()
    df = df.groupby(["iso3", "year"], as_index=False)[value_col].mean()
    print(f"Final {value_col}: {len(df)} rows")
    return df

def load_rd_intensities():
    """Load both government and gross R&D intensity."""
    msti_gov = load_msti_series(OECD_MSTI_GV, expected_measure="GV", value_col="rd_intensity_gov")
    msti_gross = load_msti_series(OECD_MSTI_GDP, expected_measure="G", value_col="rd_intensity_gross")
    msti_rd = msti_gov.merge(msti_gross, on=["iso3", "year"], how="outer")
    return msti_rd

def load_patents(path=PATENTS_FILE):
    """Load WIPO patent counts and reshape to iso3-year format."""
    # keep_default_na=False: 'NA' is Namibia's ISO2 code, and pandas would
    # otherwise read it as a missing value and silently drop the country
    df = pd.read_csv(path, skiprows=5, keep_default_na=False, na_values=[""])
    
    df["iso2"] = df["Origin"].str.strip().str.upper()
    df["iso3"] = df["iso2"].map(ISO2_TO_ISO3)
    df = df.dropna(subset=["iso3"])
    
    year_cols = [col for col in df.columns if col.isdigit() and 2000 <= int(col) <= 2024]
    
    df_long = pd.melt(
        df, 
        id_vars=["iso3"], 
        value_vars=year_cols,
        var_name="year", 
        value_name="patent_applications"
    )
    
    df_long["year"] = df_long["year"].astype(int)
    df_long["patent_applications"] = pd.to_numeric(df_long["patent_applications"], errors="coerce")
    df_long = df_long.dropna(subset=["patent_applications"])
    df_long = df_long.groupby(["iso3", "year"], as_index=False)["patent_applications"].sum()
    
    print(f"Loaded patents: {len(df_long)} country-years, {df_long['iso3'].nunique()} countries")
    
    return df_long

def load_gravity_data():
    """
    Load CEPII Gravity data from Gravity_V202211.csv.
    
    Returns:
        DataFrame with columns: isoo, isod, year, gdp_o, gdp_d, dist
        - Filtered to year 2000-2024
        - Rows with null or <= 0 values dropped
    """
    gravity_file = os.path.join(DATA_DIR, "Gravity", "Gravity_csv_V202211", "Gravity_V202211.csv")
    
    if not os.path.exists(gravity_file):
        raise RuntimeError(f"CEPII gravity file not found at: {gravity_file}")
    
    print(f"Loading CEPII gravity data from {gravity_file}...")

    # Select and rename columns: iso3_o, iso3_d, year, gdp_o, gdp_d, dist -> dist
    # Note: CEPII Gravity V202211 uses 'dist' for distance (weighted average)
    # Alternative distance columns: distw_harmonic, distw_arithmetic, distcap
    # revision: pop_o/pop_d carried through for the income control (SPEC 5)
    # revision: pull the standard bilateral gravity controls as well. Their absence
    # from the earlier specification is an omitted-variable problem the defence
    # committee raised directly, and contiguity / common language / colonial ties
    # are the usual proxies for the institutional and cultural proximity that a
    # raw distance term cannot capture.
    want = ["iso3_o", "iso3_d", "year", "gdp_o", "gdp_d", "dist", "pop_o", "pop_d",
            "contig", "comlang_off", "colony", "comcol", "rta",
            "wto_o", "wto_d", "eu_o", "eu_d"]

    # Gravity_V202211.csv is ~4.5 M rows by ~90 columns. Reading it whole and then
    # subsetting needs several GB and fails with "C error: out of memory" on a
    # machine that has anything else open. Read only the columns actually used,
    # in chunks, and drop out-of-window years as we go.
    header = pd.read_csv(gravity_file, nrows=0)
    have = [c for c in want if c in header.columns]
    missing = [c for c in want if c not in header.columns]
    if missing:
        print(f"WARNING: CEPII columns not found and skipped: {missing}")

    chunks = []
    for chunk in pd.read_csv(gravity_file, usecols=have, low_memory=False,
                             chunksize=500_000):
        chunk = chunk[(chunk["year"] >= 2000) & (chunk["year"] <= 2024)]
        if len(chunk):
            chunks.append(chunk)
    gravity = (pd.concat(chunks, ignore_index=True) if chunks
               else pd.DataFrame(columns=have))
    del chunks
    gravity = gravity.rename(columns={"iso3_o": "isoo", "iso3_d": "isod"})
    print(f"After year filter (2000-2024): {len(gravity)} rows")
    
    # Drop rows where gdp_o, gdp_d, or dist is null or <= 0
    gravity = gravity.dropna(subset=["gdp_o", "gdp_d", "dist"])
    gravity = gravity[gravity["gdp_o"] > 0]
    gravity = gravity[gravity["gdp_d"] > 0]
    gravity = gravity[gravity["dist"] > 0]
    print(f"After dropping null/zero values: {len(gravity)} rows")
    
    return gravity

# ================================================================================
# DATA PROCESSING FUNCTIONS
# ================================================================================

def parse_sic_codes(sic_string):
    """Parse SIC codes from string."""
    if pd.isna(sic_string) or sic_string is None:
        return []
    codes = []
    for part in sic_string.split('/'):
        part = part.strip()
        if part.isdigit() and len(part) == 4:
            codes.append(int(part))
    return codes

def create_sector_dummies(deals_df):
    """Create dummy variables for tech sectors."""
    df = deals_df.copy()
    df['target_sic_list'] = df['TARGET_SCUSCD'].apply(parse_sic_codes)
    df['bidder_sic_list'] = df['BIDDER_SCUSCD'].apply(parse_sic_codes)
    for code in TECH_SECTOR_CODES:
        col_name_target = f"target_sic_{code}"
        col_name_bidder = f"bidder_sic_{code}"
        df[col_name_target] = df['target_sic_list'].apply(lambda x: 1 if code in x else 0)
        df[col_name_bidder] = df['bidder_sic_list'].apply(lambda x: 1 if code in x else 0)
    df = df.drop(columns=['target_sic_list', 'bidder_sic_list'])
    return df

def aggregate_sector_dummies(deals_df, group_cols=["iso_o", "iso_d", "year"]):
    """Aggregate sector dummies to dyad-year level."""
    sector_cols = [col for col in deals_df.columns if col.startswith("target_sic_") or col.startswith("bidder_sic_")]
    agg_dict = {col: "max" for col in sector_cols}
    if "deal_id" in deals_df.columns:
        agg_dict["deal_id"] = "count"
    sector_agg = deals_df.groupby(group_cols, as_index=False).agg(agg_dict)
    if "deal_id" in sector_agg.columns:
        sector_agg = sector_agg.rename(columns={"deal_id": "ma_deals_total"})
    return sector_agg

def create_lagged_variables(panel_data, lag_vars=None):
    """
    Create country-year lags for R&D intensity and patent variables.

    NOTE (revision): the previous implementation did
        df.sort_values(['iso_o','year']); df.groupby('iso_o')[var].shift(1)
    on a DYAD-year panel. shift(1) moves back one ROW, not one YEAR, and each
    origin-year repeats once per destination -- so ~95%+ of the resulting
    "lag" was the CONTEMPORANEOUS value of the same year. That invalidated the
    lag identification strategy and made the IV first stage a near-identity
    (coef 0.9999, se 0.0005 in the old Table 13).

    Now delegates to revision_specs.make_lags(), which lags on the unique
    country-year grid and merges back. Produces both _l1 and _l2.
    """
    df = rv.make_lags(panel_data, lag_vars=lag_vars, lags=(1, 2))
    for v in (lag_vars or rv.LAG_BASE):
        if v + "_l1" in df.columns:
            print(f"Created lag: {v}_l1")
    return df


# ================================================================================
# DIAGNOSTIC FUNCTIONS
# ================================================================================

def compute_vif(ols_data, exog_vars):
    """
    Compute Variance Inflation Factors for regressors.
    VIF > 10 indicates problematic multicollinearity.
    """
    print("\n=== Computing Variance Inflation Factors (VIF) ===")
    
    vif_data = ols_data[exog_vars].dropna()
    
    if len(vif_data) < 10:
        print("WARNING: Insufficient data for VIF calculation")
        return pd.DataFrame()
    
    vif_results = []
    for i, var in enumerate(exog_vars):
        try:
            vif_value = variance_inflation_factor(vif_data.values, i)
            vif_results.append({'Variable': var, 'VIF': vif_value})
        except:
            vif_results.append({'Variable': var, 'VIF': np.nan})
    
    vif_df = pd.DataFrame(vif_results)
    print(vif_df.to_string(index=False))
    
    high_vif = vif_df[vif_df['VIF'] > 10]
    if len(high_vif) > 0:
        print("\nWARNING: Variables with VIF > 10 (potential multicollinearity):")
        print(high_vif.to_string(index=False))
    
    return vif_df

def heteroskedasticity_tests(ols_results, output_dir):
    """Perform Breusch-Pagan and White tests for heteroskedasticity."""
    print("\n=== Heteroskedasticity Tests ===")
    
    results = []
    
    try:
        bp_test = het_breuschpagan(ols_results.resid, ols_results.model.exog)
        print(f"Breusch-Pagan test: LM={bp_test[0]:.4f}, p-value={bp_test[1]:.4f}")
        results.append({
            'Test': 'Breusch-Pagan',
            'LM_statistic': bp_test[0],
            'p_value': bp_test[1],
            'Conclusion': 'Reject H0 (heteroskedastic)' if bp_test[1] < 0.05 else 'Cannot reject H0'
        })
    except Exception as e:
        print(f"Breusch-Pagan test failed: {e}")
    
    try:
        white_test = het_white(ols_results.resid, ols_results.model.exog)
        print(f"White test: LM={white_test[0]:.4f}, p-value={white_test[1]:.4f}")
        results.append({
            'Test': 'White',
            'LM_statistic': white_test[0],
            'p_value': white_test[1],
            'Conclusion': 'Reject H0 (heteroskedastic)' if white_test[1] < 0.05 else 'Cannot reject H0'
        })
    except Exception as e:
        print(f"White test failed: {e}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "heteroskedasticity_tests.csv"), index=False)
    
    return results_df

def residual_diagnostics(ols_results, output_dir, label):
    """Create residual plots and histograms for OLS diagnostics."""
    print(f"\n=== Residual Diagnostics for {label} ===")
    
    residuals = ols_results.resid
    fitted = ols_results.fittedvalues
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(fitted, residuals, alpha=0.3)
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_xlabel('Fitted values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'Residuals vs Fitted - {label}')
    
    axes[1].hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 
                 'r-', linewidth=2, label='Normal distribution')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Residual Distribution - {label}')
    axes[1].legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"residuals_{label}.png"), dpi=300)
    plt.close(fig)
    print(f"Saved residual plots to {output_dir}/residuals_{label}.png")

def check_dispersion(panel_data, value_col='total_value'):
    """Check for over/under-dispersion in count data for PPML."""
    print("\n=== Dispersion Check for PPML ===")
    
    data = panel_data[panel_data[value_col].notna() & (panel_data[value_col] >= 0)]
    
    mean_val = data[value_col].mean()
    var_val = data[value_col].var()
    
    print(f"Mean of {value_col}: {mean_val:.4f}")
    print(f"Variance of {value_col}: {var_val:.4f}")
    print(f"Variance/Mean ratio: {var_val/mean_val if mean_val > 0 else np.nan:.4f}")
    
    if var_val > mean_val:
        print("=> Evidence of OVER-dispersion. Consider Negative Binomial model.")
    elif var_val < mean_val:
        print("=> Evidence of UNDER-dispersion.")
    else:
        print("=> Dispersion is approximately equal (Poisson appropriate).")
    
    return {'mean': mean_val, 'variance': var_val, 'dispersion_ratio': var_val/mean_val if mean_val > 0 else np.nan}

# ================================================================================
# MODEL ESTIMATION FUNCTIONS
# ================================================================================

def run_gravity_models(panel_data, output_dir, label, include_lags=False, lagged_vars=None):
    """
    Run PPML and OLS models on a given panel and save results.
    
    IMPORTANT CHANGES FROM ORIGINAL:
    - PPML now keeps zero flows (total_value >= 0), only drops NaN
    - OLS is documented as "positive-flows only" specification
    - Includes basic diagnostics
    """
    print(f"\n{'='*60}")
    print(f"Running models for {label} panel")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # PPML MODEL (FIXED: keeps zeros, drops NaN only)
    print(f"\n--- PPML Estimation (label={label}) ---")
    
    ppml_vars = ["total_value", "ln_gdp_o", "ln_gdp_d"]
    ppml_data = panel_data.copy()
    
    for var in ppml_vars:
        ppml_data = ppml_data[ppml_data[var].notna()]
    
    # Keep total_value >= 0 (zeros allowed)
    ppml_data = ppml_data[ppml_data["total_value"] >= 0].copy()
    
    n_ppml_total = len(ppml_data)
    n_ppml_zeros = (ppml_data["total_value"] == 0).sum()
    print(f"PPML: {n_ppml_total} observations ({n_ppml_zeros} with zero value)")
    
    if include_lags and lagged_vars:
        tech_vars = lagged_vars
    else:
        tech_vars = [
            "rd_intensity_gov_o", "rd_intensity_gov_d",
            "rd_intensity_gross_o", "rd_intensity_gross_d",
            "patents_o", "patents_d"
        ]
    
    ppml_formula = f"total_value ~ ln_gdp_o + ln_gdp_d + {' + '.join(tech_vars)}"
    if "ln_dist" in ppml_data.columns:
        ppml_formula += " + ln_dist"
    
    ppml_results = None
    try:
        model = smf.glm(formula=ppml_formula, data=ppml_data, family=sm.families.Poisson())
        ppml_results = model.fit(cov_type="HC0")
        print(ppml_results.summary())
        
        with open(os.path.join(output_dir, "ppml_results.txt"), "w") as f:
            f.write(ppml_results.summary().as_text())
        
        vars_of_interest = tech_vars.copy()
        for v in ["ln_gdp_o", "ln_gdp_d", "ln_dist"]:
            if v in ppml_results.params.index:
                vars_of_interest.append(v)
        plot_coefficients(ppml_results, output_dir, vars_of_interest)
        
    except Exception as e:
        print(f"PPML failed for {label}: {e}")
    
    # OLS MODEL (positive flows only - documented selection)
    print(f"\n--- OLS Estimation (label={label}) ---")
    print("NOTE: OLS uses positive-flows only specification (total_value > 0)")
    print("      This introduces potential sample selection bias.")
    
    panel_data["ln_total_value"] = np.log(panel_data["total_value"].replace(0, np.nan))
    
    ols_vars = ["ln_total_value", "ln_gdp_o", "ln_gdp_d"]
    ols_data = panel_data.dropna(subset=ols_vars).copy()
    ols_data = ols_data[ols_data["ln_total_value"] > -np.inf].copy()
    
    n_ols_total = len(ols_data)
    print(f"OLS: {n_ols_total} observations (positive flows only)")
    
    ols_results = None
    if len(ols_data) > 0:
        formula_ols = f"ln_total_value ~ ln_gdp_o + ln_gdp_d + {' + '.join(tech_vars)}"
        if "ln_dist" in ols_data.columns and not ols_data["ln_dist"].isna().all():
            formula_ols += " + ln_dist"
        
        try:
            ols_model = smf.ols(formula=formula_ols, data=ols_data)
            ols_results = ols_model.fit(cov_type="HC0")
            print(ols_results.summary())
            
            with open(os.path.join(output_dir, "ols_results.txt"), "w") as f:
                f.write(ols_results.summary().as_text())
            
            exog_vars = ["ln_gdp_o", "ln_gdp_d"] + tech_vars
            if "ln_dist" in ols_data.columns:
                exog_vars.append("ln_dist")
            exog_vars = [v for v in exog_vars if v in ols_data.columns]
            
            vif_df = compute_vif(ols_data, exog_vars)
            if len(vif_df) > 0:
                vif_df.to_csv(os.path.join(output_dir, "vif_results.csv"), index=False)
            
            heteroskedasticity_tests(ols_results, output_dir)
            residual_diagnostics(ols_results, output_dir, label)
            
        except Exception as e:
            print(f"OLS estimation failed for {label}: {e}")
    
    panel_data.to_csv(os.path.join(output_dir, "gravity_panel.csv"), index=False)
    
    return ppml_results, ols_results

def run_lagged_gravity_models(panel_data, output_dir, label):
    """Run gravity models with lagged R&D and patent variables."""
    print(f"\n{'='*60}")
    print(f"Running LAGGED models for {label} panel")
    print(f"{'='*60}")
    
    df = panel_data.copy()
    df = create_lagged_variables(df)
    
    lagged_vars = [
        "rd_intensity_gov_o_l1", "rd_intensity_gov_d_l1",
        "rd_intensity_gross_o_l1", "rd_intensity_gross_d_l1",
        "patents_o_l1", "patents_d_l1"
    ]
    
    available_lags = [v for v in lagged_vars if v in df.columns]
    
    if len(available_lags) < len(lagged_vars):
        print(f"WARNING: Some lagged variables not found. Available: {available_lags}")
    
    return run_gravity_models(df, output_dir, label + "_lagged", 
                              include_lags=True, lagged_vars=available_lags)

def run_iv_2sls(panel_data, output_dir, label='tech'):
    """
    Run 2SLS/IV estimation using lagged variables as instruments.
    Durbin-Wu-Hausman test for endogeneity.
    """
    print(f"\n{'='*60}")
    print(f"Running IV/2SLS for {label} panel")
    print(f"{'='*60}")
    
    df = panel_data.copy()
    df = create_lagged_variables(df)
    
    endogenous = {
        'rd_intensity_gross_o': 'rd_intensity_gross_o_l1',
        'rd_intensity_gross_d': 'rd_intensity_gross_d_l1',
        'rd_intensity_gov_o': 'rd_intensity_gov_o_l1',
        'rd_intensity_gov_d': 'rd_intensity_gov_d_l1',
        'patents_o': 'patents_o_l1',
        'patents_d': 'patents_d_l1',
    }
    
    required_vars = ['ln_total_value', 'ln_gdp_o', 'ln_gdp_d', 'ln_dist'] + list(endogenous.keys()) + list(endogenous.values())
    iv_data = df.dropna(subset=required_vars).copy()
    iv_data = iv_data[iv_data['ln_total_value'] > -np.inf]
    
    print(f"IV estimation sample: {len(iv_data)} observations")
    
    if len(iv_data) < 50:
        print("WARNING: Too few observations for IV estimation")
        return None, None
    
    endog_list = list(endogenous.keys())
    instruments = list(endogenous.values())
    
    # Manual 2SLS
    first_stage_results = {}
    
    for endog_var in endog_list:
        instrument = endogenous[endog_var]
        fs_formula = f"{endog_var} ~ ln_gdp_o + ln_gdp_d + ln_dist + {instrument}"
        fs_model = smf.ols(formula=fs_formula, data=iv_data)
        fs_results = fs_model.fit(cov_type='HC0')
        first_stage_results[endog_var] = {
            'coefficient': fs_results.params.get(instrument, np.nan),
            'se': fs_results.bse.get(instrument, np.nan),
            'pvalue': fs_results.pvalues.get(instrument, np.nan)
        }
        print(f"First stage for {endog_var}: coef={first_stage_results[endog_var]['coefficient']:.4f}, p={first_stage_results[endog_var]['pvalue']:.4f}")
        
        iv_data[f"{endog_var}_hat"] = fs_results.fittedvalues
    
    hat_vars = [f"{v}_hat" for v in endog_list]
    ss_formula = f"ln_total_value ~ ln_gdp_o + ln_gdp_d + ln_dist + {' + '.join(hat_vars)}"
    ss_model = smf.ols(formula=ss_formula, data=iv_data)
    iv_results = ss_model.fit(cov_type='HC0')
    
    print(iv_results.summary())
    
    with open(os.path.join(output_dir, "iv_2sls_results.txt"), "w") as f:
        f.write("Manual 2SLS Results\n")
        f.write("="*60 + "\n")
        f.write("First Stage Results:\n")
        for var, res in first_stage_results.items():
            f.write(f"{var}: coef={res['coefficient']:.4f}, se={res['se']:.4f}, p={res['pvalue']:.4f}\n")
        f.write("\nSecond Stage Results:\n")
        f.write(iv_results.summary().as_text())
    
    # Durbin-Wu-Hausman Test
    print("\n--- Durbin-Wu-Hausman Test ---")
    
    hausman_results = []
    
    for endog_var in endog_list:
        instrument = endogenous[endog_var]
        
        try:
            aux_formula = f"{endog_var} ~ ln_gdp_o + ln_gdp_d + ln_dist + {instrument}"
            aux_model = smf.ols(formula=aux_formula, data=iv_data)
            aux_results = aux_model.fit(cov_type='HC0')
            
            resid = aux_results.resid
            
            struct_formula = f"ln_total_value ~ ln_gdp_o + ln_gdp_d + ln_dist + {' + '.join(endog_list)} + resid"
            struct_data = iv_data.copy()
            struct_data['resid'] = resid
            
            struct_model = smf.ols(formula=struct_formula, data=struct_data)
            struct_results = struct_model.fit(cov_type='HC0')
            
            resid_coef = struct_results.params.get('resid', 0)
            resid_se = struct_results.bse.get('resid', np.nan)
            resid_pval = struct_results.pvalues.get('resid', np.nan)
            
            is_endogenous = resid_pval < 0.05 if not np.isnan(resid_pval) else False
            
            hausman_results.append({
                'Variable': endog_var,
                'Instrument': instrument,
                'Residual_Coef': resid_coef,
                'Residual_SE': resid_se,
                'P_Value': resid_pval,
                'Endogenous_5pct': is_endogenous
            })
            
            print(f"Hausman test for {endog_var}: coef={resid_coef:.4f}, p={resid_pval:.4f}, endogenous={is_endogenous}")
            
        except Exception as e:
            print(f"Hausman test failed for {endog_var}: {e}")
    
    hausman_df = pd.DataFrame(hausman_results)
    hausman_df.to_csv(os.path.join(output_dir, "iv_hausman_tests.csv"), index=False)
    
    with open(os.path.join(output_dir, "iv_hausman_tests.txt"), "w") as f:
        f.write("Durbin-Wu-Hausman Test Results\n")
        f.write("="*60 + "\n")
        f.write("H0: Variables are exogenous\n")
        f.write("If p-value < 0.05, reject H0 - variable is endogenous\n\n")
        f.write(hausman_df.to_string(index=False))
    
    return iv_results, hausman_results

def run_fixed_effects_models(panel_data, output_dir, label, sample_restriction=None):
    """Run gravity models with fixed effects."""
    print(f"\n{'='*60}")
    print(f"Running Fixed Effects models for {label} panel")
    print(f"{'='*60}")
    
    df = panel_data.copy()
    
    if sample_restriction:
        if 'year' in sample_restriction:
            year_min, year_max = sample_restriction['year']
            df = df[(df['year'] >= year_min) & (df['year'] <= year_max)]
            print(f"Restricted to years {year_min}-{year_max}: {len(df)} obs")
    
    if len(df) < 100:
        print("WARNING: Too few observations for FE estimation")
    
    n_origin = df['iso_o'].nunique() if 'iso_o' in df.columns else 0
    n_dest = df['iso_d'].nunique() if 'iso_d' in df.columns else 0
    n_year = df['year'].nunique() if 'year' in df.columns else 0
    
    print(f"FE dimensions: {n_origin} origins, {n_dest} destinations, {n_year} years")
    
    fe_vars = ['total_value', 'ln_gdp_o', 'ln_gdp_d']
    fe_data = df.dropna(subset=fe_vars).copy()
    
    tech_vars = [
        "rd_intensity_gov_o", "rd_intensity_gov_d",
        "rd_intensity_gross_o", "rd_intensity_gross_d",
        "patents_o", "patents_d"
    ]
    
    has_dist = 'ln_dist' in fe_data.columns
    
    print(f"\n--- PPML with Fixed Effects ---")
    
    try:
        fe_formula = f"total_value ~ C(iso_o) + C(iso_d) + C(year) + ln_gdp_o + ln_gdp_d + {' + '.join(tech_vars)}"
        if has_dist:
            fe_formula += " + ln_dist"
        
        model = smf.glm(formula=fe_formula, data=fe_data, family=sm.families.Poisson())
        ppml_fe_results = model.fit(cov_type="HC0", maxiter=100)
        print(f"PPML FE: {len(fe_data)} observations used")
        print(ppml_fe_results.summary())
        
        with open(os.path.join(output_dir, "ppml_fe_results.txt"), "w") as f:
            f.write("PPML with Fixed Effects\n")
            f.write("="*60 + "\n")
            f.write(ppml_fe_results.summary().as_text())
        
    except Exception as e:
        print(f"PPML FE failed: {e}")
        ppml_fe_results = None
    
    print(f"\n--- OLS with Fixed Effects ---")
    
    fe_data["ln_total_value"] = np.log(fe_data["total_value"].replace(0, np.nan))
    ols_fe_data = fe_data.dropna(subset=["ln_total_value"])
    
    try:
        ols_fe_formula = f"ln_total_value ~ C(iso_o) + C(iso_d) + C(year) + ln_gdp_o + ln_gdp_d + {' + '.join(tech_vars)}"
        if has_dist:
            ols_fe_formula += " + ln_dist"
        
        ols_fe_model = smf.ols(formula=ols_fe_formula, data=ols_fe_data)
        ols_fe_results = ols_fe_model.fit(cov_type="HC0")
        print(f"OLS FE: {len(ols_fe_data)} observations used")
        print(ols_fe_results.summary())
        
        with open(os.path.join(output_dir, "ols_fe_results.txt"), "w") as f:
            f.write("OLS with Fixed Effects\n")
            f.write("="*60 + "\n")
            f.write(ols_fe_results.summary().as_text())
        
    except Exception as e:
        print(f"OLS FE failed: {e}")
        ols_fe_results = None
    
    return ppml_fe_results, ols_fe_results

def run_sample_selection_analysis(panel_data, output_dir, label='tech'):
    """Analyze and address sample selection bias."""
    print(f"\n{'='*60}")
    print(f"Running Sample Selection Analysis for {label} panel")
    print(f"{'='*60}")
    
    df = panel_data.copy()
    
    # Sample Selection Diagnostics
    print("\n--- Sample Selection Diagnostics ---")
    
    df['in_ppml_sample'] = (
        df['total_value'].notna() & 
        df['ln_gdp_o'].notna() & 
        df['ln_gdp_d'].notna() &
        (df['total_value'] >= 0)
    ).astype(int)
    
    df['ln_total_value'] = np.log(df['total_value'].replace(0, np.nan))
    df['in_ols_sample'] = (
        df['ln_total_value'].notna() & 
        df['ln_gdp_o'].notna() & 
        df['ln_gdp_d'].notna()
    ).astype(int)
    
    compare_vars = ['ln_gdp_o', 'ln_gdp_d', 'ln_dist', 
                    'rd_intensity_gross_o', 'rd_intensity_gross_d',
                    'patents_o', 'patents_d']
    compare_vars = [v for v in compare_vars if v in df.columns]
    
    diagnostics = []
    
    for sample_name, sample_var in [('PPML', 'in_ppml_sample'), ('OLS', 'in_ols_sample')]:
        in_sample = df[df[sample_var] == 1]
        out_sample = df[df[sample_var] == 0]
        
        print(f"\n{sample_name} Sample: {len(in_sample)} in, {len(out_sample)} out")
        
        for var in compare_vars:
            in_mean = in_sample[var].mean()
            out_mean = out_sample[var].mean()
            in_median = in_sample[var].median()
            out_median = out_sample[var].median()
            
            diagnostics.append({
                'Sample': sample_name,
                'Variable': var,
                'In_Mean': in_mean,
                'Out_Mean': out_mean,
                'In_Median': in_median,
                'Out_Median': out_median,
                'Mean_Diff': in_mean - out_mean
            })
    
    diag_df = pd.DataFrame(diagnostics)
    diag_df.to_csv(os.path.join(output_dir, "selection_diagnostics.csv"), index=False)
    print(f"\nSaved selection diagnostics to {output_dir}/selection_diagnostics.csv")
    
    # Heckman Two-Step Selection Model
    print("\n--- Heckman Two-Step Selection Model ---")
    
    heck_data = df.copy()
    heck_data['selection'] = (heck_data['total_value'] > 0).astype(int)
    
    select_exog = ['ln_gdp_o', 'ln_gdp_d']
    if 'ln_dist' in heck_data.columns:
        select_exog.append('ln_dist')
    
    select_vars = select_exog + ['rd_intensity_gross_o', 'rd_intensity_gross_d']
    
    heck_data_select = heck_data.dropna(subset=select_vars + ['selection'])
    
    if len(heck_data_select) < 50:
        print("Insufficient data for Heckman selection")
        return diag_df, None
    
    try:
        select_formula = f"selection ~ {' + '.join(select_vars)}"
        select_model = smf.probit(formula=select_formula, data=heck_data_select)
        select_results = select_model.fit(disp=0)
        print(select_results.summary())
        
        heck_data_select['z'] = select_results.predict()
        heck_data_select['imr'] = stats.norm.pdf(heck_data_select['z']) / stats.norm.cdf(heck_data_select['z'])
        
        outcome_data = heck_data_select[heck_data_select['selection'] == 1].copy()
        outcome_data = outcome_data.dropna(subset=['ln_total_value', 'imr'])
        
        outcome_vars = select_exog + [
            'rd_intensity_gov_o', 'rd_intensity_gov_d',
            'rd_intensity_gross_o', 'rd_intensity_gross_d',
            'patents_o', 'patents_d'
        ]
        outcome_vars = [v for v in outcome_vars if v in outcome_data.columns]
        
        if len(outcome_data) < 50:
            print("Insufficient data for outcome equation")
            return diag_df, None
        
        outcome_formula = f"ln_total_value ~ {' + '.join(outcome_vars)} + imr"
        outcome_model = smf.ols(formula=outcome_formula, data=outcome_data)
        outcome_results = outcome_model.fit(cov_type='HC0')
        
        print(outcome_results.summary())
        
        with open(os.path.join(output_dir, "heckman_results.txt"), "w") as f:
            f.write("Heckman Two-Step Selection Model\n")
            f.write("="*60 + "\n\n")
            f.write("Step 1: Probit Selection Equation\n")
            f.write("-"*40 + "\n")
            f.write(select_results.summary().as_text())
            f.write("\n\nStep 2: Outcome Equation with IMR\n")
            f.write("-"*40 + "\n")
            f.write(outcome_results.summary().as_text())
            
            lambda_coef = outcome_results.params.get('imr', np.nan)
            lambda_pval = outcome_results.pvalues.get('imr', np.nan)
            
            f.write("\n\nInterpretation:\n")
            f.write("-"*40 + "\n")
            f.write(f"IMR coefficient: {lambda_coef:.4f}\n")
            f.write(f"IMR p-value: {lambda_pval:.4f}\n")
            
            if lambda_pval < 0.05:
                if lambda_coef > 0:
                    f.write("=> Positive and significant IMR suggests sample selection on unobservables\n")
                else:
                    f.write("=> Negative and significant IMR suggests negative selection\n")
            else:
                f.write("=> IMR not significant - selection may not be a major concern\n")
        
        print(f"\nHeckman IMR coefficient: {lambda_coef:.4f}, p-value: {lambda_pval:.4f}")
        
    except Exception as e:
        print(f"Heckman estimation failed: {e}")
        outcome_results = None
    
    return diag_df, outcome_results

def run_robustness_checks(panel_data, output_dir, label):
    """Run robustness checks with alternative specifications."""
    print(f"\n{'='*60}")
    print(f"Running Robustness Checks for {label} panel")
    print(f"{'='*60}")
    
    df = panel_data.copy()
    results_summary = []
    
    # Check 1: Total deals instead of value
    print("\n--- Robustness: Total Deals instead of Value ---")
    try:
        deals_data = df.dropna(subset=['total_deals', 'ln_gdp_o', 'ln_gdp_d']).copy()
        deals_data = deals_data[deals_data['total_deals'] > 0]
        
        tech_vars = [
            "rd_intensity_gov_o", "rd_intensity_gov_d",
            "rd_intensity_gross_o", "rd_intensity_gross_d",
            "patents_o", "patents_d"
        ]
        
        formula = f"total_deals ~ ln_gdp_o + ln_gdp_d + {' + '.join(tech_vars)}"
        if 'ln_dist' in deals_data.columns:
            formula += " + ln_dist"
        
        model = smf.glm(formula=formula, data=deals_data, family=sm.families.Poisson())
        results = model.fit(cov_type='HC0')
        
        print(results.summary())
        
        with open(os.path.join(output_dir, "robustness_deals.txt"), "w") as f:
            f.write("Robustness: Total Deals as DV\n")
            f.write("="*60 + "\n")
            f.write(results.summary().as_text())
        
        for var in tech_vars:
            results_summary.append({
                'Specification': 'total_deals',
                'Variable': var,
                'Coefficient': results.params.get(var, np.nan),
                'Std_Error': results.bse.get(var, np.nan),
                'P_Value': results.pvalues.get(var, np.nan)
            })
        
    except Exception as e:
        print(f"Robustness check failed: {e}")
    
    # Check 2: Only Gross R&D
    print("\n--- Robustness: Only Gross R&D ---")
    try:
        rd_vars = ["rd_intensity_gross_o", "rd_intensity_gross_d", "patents_o", "patents_d"]
        rd_data = df.dropna(subset=['total_value', 'ln_gdp_o', 'ln_gdp_d'] + rd_vars).copy()
        rd_data = rd_data[rd_data['total_value'] > 0]
        
        formula = f"total_value ~ ln_gdp_o + ln_gdp_d + {' + '.join(rd_vars)}"
        if 'ln_dist' in rd_data.columns:
            formula += " + ln_dist"
        
        model = smf.glm(formula=formula, data=rd_data, family=sm.families.Poisson())
        results = model.fit(cov_type='HC0')
        
        print(results.summary())
        
        with open(os.path.join(output_dir, "robustness_gross_rd.txt"), "w") as f:
            f.write("Robustness: Gross R&D Only\n")
            f.write("="*60 + "\n")
            f.write(results.summary().as_text())
        
        for var in rd_vars:
            results_summary.append({
                'Specification': 'gross_rd_only',
                'Variable': var,
                'Coefficient': results.params.get(var, np.nan),
                'Std_Error': results.bse.get(var, np.nan),
                'P_Value': results.pvalues.get(var, np.nan)
            })
        
    except Exception as e:
        print(f"Robustness check failed: {e}")
    
    # Check 3: Log patents
    print("\n--- Robustness: Log(1+Patents) ---")
    try:
        df['log_patents_o'] = np.log1p(df['patents_o'].fillna(0))
        df['log_patents_d'] = np.log1p(df['patents_d'].fillna(0))
        
        log_pat_vars = ["log_patents_o", "log_patents_d"]
        log_data = df.dropna(subset=['ln_total_value', 'ln_gdp_o', 'ln_gdp_d'] + log_pat_vars).copy()
        
        formula = f"ln_total_value ~ ln_gdp_o + ln_gdp_d + rd_intensity_gov_o + rd_intensity_gov_d + rd_intensity_gross_o + rd_intensity_gross_d + {' + '.join(log_pat_vars)}"
        if 'ln_dist' in log_data.columns:
            formula += " + ln_dist"
        
        model = smf.ols(formula=formula, data=log_data)
        results = model.fit(cov_type='HC0')
        
        print(results.summary())
        
        with open(os.path.join(output_dir, "robustness_log_patents.txt"), "w") as f:
            f.write("Robustness: Log(1+Patents)\n")
            f.write("="*60 + "\n")
            f.write(results.summary().as_text())
        
        for var in log_pat_vars:
            results_summary.append({
                'Specification': 'log_patents',
                'Variable': var,
                'Coefficient': results.params.get(var, np.nan),
                'Std_Error': results.bse.get(var, np.nan),
                'P_Value': results.pvalues.get(var, np.nan)
            })
        
    except Exception as e:
        print(f"Robustness check failed: {e}")
    
    summary_df = pd.DataFrame(results_summary)
    if len(summary_df) > 0:
        summary_df['Significant_5pct'] = summary_df['P_Value'] < 0.05
        summary_df.to_csv(os.path.join(output_dir, "robustness_summary.csv"), index=False)
        print(f"\nSaved robustness summary to {output_dir}/robustness_summary.csv")
    
    return summary_df

def run_mechanism_tests(panel_data, output_dir, label='tech'):
    """Mechanism tests for technology channel."""
    print(f"\n{'='*60}")
    print(f"Running Mechanism Tests for {label} panel")
    print(f"{'='*60}")
    
    df = panel_data.copy()
    
    # Test: Tech deals as dependent variable
    print("\n--- Mechanism: Tech Deals as DV ---")
    
    if 'tech_deals' in df.columns:
        tech_deals_data = df.dropna(subset=['tech_deals', 'ln_gdp_o', 'ln_gdp_d']).copy()
        
        tech_vars = [
            "rd_intensity_gov_o", "rd_intensity_gov_d",
            "rd_intensity_gross_o", "rd_intensity_gross_d",
            "patents_o", "patents_d"
        ]
        
        try:
            formula = f"tech_deals ~ ln_gdp_o + ln_gdp_d + {' + '.join(tech_vars)}"
            if 'ln_dist' in tech_deals_data.columns:
                formula += " + ln_dist"
            
            model = smf.glm(formula=formula, data=tech_deals_data, family=sm.families.Poisson())
            results = model.fit(cov_type='HC0')
            
            print(results.summary())
            
            with open(os.path.join(output_dir, "mechanism_techdeals.txt"), "w") as f:
                f.write("Mechanism Test: Tech Deals as DV\n")
                f.write("="*60 + "\n")
                f.write(results.summary().as_text())
            
        except Exception as e:
            print(f"Tech deals mechanism test failed: {e}")
    
    print("\nMechanism tests complete.")

# ================================================================================
# VISUALIZATION FUNCTIONS
# ================================================================================

def plot_rd_intensity_trends(msti_rd, output_dir):
    """Plot time series of Government vs Gross R&D intensity."""
    fig, ax = plt.subplots(figsize=(8, 5))
    msti_rd_year = msti_rd.groupby("year", as_index=False)[["rd_intensity_gov", "rd_intensity_gross"]].mean()
    msti_rd_year["rd_intensity_gov_plot"] = msti_rd_year["rd_intensity_gov"]
    sns.lineplot(data=msti_rd_year, x="year", y="rd_intensity_gov_plot", label="Government R&D (scaled, % GDP)", ax=ax)
    sns.lineplot(data=msti_rd_year, x="year", y="rd_intensity_gross", label="Gross R&D (% GDP)", ax=ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("R&D intensity (comparable units)")
    ax.set_title("Average Government vs Gross R&D Intensity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_rd_intensity_trends.png"), dpi=300)
    plt.close(fig)
    print(f"Saved: fig_rd_intensity_trends.png")

def plot_patent_trends(patents, output_dir):
    """Plot average patents per country over time."""
    yearly = patents.groupby("year", as_index=False)["patent_applications"].mean()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=yearly, x="year", y="patent_applications", marker="o", ax=ax)
    ax.set_title("Average Patent Applications Per Country (WIPO)")
    ax.set_ylabel("Patents per country")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_patent_trends.png"), dpi=300)
    plt.close()
    print("Saved: fig_patent_trends.png")

def plot_ma_vs_rd(panel, output_dir):
    """Scatter plots of M&A deals vs R&D intensity."""
    sample = panel.dropna(subset=["rd_intensity_gross_o", "ma_deals_total"]).copy()
    if len(sample) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=sample, x="rd_intensity_gross_o", y="ma_deals_total", alpha=0.3, ax=ax)
        sns.regplot(data=sample, x="rd_intensity_gross_o", y="ma_deals_total", 
                    scatter=False, line_kws={"color": "red", "lw": 2}, ax=ax, label="OLS fit")
        ax.set_xlabel("Acquirer gross R&D intensity (% of GDP)")
        ax.set_ylabel("Number of M&A deals")
        ax.set_title("Cross-border M&A vs Acquirer R&D Intensity")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig_ma_vs_rd_acquirer.png"), dpi=300)
        plt.close(fig)
        print(f"Saved: fig_ma_vs_rd_acquirer.png")
    
    sample_d = panel.dropna(subset=["rd_intensity_gross_d", "ma_deals_total"]).copy()
    if len(sample_d) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=sample_d, x="rd_intensity_gross_d", y="ma_deals_total", alpha=0.3, ax=ax)
        sns.regplot(data=sample_d, x="rd_intensity_gross_d", y="ma_deals_total", 
                    scatter=False, line_kws={"color": "red", "lw": 2}, ax=ax, label="OLS fit")
        ax.set_xlabel("Target gross R&D intensity (% of GDP)")
        ax.set_ylabel("Number of M&A deals")
        ax.set_title("Cross-border M&A vs Target R&D Intensity")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig_ma_vs_rd_target.png"), dpi=300)
        plt.close(fig)
        print(f"Saved: fig_ma_vs_rd_target.png")
    
    sample_pat_o = panel.dropna(subset=["patents_o", "total_value"]).copy()
    if len(sample_pat_o) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=sample_pat_o, x="patents_o", y="total_value", alpha=0.3, ax=ax)
        sns.regplot(data=sample_pat_o, x="patents_o", y="total_value", scatter=False, 
                    line_kws={"color": "red", "lw": 2}, ax=ax)
        ax.set_xlabel("Acquirer patent applications")
        ax.set_ylabel("Total M&A value")
        ax.set_title("M&A Value vs Acquirer Patents")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig_ma_vs_patents_acquirer.png"), dpi=300)
        plt.close()
        print("Saved: fig_ma_vs_patents_acquirer.png")
    
    sample_pat_d = panel.dropna(subset=["patents_d", "total_value"]).copy()
    if len(sample_pat_d) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=sample_pat_d, x="patents_d", y="total_value", alpha=0.3, ax=ax)
        sns.regplot(data=sample_pat_d, x="patents_d", y="total_value", scatter=False, 
                    line_kws={"color": "red", "lw": 2}, ax=ax)
        ax.set_xlabel("Target patent applications")
        ax.set_ylabel("Total M&A value")
        ax.set_title("M&A Value vs Target Patents")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig_ma_vs_patents_target.png"), dpi=300)
        plt.close()
        print("Saved: fig_ma_vs_patents_target.png")

def plot_annual_tech_nontech_value(deals, output_dir):
    """Plot annual cumulative M&A deal value by target type."""
    deals_tech = deals[deals["is_tech_target"] == 1].copy()
    deals_nontech = deals[deals["is_tech_target"] == 0].copy()
    
    yearly_tech = deals_tech.groupby("year", as_index=False)["deal_value"].sum()
    yearly_tech = yearly_tech.rename(columns={"deal_value": "total_value_tech"})
    
    yearly_nontech = deals_nontech.groupby("year", as_index=False)["deal_value"].sum()
    yearly_nontech = yearly_nontech.rename(columns={"deal_value": "total_value_nontech"})
    
    yearly = yearly_tech.merge(yearly_nontech, on="year", how="outer").fillna(0)
    
    yearly_deals = deals.groupby("year", as_index=False).size()
    yearly_deals = yearly_deals.rename(columns={"size": "num_deals"})
    
    yearly = yearly.merge(yearly_deals, on="year", how="left").fillna({"num_deals": 0})

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = yearly["year"]
    width = 0.8

    ax1.bar(x, yearly["total_value_nontech"], width, label="Non-tech targets (value)", color="#b0c4de")
    ax1.bar(x, yearly["total_value_tech"], width, bottom=yearly["total_value_nontech"],
            label="Tech targets (value)", color="#1f77b4")

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total M&A deal value (USD)")
    ax1.set_title("Annual M&A Deal Value: Tech vs Non-Tech Targets, with Deal Counts")

    ax2 = ax1.twinx()
    ax2.plot(x, yearly["num_deals"], color="black", marker="o", linestyle="-", label="Number of deals")
    ax2.set_ylabel("Number of deals")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fname = os.path.join(output_dir, "fig_annual_tech_nontech_value.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"Saved: {fname}")

def plot_coefficients(results, output_dir, vars_of_interest=None):
    """Plot coefficient estimates with confidence intervals."""
    if vars_of_interest is None:
        vars_of_interest = ["ln_gdp_o", "ln_gdp_d", "rd_intensity_gov_o", 
                            "rd_intensity_gov_d", "rd_intensity_gross_o", "rd_intensity_gross_d",
                            "patents_o", "patents_d"]

    if results is None:
        print("WARNING: No results provided - skipping plot.")
        return
    
    params = results.params
    conf = results.conf_int()
    conf.columns = ['ci_low', 'ci_high']
    coef_df = pd.concat([params.rename('coef'), conf], axis=1).reset_index().rename(columns={"index": "variable"})
    coef_df = coef_df[coef_df["variable"].isin(vars_of_interest)].copy()
    
    if len(coef_df) == 0:
        print("ERROR: No matching variables in results!")
        return
    
    coef_df = coef_df.sort_values('coef')
    y_pos = np.arange(len(coef_df))
    left_err = coef_df["coef"] - coef_df["ci_low"]
    right_err = coef_df["ci_high"] - coef_df["coef"]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.errorbar(coef_df["coef"], y_pos, xerr=[left_err, right_err], fmt="o", color="black", 
                ecolor="gray", capsize=5, markersize=8)
    ax.axvline(0, color="red", linestyle="--", lw=2, label="No effect")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_df["variable"])
    ax.set_xlabel("Coefficient (PPML)")
    ax.set_title("Gravity: Tech/R&D Effects on M&A")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_gravity_coefficients.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot with {len(coef_df)} vars.")

# ================================================================================
# COMPILATION FUNCTIONS
# ================================================================================

def compile_all_tables(output_dir):
    """
    Compile all coefficient tables from the model runs into a single well-formatted file.
    
    This function reads all output files from the regression pipeline and creates:
    1. ALL_RESULTS_TABLES.txt - formatted text file with all tables
    2. ALL_RESULTS_TABLES.csv - CSV with status tracking
    3. MASTER_COEFFICIENT_TABLE.csv - coefficient comparison across models
    """
    from datetime import datetime
    
    print("\n" + "="*60)
    print("COMPILING ALL RESULTS TABLES")
    print("="*60)
    
    # Define all tables to compile (table number, name, file paths)
    # Note: The pipeline runs baseline first, then lagged models overwrite the same files
    tables_config = [
        # 1-2: BASELINE - Full Panel (these files contain LAGGED results - noted in output)
        (1, "BASELINE PPML - Full Panel", [os.path.join(output_dir, "ppml_results.txt")]),
        (2, "BASELINE OLS - Full Panel", [os.path.join(output_dir, "ols_results.txt")]),
        # 3-4: BASELINE - Tech Panel
        (3, "BASELINE PPML - Tech Panel", [os.path.join(output_dir, "tech", "ppml_results.txt")]),
        (4, "BASELINE OLS - Tech Panel", [os.path.join(output_dir, "tech", "ols_results.txt")]),
        # 5-6: BASELINE - Non-Tech Panel
        (5, "BASELINE PPML - Non-Tech Panel", [os.path.join(output_dir, "nontech", "ppml_results.txt")]),
        (6, "BASELINE OLS - Non-Tech Panel", [os.path.join(output_dir, "nontech", "ols_results.txt")]),
        # 7-12: LAGGED - Same files as baseline (lagged overwrites baseline in pipeline)
        (7, "LAGGED PPML - Full Panel", [os.path.join(output_dir, "ppml_results.txt")]),
        (8, "LAGGED OLS - Full Panel", [os.path.join(output_dir, "ols_results.txt")]),
        (9, "LAGGED PPML - Tech Panel", [os.path.join(output_dir, "tech", "ppml_results.txt")]),
        (10, "LAGGED OLS - Tech Panel", [os.path.join(output_dir, "tech", "ols_results.txt")]),
        (11, "LAGGED PPML - Non-Tech Panel", [os.path.join(output_dir, "nontech", "ppml_results.txt")]),
        (12, "LAGGED OLS - Non-Tech Panel", [os.path.join(output_dir, "nontech", "ols_results.txt")]),
        # 13: IV/2SLS
        (13, "IV/2SLS - Tech Panel (second stage)", [os.path.join(output_dir, "tech", "iv_2sls_results.txt")]),
        # 14: Durbin-Wu-Hausman
        (14, "DURBIN-WU-HAUSMAN TEST - Tech Panel", [os.path.join(output_dir, "tech", "iv_hausman_tests.csv")]),
        # 15: Fixed Effects
        (15, "FIXED EFFECTS OLS - Tech Panel", [os.path.join(output_dir, "tech", "ols_fe_results.txt")]),
        # 16-17: Heckman
        (16, "HECKMAN Step 1 (Probit Selection Equation) - Tech Panel", [os.path.join(output_dir, "tech", "heckman_results.txt")]),
        (17, "HECKMAN Step 2 (Outcome Equation with IMR) - Tech Panel", [os.path.join(output_dir, "tech", "heckman_results.txt")]),
        # 18-20: Robustness
        (18, "ROBUSTNESS: Total Deals as DV - Tech Panel", [os.path.join(output_dir, "tech", "robustness_deals.txt")]),
        (19, "ROBUSTNESS: Gross R&D Only - Tech Panel", [os.path.join(output_dir, "tech", "robustness_gross_rd.txt")]),
        (20, "ROBUSTNESS: Log(1+Patents) - Tech Panel", [os.path.join(output_dir, "tech", "robustness_log_patents.txt")]),
        # 21: Mechanism Test
        (21, "MECHANISM TEST: Tech Deals as DV - Full Panel", [os.path.join(output_dir, "mechanism_techdeals.txt")]),
        # 22-24: VIF Diagnostics
        (22, "VIF DIAGNOSTICS - Full Panel OLS", [os.path.join(output_dir, "vif_results.csv")]),
        (23, "VIF DIAGNOSTICS - Tech Panel OLS", [os.path.join(output_dir, "tech", "vif_results.csv")]),
        (24, "VIF DIAGNOSTICS - Non-Tech Panel OLS", [os.path.join(output_dir, "nontech", "vif_results.csv")]),
        # 25: Heteroskedasticity
        (25, "HETEROSKEDASTICITY TESTS - All Panels", [
            os.path.join(output_dir, "heteroskedasticity_tests.csv"),
            os.path.join(output_dir, "tech", "heteroskedasticity_tests.csv"),
            os.path.join(output_dir, "nontech", "heteroskedasticity_tests.csv")
        ]),
        # 26: Selection Diagnostics
        (26, "SELECTION DIAGNOSTICS - Tech Panel", [os.path.join(output_dir, "tech", "selection_diagnostics.csv")]),
    ]
    
    # Track results
    table_results = []
    successful_count = 0
    missing_tables = []
    obs_counts = {}
    
    # Initialize output content
    output_lines = []
    
    # Cover page
    cover_page = """================================================================================
ALL ESTIMATION RESULTS - COMPILED OUTPUT
Thesis: The Global M&A Market in the Era of Technological Transformation
Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
Output directory: """ + output_dir + """
Total tables attempted: 26
================================================================================

NOTE: In this pipeline, lagged models overwrite baseline results in the same 
output files. Tables 1-6 (BASELINE) and 7-12 (LAGGED) share the same files.
The files contain the lagged specification results.

"""
    output_lines.append(cover_page)
    
    # Process each table
    for table_num, table_name, file_paths in tables_config:
        # Find first existing file
        content = None
        file_used = None
        for fp in file_paths:
            if os.path.exists(fp):
                try:
                    if fp.endswith('.csv'):
                        df = pd.read_csv(fp)
                        content = df.to_string(index=False)
                    else:
                        with open(fp, 'r', encoding='utf-8') as f:
                            content = f.read()
                    file_used = fp
                    break
                except Exception as e:
                    print(f"Error reading {fp}: {e}")
                    continue
        
        # Extract observation count if available
        obs_count = "N/A"
        if content:
            # Try to extract observation count from text
            obs_match = re.search(r'No\. Observations:\s*(\d+)', content)
            if obs_match:
                obs_count = obs_match.group(1)
                obs_counts[table_name] = obs_count
        
        # Format table section
        section_header = "\n" + "="*80 + "\nTABLE " + str(table_num) + ": " + table_name + "\n" + "="*80 + "\n\n"
        
        if content:
            table_results.append({
                'TableNumber': table_num,
                'TableName': table_name,
                'Status': 'OK',
                'ObsCount': obs_count
            })
            successful_count += 1
            output_lines.append(section_header)
            output_lines.append(content)
            output_lines.append("\n")
        else:
            missing_tables.append(table_name)
            table_results.append({
                'TableNumber': table_num,
                'TableName': table_name,
                'Status': 'MISSING',
                'ObsCount': obs_count
            })
            output_lines.append(section_header)
            output_lines.append("[TABLE NOT AVAILABLE - model did not converge or file missing]\n")
    
    # Summary Statistics section
    summary_section = """
================================================================================
SUMMARY STATISTICS
================================================================================

Total tables attempted: 26
Tables successfully loaded: """ + str(successful_count) + """
Tables missing: """ + str(len(missing_tables)) + """

"""
    
    if missing_tables:
        summary_section += "Missing tables:\n"
        for mt in missing_tables:
            summary_section += "  - " + mt + "\n"
    else:
        summary_section += "All tables were successfully loaded.\n"
    
    # Observation counts from baseline models
    summary_section += "\n" + "="*80 + "\n"
    summary_section += "OBSERVATION COUNTS FROM BASELINE MODELS\n"
    summary_section += "="*80 + "\n\n"
    
    baseline_obs = {
        "BASELINE PPML - Full Panel": obs_counts.get("BASELINE PPML - Full Panel", "N/A"),
        "BASELINE OLS - Full Panel": obs_counts.get("BASELINE OLS - Full Panel", "N/A"),
        "BASELINE PPML - Tech Panel": obs_counts.get("BASELINE PPML - Tech Panel", "N/A"),
        "BASELINE OLS - Tech Panel": obs_counts.get("BASELINE OLS - Tech Panel", "N/A"),
        "BASELINE PPML - Non-Tech Panel": obs_counts.get("BASELINE PPML - Non-Tech Panel", "N/A"),
        "BASELINE OLS - Non-Tech Panel": obs_counts.get("BASELINE OLS - Non-Tech Panel", "N/A"),
    }
    
    for name, obs in baseline_obs.items():
        summary_section += f"{name}: {obs} observations\n"
    
    output_lines.append(summary_section)
    
    # ========================================================================
    # MASTER COEFFICIENT COMPARISON TABLE
    # ========================================================================
    
    # Map file variables to standard names
    var_mapping = {
        'ln_gdp_o': 'lngdp_o',
        'ln_gdp_d': 'lngdp_d',
        'ln_dist': 'lndist',
        'rd_intensity_gov_o': 'rdgov_o',
        'rd_intensity_gov_o_l1': 'rdgov_o',
        'rd_intensity_gov_d': 'rdgov_d',
        'rd_intensity_gov_d_l1': 'rdgov_d',
        'rd_intensity_gross_o': 'rdgross_o',
        'rd_intensity_gross_o_l1': 'rdgross_o',
        'rd_intensity_gross_d': 'rdgross_d',
        'rd_intensity_gross_d_l1': 'rdgross_d',
        'patents_o': 'patents_o',
        'patents_o_l1': 'patents_o',
        'patents_d': 'patents_d',
        'patents_d_l1': 'patents_d',
    }
    
    # Extract coefficients from result files
    def extract_coefs(file_path):
        coefs = {}
        if not os.path.exists(file_path):
            return coefs
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            for line in lines:
                if line.startswith('=') or line.startswith('-') or line.startswith(' ') or not line.strip():
                    continue
                match = re.match(r'^(\S+)\s+([\-\+\d\.eE\+]+)\s+([\-\+\d\.eE\+]+)\s+([\-\+\d\.eE\+]+)\s+([\-\+\d\.eE\+]+)', line.strip())
                if match:
                    var_name = match.group(1)
                    coef = match.group(2)
                    pval = match.group(5)
                    
                    std_name = var_mapping.get(var_name)
                    if std_name:
                        coefs[std_name] = {'coef': coef, 'pval': pval, 'var': var_name}
        except Exception as e:
            print(f"Error extracting coefs from {file_path}: {e}")
        
        return coefs
    
    # Build master coefficient table
    master_coefs = []
    
    specs = [
        ("PPML Full Lagged", os.path.join(output_dir, "ppml_results.txt"), True),
        ("PPML Tech Lagged", os.path.join(output_dir, "tech", "ppml_results.txt"), True),
        ("PPML Non-Tech Lagged", os.path.join(output_dir, "nontech", "ppml_results.txt"), True),
        ("OLS Full Lagged", os.path.join(output_dir, "ols_results.txt"), True),
        ("OLS Tech Lagged", os.path.join(output_dir, "tech", "ols_results.txt"), True),
        ("OLS Non-Tech Lagged", os.path.join(output_dir, "nontech", "ols_results.txt"), True),
        ("OLS Tech FE", os.path.join(output_dir, "tech", "ols_fe_results.txt"), False),
        ("IV/2SLS Tech", os.path.join(output_dir, "tech", "iv_2sls_results.txt"), False),
    ]
    
    for spec_name, file_path, expect_lagged in specs:
        coefs = extract_coefs(file_path)
        
        row = {'Specification': spec_name}
        for var in ['lngdp_o', 'lngdp_d', 'lndist', 'rdgov_o', 'rdgov_d', 'rdgross_o', 'rdgross_d', 'patents_o', 'patents_d']:
            if var in coefs:
                c = coefs[var]['coef']
                p = coefs[var]['pval']
                try:
                    pval = float(p)
                    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                except:
                    stars = ''
                row[var] = f"{c}({stars})"
            else:
                row[var] = "N/A"
        master_coefs.append(row)
    
    # Create master coefficient comparison section
    master_section = "\n" + "="*80 + "\n"
    master_section += "MASTER COEFFICIENT COMPARISON\n"
    master_section += "="*80 + "\n\n"
    master_section += "Note: The pipeline overwrites baseline files with lagged results.\n"
    master_section += "Variables: lngdp_o, lngdp_d, lndist, rdgov_o, rdgov_d, rdgross_o, rdgross_d, patents_o, patents_d\n"
    master_section += "(Coefficient with p-value stars: ***p<0.01, **p<0.05, *p<0.10)\n\n"
    
    header = f"{'Specification':<22} | {'lngdp_o':<12} | {'lngdp_d':<12} | {'lndist':<12} | {'rdgov_o':<12} | {'rdgov_d':<12} | {'rdgross_o':<12} | {'rdgross_d':<12} | {'patents_o':<12} | {'patents_d':<12}\n"
    master_section += header
    master_section += "-" * 230 + "\n"
    
    for row in master_coefs:
        line = f"{row['Specification']:<22} | {row['lngdp_o']:<12} | {row['lngdp_d']:<12} | {row['lndist']:<12} | {row['rdgov_o']:<12} | {row['rdgov_d']:<12} | {row['rdgross_o']:<12} | {row['rdgross_d']:<12} | {row['patents_o']:<12} | {row['patents_d']:<12}\n"
        master_section += line
    
    output_lines.append(master_section)
    
    # Write ALL_RESULTS_TABLES.txt
    txt_output_path = os.path.join(output_dir, "ALL_RESULTS_TABLES.txt")
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(output_lines))
    print(f"Created: {txt_output_path}")
    
    # Write ALL_RESULTS_TABLES.csv
    csv_df = pd.DataFrame(table_results)
    csv_output_path = os.path.join(output_dir, "ALL_RESULTS_TABLES.csv")
    csv_df.to_csv(csv_output_path, index=False)
    print(f"Created: {csv_output_path}")
    
    # Write MASTER_COEFFICIENT_TABLE.csv
    master_csv_data = []
    for spec_name, file_path, expect_lagged in specs:
        coefs = extract_coefs(file_path)
        row = {'Specification': spec_name}
        for var in ['lngdp_o', 'lngdp_d', 'lndist', 'rdgov_o', 'rdgov_d', 'rdgross_o', 'rdgross_d', 'patents_o', 'patents_d']:
            if var in coefs:
                c = coefs[var]['coef']
                p = coefs[var]['pval']
                row[f'{var}_coef'] = c
                row[f'{var}_pval'] = p
            else:
                row[f'{var}_coef'] = ''
                row[f'{var}_pval'] = ''
        master_csv_data.append(row)
    
    master_csv_path = os.path.join(output_dir, "MASTER_COEFFICIENT_TABLE.csv")
    master_df = pd.DataFrame(master_csv_data)
    master_df.to_csv(master_csv_path, index=False)
    print(f"Created: {master_csv_path}")
    
    print(f"\nCompilation complete!")
    print(f"  - {successful_count}/26 tables loaded successfully")
    print(f"  - {len(missing_tables)} tables missing")
    
    return table_results, master_coefs




# MAIN EXECUTION
# ================================================================================

if __name__ == "__main__":
    import atexit
    # runs on normal exit AND after an uncaught exception, so the last lines of
    # the log always say whether the export is complete
    atexit.register(validate_exports)
    print("Loading M&A data...")
    deals = parse_mna_xml(MNA_XML_FILE)
    print(f"Loaded {len(deals)} deals from XML.")
    print("Deal columns:", deals.columns.tolist())
    # v14: the complete raw extract, every record and every field, before any
    # filter. Everything downstream can be rebuilt from this file.
    export_df(deals, "deals_raw_full.csv.gz")
    if len(deals) == 0:
        print("ERROR: No deals loaded.")
        exit(1)
    if 'COMPLETION_DATE' in deals.columns:
        deals['year'] = pd.to_numeric(deals['COMPLETION_DATE'].str[:4], errors='coerce')
    else:
        print("ERROR: No COMPLETION_DATE column found!")
        exit(1)
    deals = deals[deals['year'].notna()].copy()
    deals['year'] = deals['year'].astype(int)
    
    # Map acquirer country (iso_o) and target country (iso_d)
    if "BIDDER_CTRYC" in deals.columns:
        deals["iso_o"] = deals["BIDDER_CTRYC"].str.upper().str.strip().map(ISO2_TO_ISO3)
    elif "ACQUIROR_CTRYC" in deals.columns:
        deals["iso_o"] = deals["ACQUIROR_CTRYC"].str.upper().str.strip().map(ISO2_TO_ISO3)
    elif "ACQUIROR_CTRY_ISO" in deals.columns:
        deals["iso_o"] = deals["ACQUIROR_CTRY_ISO"].str.upper().str.strip().map(ISO2_TO_ISO3)
    if "TARGETCTRYC" in deals.columns:
        deals["iso_d"] = deals["TARGETCTRYC"].str.upper().str.strip().map(ISO2_TO_ISO3)
    elif "TARGET_CTRYC" in deals.columns:
        deals["iso_d"] = deals["TARGET_CTRYC"].str.upper().str.strip().map(ISO2_TO_ISO3)
    elif "TARGET_CTRY_ISO" in deals.columns:
        deals["iso_d"] = deals["TARGET_CTRY_ISO"].str.upper().str.strip().map(ISO2_TO_ISO3)
    if "iso_o" in deals.columns and "iso_d" in deals.columns:
        print(f"Unique iso_o: {deals['iso_o'].nunique()}, iso_d: {deals['iso_d'].nunique()}")
    deals = deals[deals["iso_o"].notna() & deals["iso_d"].notna()].copy()
    print(f"Deals with valid country codes: {len(deals)}")

    # ==========================================================================
    # revision: restrict to genuinely CROSS-BORDER deals.
    # The Orbis extract contains records whose acquirer and target country are the
    # same (CHN->CHN, USA->USA, ...). They accounted for 5.05% of the estimation
    # sample, 9.07% of deal value and 10.15% of deal count. Domestic pairs carry
    # CEPII internal distance (near zero) together with unusually large values,
    # which pulls the distance elasticity toward zero: on the full panel the
    # coefficient moves from -0.259 to -0.377 once they are removed.
    # ==========================================================================
    _n_before = len(deals)
    deals = deals[deals["iso_o"] != deals["iso_d"]].copy()
    print(f"Dropped {_n_before - len(deals)} same-country (domestic) records; "
          f"{len(deals)} cross-border deals remain")
    
    # Define deal_value from ALLDLVALUE or TRGTVAL
    if "ALLDLVALUE" in deals.columns:
        deals["deal_value"] = pd.to_numeric(deals["ALLDLVALUE"], errors="coerce")
        print("Using ALLDLVALUE for deal_value")
    elif "TRGTVAL" in deals.columns:
        deals["deal_value"] = pd.to_numeric(deals["TRGTVAL"], errors="coerce")
        print("Using TRGTVAL for deal_value")
    else:
        print("WARNING: No ALLDLVALUE or TRGTVAL column found for deal values.")
        deals["deal_value"] = np.nan
    
    print("Creating sector dummies...")
    deals = create_sector_dummies(deals)
    sector_agg = aggregate_sector_dummies(deals)
    print(f"Aggregated to {len(sector_agg)} dyad-year observations.")
    
    print("Loading R&D intensity data...")
    msti_rd = load_rd_intensities()
    export_df(msti_rd, "msti_loaded.csv")
    print(f"Loaded R&D data: {len(msti_rd)} observations.")
    
    patents = load_patents(PATENTS_FILE)
    export_df(patents, "patents_loaded.csv")

    print("\n--- R&D Intensity Diagnostics ---")
    print("RD intensity summary:")
    print(msti_rd[["rd_intensity_gov", "rd_intensity_gross"]].describe())
    print("--- End R&D Diagnostics ---\n")
    
    # Identify tech sector columns
    sector_target_cols_deal = [c for c in deals.columns if c.startswith("target_sic_")]
    deals["is_tech_target"] = (deals[sector_target_cols_deal].sum(axis=1) > 0).astype(int)
    # revision: acquirer-side classification, for the robustness check the
    # supervisor requested (a technology acquirer buying a non-technology target
    # lands in the non-technology panel under target-side classification)
    sector_bidder_cols_deal = [c for c in deals.columns if c.startswith("bidder_sic_")]
    deals["is_tech_acquirer"] = (deals[sector_bidder_cols_deal].sum(axis=1) > 0).astype(int)
    print(f"Tech-acquirer deals: {deals['is_tech_acquirer'].sum()}  "
          f"| both sides tech: {((deals['is_tech_target']==1)&(deals['is_tech_acquirer']==1)).sum()}")
    # v14: written here, before any estimation, so that a failure later in the
    # pipeline cannot cost this file. Raw SIC strings travel with it so the
    # sector definition can be varied without another run.
    _dm_cols = ["iso_o", "iso_d", "year", "deal_value", "is_tech_target", "is_tech_acquirer"]
    _dm_extra = [c for c in ("RECORD_ID", "TARGET_SCUSCD", "BIDDER_SCUSCD", "BIDDER_STRATEGIC_CTRYC",
                             "STAKE", "FINLSTAK", "DEALTYPE") if c in deals.columns]
    export_df(deals[_dm_cols + _dm_extra], "deals_min.csv")
    try:
        deals[_dm_cols].to_csv(os.path.join(DATA_DIR, "output", "deals_min.csv"), index=False)
    except Exception as _e:
        print(f"[EXPORT] legacy copy of deals_min.csv not written: {_e}")
    
    if "deal_value" not in deals.columns:
        deals["deal_value"] = np.nan
    deals["deal_value"] = pd.to_numeric(deals["deal_value"], errors="coerce")
    
    print(f"Tech-target deals: {deals['is_tech_target'].sum()}")
    print(f"Non-tech-target deals: {(deals['is_tech_target'] == 0).sum()}")
    
    # Aggregate full panel (all deals)
    panel_base = deals.groupby(["iso_o", "iso_d", "year"], as_index=False).agg(
        total_deals=("is_tech_target", "size"),
        tech_deals=("is_tech_target", "sum"),
        total_value=("deal_value", "sum"),
    )
    
    sector_cols_to_keep = [c for c in sector_agg.columns if c.startswith("target_sic_") or c.startswith("bidder_sic_")]
    sector = sector_agg[["iso_o", "iso_d", "year"] + sector_cols_to_keep]
    
    # Build main panel
    panel = panel_base.merge(sector, on=["iso_o", "iso_d", "year"], how="left")
    if "ma_deals_total" not in panel.columns:
        deal_counts = deals.groupby(["iso_o", "iso_d", "year"], as_index=False).size()
        deal_counts = deal_counts.rename(columns={"size": "ma_deals_total"})
        panel = panel.merge(deal_counts, on=["iso_o", "iso_d", "year"], how="left")
    panel = panel.merge(msti_rd.rename(columns={"iso3": "iso_o", "rd_intensity_gov": "rd_intensity_gov_o", "rd_intensity_gross": "rd_intensity_gross_o"}), on=["iso_o", "year"], how="left")
    panel = panel.merge(msti_rd.rename(columns={"iso3": "iso_d", "rd_intensity_gov": "rd_intensity_gov_d", "rd_intensity_gross": "rd_intensity_gross_d"}), on=["iso_d", "year"], how="left")
    
    panel = panel.merge(
        patents.rename(columns={"iso3": "iso_o", "patent_applications": "patents_o"}),
        on=["iso_o", "year"],
        how="left"
    )
    
    panel = panel.merge(
        patents.rename(columns={"iso3": "iso_d", "patent_applications": "patents_d"}),
        on=["iso_d", "year"],
        how="left"
    )
    
    for col in ["rd_intensity_gov_o", "rd_intensity_gov_d", "rd_intensity_gross_o", "rd_intensity_gross_d", "patents_o", "patents_d"]:
        if col in panel.columns:
            panel[col] = panel[col].clip(lower=0)
    
    print(f"Panel after R&D merge: {len(panel)} observations.")
    _biv = [c for c in ("contig", "comlang_off", "colony", "comcol", "rta",
                        "wto_o", "wto_d", "eu_o", "eu_d") if c in panel.columns]
    print(f"Bilateral controls available from CEPII: {_biv}")
    print("Patent coverage:", panel[["patents_o", "patents_d"]].notna().sum())

    # For PPML: missing totalvalue should become 0 (true zero flow in gravity framework)
    # This is econometrically correct - PPML handles zeros natively
    panel["total_value"] = panel["total_value"].fillna(0)
    
    # Load CEPII gravity data using the new function
    try:
        gravity = load_gravity_data()
        print(f"Loaded CEPII gravity data: {len(gravity)} rows")
        export_df(gravity, "gravity_cepii_2000_2024.csv.gz")
        validate_exports()
        
        # Merge gravity data - rename columns for merge
        panel = panel.merge(
            gravity.rename(columns={"isoo": "iso_o", "isod": "iso_d"}),
            on=["iso_o", "iso_d", "year"],
            how="left"
        )
        
        # Create log-transformed gravity variables
        panel["ln_gdp_o"] = np.log(panel["gdp_o"])
        panel["ln_gdp_d"] = np.log(panel["gdp_d"])
        panel["ln_dist"] = np.log(panel["dist"])
        
        # Check merge success
        n_matched = panel["ln_gdp_o"].notna().sum()
        print(f"CEPII merge: {n_matched} matched rows out of {len(panel)}")
        
        if n_matched < 1000:
            raise RuntimeError(
                "CEPII gravity data did not merge properly. Check column names and ISO3 codes in GravityV202211.csv. "
                f"Only {n_matched} rows matched."
            )
        
        # Validate that gravity variables have meaningful variation
        for var in ["ln_gdp_o", "ln_gdp_d", "ln_dist"]:
            if var in panel.columns:
                std_val = panel[var].std()
                if std_val < 0.01:
                    raise RuntimeError(
                        f"Variable {var} has std = {std_val:.6f} < 0.01. "
                        "Check CEPII data quality."
                    )
                print(f"{var}: mean={panel[var].mean():.4f}, std={std_val:.4f}")
        
        CEPII_USED = True
        
    except RuntimeError as e:
        print(f"ERROR: {e}")
        CEPII_USED = False
        raise

    # Create output directories
    output_dir = os.path.join(DATA_DIR, "output")
    output_dir_tech = os.path.join(DATA_DIR, "output", "tech")
    output_dir_nontech = os.path.join(DATA_DIR, "output", "nontech")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_tech, exist_ok=True)
    os.makedirs(output_dir_nontech, exist_ok=True)

    # Create separate tech and non-tech panels
    print("\n=== Creating separate tech and non-tech panels ===")
    
    deals_tech = deals[deals["is_tech_target"] == 1].copy()
    panel_tech = deals_tech.groupby(["iso_o", "iso_d", "year"], as_index=False).agg(
        total_deals=("is_tech_target", "size"),
        total_value=("deal_value", "sum"),
    )
    panel_tech["ma_deals_total"] = panel_tech["total_deals"]
    
    deals_nontech = deals[deals["is_tech_target"] == 0].copy()
    panel_nontech = deals_nontech.groupby(["iso_o", "iso_d", "year"], as_index=False).agg(
        total_deals=("is_tech_target", "size"),
        total_value=("deal_value", "sum"),
    )
    panel_nontech["ma_deals_total"] = panel_nontech["total_deals"]
    
    # Merge sector dummies and variables to tech panel
    panel_tech = panel_tech.merge(sector, on=["iso_o", "iso_d", "year"], how="left")
    panel_tech = panel_tech.merge(msti_rd.rename(columns={"iso3": "iso_o", "rd_intensity_gov": "rd_intensity_gov_o", "rd_intensity_gross": "rd_intensity_gross_o"}), on=["iso_o", "year"], how="left")
    panel_tech = panel_tech.merge(msti_rd.rename(columns={"iso3": "iso_d", "rd_intensity_gov": "rd_intensity_gov_d", "rd_intensity_gross": "rd_intensity_gross_d"}), on=["iso_d", "year"], how="left")
    panel_tech = panel_tech.merge(patents.rename(columns={"iso3": "iso_o", "patent_applications": "patents_o"}), on=["iso_o", "year"], how="left")
    panel_tech = panel_tech.merge(patents.rename(columns={"iso3": "iso_d", "patent_applications": "patents_d"}), on=["iso_d", "year"], how="left")
    
    # Merge sector dummies and variables to non-tech panel
    panel_nontech = panel_nontech.merge(sector, on=["iso_o", "iso_d", "year"], how="left")
    panel_nontech = panel_nontech.merge(msti_rd.rename(columns={"iso3": "iso_o", "rd_intensity_gov": "rd_intensity_gov_o", "rd_intensity_gross": "rd_intensity_gross_o"}), on=["iso_o", "year"], how="left")
    panel_nontech = panel_nontech.merge(msti_rd.rename(columns={"iso3": "iso_d", "rd_intensity_gov": "rd_intensity_gov_d", "rd_intensity_gross": "rd_intensity_gross_d"}), on=["iso_d", "year"], how="left")
    panel_nontech = panel_nontech.merge(patents.rename(columns={"iso3": "iso_o", "patent_applications": "patents_o"}), on=["iso_o", "year"], how="left")
    panel_nontech = panel_nontech.merge(patents.rename(columns={"iso3": "iso_d", "patent_applications": "patents_d"}), on=["iso_d", "year"], how="left")
    
    # Add gravity data
    if "ln_gdp_o" in panel.columns:
        _grav_cols = ["iso_o", "iso_d", "year", "ln_gdp_o", "ln_gdp_d", "ln_dist"]
        _grav_cols += [c for c in ("pop_o", "pop_d", "contig", "comlang_off", "colony",
                                   "comcol", "rta", "wto_o", "wto_d", "eu_o", "eu_d")
                       if c in panel.columns]
        panel_tech = panel_tech.merge(panel[_grav_cols], on=["iso_o", "iso_d", "year"], how="left")
        panel_nontech = panel_nontech.merge(panel[_grav_cols], on=["iso_o", "iso_d", "year"], how="left")
    
    for col in ["rd_intensity_gov_o", "rd_intensity_gov_d", "rd_intensity_gross_o", "rd_intensity_gross_d", "patents_o", "patents_d"]:
        if col in panel_tech.columns:
            panel_tech[col] = panel_tech[col].clip(lower=0)
        if col in panel_nontech.columns:
            panel_nontech[col] = panel_nontech[col].clip(lower=0)
    
    # For PPML: missing totalvalue becomes 0 (true zero flow in gravity framework)
    panel_tech["total_value"] = panel_tech["total_value"].fillna(0)
    panel_nontech["total_value"] = panel_nontech["total_value"].fillna(0)
    
    print(f"Tech panel: {len(panel_tech)} dyad-year observations")
    print(f"Non-tech panel: {len(panel_nontech)} dyad-year observations")

    # ============================================
    # 1. BASELINE MODELS (PPML and OLS)
    # ============================================
    print("\n" + "="*60)
    print("1. BASELINE MODELS")
    print("="*60)
    
    ppml_all, ols_all = run_gravity_models(panel, output_dir, "all")
    ppml_tech, ols_tech = run_gravity_models(panel_tech, output_dir_tech, "tech")
    ppml_nontech, ols_nontech = run_gravity_models(panel_nontech, output_dir_nontech, "non-tech")
    
    # ============================================
    # 2. ENDOGENEITY TESTS - LAGGED VARIABLES
    # ============================================
    print("\n" + "="*60)
    print("2. ENDOGENEITY TESTS - LAGGED VARIABLES")
    print("="*60)
    
    run_lagged_gravity_models(panel, output_dir, "all")
    run_lagged_gravity_models(panel_tech, output_dir_tech, "tech")
    run_lagged_gravity_models(panel_nontech, output_dir_nontech, "non-tech")
    
    # ============================================
    # 3. IV/2SLS ESTIMATION (Tech panel only)
    # ============================================
    print("\n" + "="*60)
    print("3. IV/2SLS ESTIMATION")
    print("="*60)
    
    panel_tech["ln_total_value"] = np.log(panel_tech["total_value"].replace(0, np.nan))
    run_iv_2sls(panel_tech, output_dir_tech, "tech")
    
    # ============================================
    # 4. FIXED EFFECTS
    # ============================================
    print("\n" + "="*60)
    print("4. FIXED EFFECTS MODELS")
    print("="*60)
    
    run_fixed_effects_models(panel_tech, output_dir_tech, "tech", sample_restriction={'year': (2005, 2015)})
    
    # ============================================
    # 5. SAMPLE SELECTION ANALYSIS
    # ============================================
    print("\n" + "="*60)
    print("5. SAMPLE SELECTION ANALYSIS")
    print("="*60)
    
    run_sample_selection_analysis(panel_tech, output_dir_tech, "tech")
    
    # ============================================
    # 6. ROBUSTNESS CHECKS
    # ============================================
    print("\n" + "="*60)
    print("6. ROBUSTNESS CHECKS")
    print("="*60)
    
    run_robustness_checks(panel_tech, output_dir_tech, "tech")
    
    # ============================================
    # 7. MECHANISM TESTS
    # ============================================
    print("\n" + "="*60)
    print("7. MECHANISM TESTS")
    print("="*60)
    
    run_mechanism_tests(panel, output_dir, "all")
    
    # ============================================
    # Generate visualizations
    # ============================================
    print("\nGenerating visualizations...")
    plot_rd_intensity_trends(msti_rd, output_dir)
    plot_annual_tech_nontech_value(deals, output_dir)
    plot_patent_trends(patents, output_dir)
    plot_ma_vs_rd(panel, output_dir)

    # Save additional data files
    panel.to_csv(os.path.join(output_dir, "mna_gravity_panel.csv"), index=False)
    # revision: persist the sectoral panels too, so a re-check of a single
    # specification does not require re-parsing the 3 GB Orbis XML
    panel_tech.to_csv(os.path.join(output_dir, "panel_tech.csv"), index=False)
    panel_nontech.to_csv(os.path.join(output_dir, "panel_nontech.csv"), index=False)
    msti_rd.to_csv(os.path.join(output_dir, "msti_rd_intensities.csv"), index=False)
    sector_agg.to_csv(os.path.join(output_dir, "sector_dummies_aggregated.csv"), index=False)
    
    print("\n" + "="*60)
    print("All outputs saved in:", output_dir)
    print("Tech panel outputs in:", output_dir_tech)
    print("Non-tech panel outputs in:", output_dir_nontech)
    print("="*60)

    # ==========================================================================
    # 8. REVISION SPECS  (замечания научрука 1.1, 2.1, 2.2, 2.3, 3.2, 3.3)
    # ==========================================================================
    print("\n" + "="*60)
    print("8. REVISION SPECS")
    print("="*60)

    out_rev = os.path.join(output_dir, "revision")
    os.makedirs(out_rev, exist_ok=True)

    # FIX 0 -- correct country-year lags (_l1 and _l2)
    panel         = rv.make_lags(panel,         lags=(1, 2))
    panel_tech    = rv.make_lags(panel_tech,    lags=(1, 2))
    panel_nontech = rv.make_lags(panel_nontech, lags=(1, 2))
    print("LAG AUDIT (share of rows where lag == contemporaneous; must be 0.0):",
          rv.audit_lags(panel))

    # save the lagged panels so round 2 can be rerun in a minute without
    # rebuilding anything from the Orbis extract:
    #     python revision_specs.py --panels output --out output/revision
    panel.to_csv(os.path.join(output_dir, "panel_full.csv"), index=False)
    panel_tech.to_csv(os.path.join(output_dir, "panel_tech.csv"), index=False)
    panel_nontech.to_csv(os.path.join(output_dir, "panel_nontech.csv"), index=False)
    deals[["iso_o", "iso_d", "year", "deal_value", "is_tech_target", "is_tech_acquirer"]].to_csv(
        os.path.join(output_dir, "deals_min.csv"), index=False)

    for lbl, pnl in (("full", panel), ("tech", panel_tech), ("nontech", panel_nontech)):
        print(f"\n----- revision specs: {lbl} panel -----")
        rv.spec9_coverage(pnl, lbl, out_rev)
        rv.spec8_descriptives(pnl, lbl, out_rev)
        rv.spec1_year_fe(pnl, lbl, out_rev)           # PRIORITY 1
        rv.spec4_log_patents(pnl, lbl, out_rev)       # PRIORITY 2
        rv.spec2_time_split(pnl, lbl, out_rev, 2010)  # PRIORITY 3
        rv.spec7_two_year_lag(pnl, lbl, out_rev)
        rv.spec5_income_control(pnl, lbl, out_rev)

    rv.spec2_time_split(panel_tech, "tech", out_rev, 2008)
    rv.spec2_time_split(panel_tech, "tech", out_rev, 2012)
    rv.spec3_sector_interaction(panel_tech, panel_nontech, out_rev)
    rv.spec10_break_by_sector(panel_tech, panel_nontech, out_rev, 2010)
    rv.spec10_break_by_sector(panel_tech, panel_nontech, out_rev, 2008)
    # spec6 now infers the sector from the label and measures disclosure inside
    # that sector's own deals. Round 1 passed the full deal table with a sectoral
    # panel, so a technology cell counted as disclosed whenever an unrelated
    # non-technology deal in the same dyad-year reported a value.
    rv.spec6_disclosure(deals, panel_tech,    "tech",    out_rev)
    rv.spec6_disclosure(deals, panel_nontech, "nontech", out_rev)

    # ---- ROUND 2: referee-report specifications (11, 12, 13, 15, 16) ---------
    rv.run_round2(panel, panel_tech, panel_nontech, out_rev)

    rv.dump(out_rev)
    print("\nREVISION OUTPUT:", out_rev)

    # Compile all results tables
    compile_all_tables(output_dir)


