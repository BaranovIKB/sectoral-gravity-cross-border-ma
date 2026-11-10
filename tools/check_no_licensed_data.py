"""Stop a commit that would add licensed Orbis (Bureau van Dijk) data to the repository.

Checks every file that git would commit: path patterns of the deal extract and panels,
Orbis field names in table headers, and files under data/orbis other than its README.
Exit code 1 lists the offending files.

usage: python tools/check_no_licensed_data.py          (all tracked and staged files)
installed as .git/hooks/pre-commit by tools/install_hook.py
"""
import gzip, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAD_PATH = re.compile(r"(^|/)(deals_(min|raw)[^/]*|panel_[^/]*\.csv|mna_gravity_panel[^/]*|gravity_panel[^/]*|sector_dummies[^/]*|[^/]*\.xml)$", re.I)
# Orbis M&A export fields and the deal-level columns built from them
BAD_COLS = {"alldlvalue", "trgtval", "target_scuscd", "bidder_scuscd", "bidder_ctryc", "target_ctryc",
            "bidder_strategic_ctryc", "completion_date", "deal_value", "is_tech_target", "is_tech_acquirer",
            "finlstak", "dealtype", "target_name", "bidder_name", "deal_number", "total_value", "tech_deals"}
SYNTHETIC = ("data/synthetic/", "results/synthetic/")


def files():
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split("\n")
    return [f for f in out if f]


def header(path):
    p = ROOT / path
    try:
        if path.endswith(".gz"):
            with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as f:
                return f.readline()
        if path.endswith((".csv", ".tsv", ".txt")):
            with open(p, encoding="utf-8", errors="ignore") as f:
                return f.readline()
    except OSError:
        return ""
    return ""


bad = []
for f in files():
    if f.startswith(SYNTHETIC):
        continue
    if f.startswith("data/orbis/") and f != "data/orbis/README.md":
        bad.append((f, "file under data/orbis"))
    elif BAD_PATH.search(f):
        bad.append((f, "file name of the deal extract or a panel"))
    else:
        cols = {c.strip().strip('"').lower() for c in re.split(r"[,\t;]", header(f))}
        hit = cols & BAD_COLS
        if hit:
            bad.append((f, "deal-level columns: " + ", ".join(sorted(hit))))

if bad:
    print("Licensed data check FAILED. Remove these files from the commit:")
    for f, why in bad:
        print(f"  {f}: {why}")
    sys.exit(1)
print(f"Licensed data check passed: {len(files())} files, none with Orbis records or deal-level columns.")
