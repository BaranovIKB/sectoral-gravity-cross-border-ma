"""Run the whole pipeline. Default: synthetic data and a fast setting, a few minutes on a laptop.

python run_all.py                     synthetic data, 200 placebo draws
python run_all.py --reps 10000        synthetic data, full number of draws
THESIS_DATA=orbis python run_all.py --reps 10000 --block-reps 2000     your own licensed extract
"""
import argparse, os, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--block-reps", type=int, default=100)
ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
a = ap.parse_args()
env = dict(os.environ)
data = env.setdefault("THESIS_DATA", "synthetic")
if a.reps < 2000:
    env["THESIS_FAST"] = "1"
src = Path(__file__).resolve().parent / "src"


def run(*args):
    print(">>", *args, flush=True)
    subprocess.run([sys.executable, *args], cwd=src, env=env, check=True)


if data == "synthetic":
    run("make_synthetic_deals.py")
    run("test_recovery.py")
run("01_reproduce.py")
run("02_ri_placebo.py", "--design", "iid", "--reps", str(a.reps), "--workers", str(a.workers))
run("02_ri_placebo.py", "--design", "dyadshare", "--reps", str(a.block_reps), "--workers", str(a.workers))
run("05_ri_summary.py")
run("04_new_specs.py")
run("06_figures.py")
if data == "orbis":
    run("check_against_thesis.py")
print("done: results in results/" + data)
