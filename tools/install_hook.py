"""Install the licensed-data check as a git pre-commit hook in this clone."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
hook = root / ".git" / "hooks" / "pre-commit"
hook.write_text("#!/bin/sh\npython tools/check_no_licensed_data.py || exit 1\n", encoding="utf-8")
hook.chmod(0o755)
print("installed", hook)
