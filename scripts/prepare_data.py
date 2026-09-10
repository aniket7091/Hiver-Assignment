"""Run the canonical raw-data extraction, reconstruction, and validation flow."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# `04_build_conversations.py` is retained as an earlier reconstruction
# experiment. The v2 extractor is the documented canonical preparation path.
PREPARATION_STEPS = (
    "scripts/data/03_extract_amazonhelp.py",
    "scripts/data/04_extract_amazonhelp_context_v2.py",
    "scripts/data/04_validate_context.py",
    "scripts/data/05_reconstruct_conversations.py",
    "scripts/data/06_validate_data.py",
)


def main() -> None:
    for relative_script in PREPARATION_STEPS:
        print(f"\nRunning {relative_script}")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / relative_script)],
            check=True,
            cwd=PROJECT_ROOT,
        )


if __name__ == "__main__":
    main()
