"""Thin wrapper around the shared summary logic with local defaults."""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SHARED_ROOT = THIS_DIR.parent / "instructPix2Pix"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

# Patch THIS_DIR in the shared module so paths resolve to this project
import summary as _shared_summary

_shared_summary.THIS_DIR = THIS_DIR
_shared_summary.DEFAULT_OUT_ROOT = THIS_DIR / "out"
_shared_summary.DEFAULT_REPORT_DIR = THIS_DIR / "report"

if __name__ == "__main__":
    _shared_summary.main()
