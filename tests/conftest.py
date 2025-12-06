"""Test configuration helpers."""

import sys
from pathlib import Path


# Ensure the project root (which contains the ``src`` package) is always importable even
# when PYTHONSAFEPATH or similar settings remove the working directory from sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))