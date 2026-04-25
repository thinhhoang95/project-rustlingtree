"""Test package marker.

This allows test modules to import shared helpers via `tests.helpers` and
ensures the in-repo `src/` tree is importable during local pytest runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
