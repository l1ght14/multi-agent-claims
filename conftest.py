"""Make `src` importable as a package from the repository root, so tests, the claim runner and
the replay check all reach the library the same way instead of each adding `src/` to
`sys.path` and creating a second import convention."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
