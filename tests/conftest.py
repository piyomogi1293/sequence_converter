"""Pytest configuration."""

import sys
from pathlib import Path

# Add src to path for pytest to find modules during import
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

# Ensure src is in path before other operations
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
