"""Pytest configuration."""

import sys
from pathlib import Path

# プロジェクトルートのsrcディレクトリをsys.pathに追加
# これは、pytestがテストモジュールをインポートする前に実行される
project_root = Path(__file__).parent  # conftest.pyの親 = プロジェクトルート
src_path = project_root / "src"

if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def pytest_configure(config):
    """Called after command line options have been parsed."""
    # Ensure src is in the path before any test imports
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
