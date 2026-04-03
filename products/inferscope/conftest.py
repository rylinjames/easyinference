# conftest.py at project root — ensures src/ is on sys.path for pytest
# This is needed because the package uses src/ layout and
# `uv run pytest` may not install the editable package.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
