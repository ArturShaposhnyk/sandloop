import sys
from pathlib import Path

# Make top-level modules (config, geometry, capture, ...) importable when running `pytest`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
