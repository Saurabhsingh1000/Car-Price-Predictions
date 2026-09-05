"""Root conftest.py — ensures the project root is on sys.path.

This allows pytest to import ``src`` and ``app`` packages on any platform /
CI runner without needing an editable install or PYTHONPATH manipulation in
the shell.
"""

import sys
from pathlib import Path

# Insert the project root (directory containing this file) at the front of
# sys.path so that ``import src`` and ``import app`` always resolve correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
