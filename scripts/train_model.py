#!/usr/bin/env python3
"""
Entry-point script to train the used-car price prediction model.

Usage:
    python scripts/train_model.py

Runs the full pipeline in src/models/train.py: load -> clean -> engineer
features -> compare models -> tune -> evaluate -> save artifacts to models/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.train import run_training_pipeline  # noqa: E402


def main() -> None:
    result = run_training_pipeline()
    print("\n" + "=" * 60)
    print(f"Best model: {result.best_model_name}")
    print(f"Best params: {result.best_params}")
    print("Test set metrics:")
    for key, value in result.test_metrics.items():
        print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
