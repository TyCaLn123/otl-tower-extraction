from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otl_tower_extraction import load_config, TowerExtractionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = TowerExtractionPipeline(cfg).run()
    print(result)


if __name__ == "__main__":
    main()
