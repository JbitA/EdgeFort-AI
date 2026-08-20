from __future__ import annotations

import argparse
import json
from edgeai.showcase import run_showcase


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the GitHub showcase experiment")
    parser.add_argument("--dataset", choices=["digits", "uci-har"], default="digits")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", default="results/showcase")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-iterations", type=int, default=20)
    args = parser.parse_args()
    result = run_showcase(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        benchmark_iterations=args.benchmark_iterations,
    )
    print(json.dumps({"dataset": result["dataset"]["name"], "output_dir": args.output_dir}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
