from __future__ import annotations

import argparse
import json

from .datasets import fetch_uci_har
from .evaluation import evaluate, robustness_campaign
from .showcase import run_showcase


def main(argv=None):
    p = argparse.ArgumentParser(prog="edgeai")
    sub = p.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="run the compact offline reference evaluation")
    ev.add_argument("--output")

    rb = sub.add_parser("robustness", help="run the multi-seed quantization robustness campaign")
    rb.add_argument("--output")

    ds = sub.add_parser("dataset", help="acquire public showcase datasets with provenance")
    ds_sub = ds.add_subparsers(dest="dataset_command", required=True)
    fetch = ds_sub.add_parser("fetch", help="download and safely extract a public dataset")
    fetch.add_argument("dataset", choices=["uci-har"])
    fetch.add_argument("--destination", default="data/uci-har")
    fetch.add_argument("--force", action="store_true")

    sh = sub.add_parser("showcase", help="run the GitHub-facing lifecycle experiment")
    sh.add_argument("--dataset", choices=["digits", "uci-har"], default="digits")
    sh.add_argument("--data-dir")
    sh.add_argument("--output-dir", default="results/showcase")
    sh.add_argument("--seed", type=int, default=42)
    sh.add_argument("--benchmark-iterations", type=int, default=20)

    args = p.parse_args(argv)

    if args.command == "dataset":
        if args.dataset_command == "fetch" and args.dataset == "uci-har":
            path = fetch_uci_har(args.destination, force=args.force)
            print(path)
            return 0

    if args.command == "showcase":
        result = run_showcase(
            dataset=args.dataset,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            benchmark_iterations=args.benchmark_iterations,
        )
        print(
            json.dumps(
                {
                    "dataset": result["dataset"]["name"],
                    "results": str(args.output_dir),
                    "bad_signed_release": result["deployment"]["bad_signed_release"],
                },
                indent=2,
                allow_nan=False,
            )
        )
        return 0

    data = evaluate() if args.command == "evaluate" else robustness_campaign()
    text = json.dumps(data, indent=2, allow_nan=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
