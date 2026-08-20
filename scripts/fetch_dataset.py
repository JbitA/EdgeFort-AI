from __future__ import annotations

import argparse
from edgeai.datasets import fetch_uci_har


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire a public showcase dataset with provenance")
    parser.add_argument("dataset", choices=["uci-har"])
    parser.add_argument("--destination", default="data/uci-har")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.dataset == "uci-har":
        path = fetch_uci_har(args.destination, force=args.force)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
