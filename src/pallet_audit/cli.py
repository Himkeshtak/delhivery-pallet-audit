from __future__ import annotations

import argparse

from .io import run_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess one detected pallet from JSON evidence")
    parser.add_argument("input", help="input request JSON")
    parser.add_argument("-o", "--output", default="assessment.json")
    args = parser.parse_args()
    run_file(args.input, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
