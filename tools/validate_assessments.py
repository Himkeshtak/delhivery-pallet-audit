"""Validate every generated per-pallet assessment against the public schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--schema", type=Path,
                        default=Path("schemas/assessment.schema.json"))
    args = parser.parse_args()
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise SystemExit("Install jsonschema to validate assessment artefacts") from exc

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    paths = sorted(args.input_dir.glob("*_pallet-*.json"))
    if not paths:
        raise SystemExit(f"No per-pallet assessment JSON found in {args.input_dir}")
    errors = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        for error in validator.iter_errors(record):
            errors.append({"file": str(path), "path": list(error.path),
                           "message": error.message})
    if errors:
        print(json.dumps(errors, indent=2))
        raise SystemExit(f"{len(errors)} schema validation error(s)")
    print(json.dumps({"validated": len(paths), "schema": str(args.schema)}, indent=2))


if __name__ == "__main__":
    main()
