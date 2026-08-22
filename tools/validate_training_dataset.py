"""Fail fast on missing, malformed, or out-of-range synthetic YOLO artefacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def validate(root: Path) -> dict:
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    expected = int(manifest["total_images"])
    summary: dict[str, object] = {"expected_images": expected, "splits": {}}
    hashes: set[str] = set()
    for split in ("train", "val", "test"):
        pose_images = sorted((root / "pallet_pose" / "images" / split).glob("*.jpg"))
        pose_labels = sorted((root / "pallet_pose" / "labels" / split).glob("*.txt"))
        seg_images = sorted((root / "load_seg" / "images" / split).glob("*.jpg"))
        seg_labels = sorted((root / "load_seg" / "labels" / split).glob("*.txt"))
        counts = {len(pose_images), len(pose_labels), len(seg_images), len(seg_labels)}
        if len(counts) != 1:
            raise ValueError(f"image/label count mismatch in {split}: {counts}")
        for image, label in zip(pose_images, pose_labels):
            if image.stem != label.stem:
                raise ValueError(f"unpaired pose files: {image}, {label}")
            fields = label.read_text(encoding="utf-8").split()
            if len(fields) != 17:
                raise ValueError(f"pose label must have 17 fields: {label}")
            bbox_values = [float(value) for value in fields[1:5]]
            if any(value < 0 or value > 1 for value in bbox_values):
                raise ValueError(f"pose box outside [0,1]: {label}")
            for index in range(5, 17, 3):
                x, y, visibility = map(float, fields[index:index + 3])
                if x < 0 or x > 1 or y < 0 or y > 1:
                    raise ValueError(f"pose keypoint outside [0,1]: {label}")
                if visibility not in (0, 1, 2):
                    raise ValueError(f"invalid COCO visibility: {label}")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes:
                raise ValueError(f"duplicate image crosses dataset rows: {image}")
            hashes.add(digest)
        for label in seg_labels:
            for line in label.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) < 7 or len(fields) % 2 == 0:
                    raise ValueError(f"invalid segmentation polygon: {label}")
                if any(float(value) < 0 or float(value) > 1 for value in fields[1:]):
                    raise ValueError(f"segmentation coordinate outside [0,1]: {label}")
        summary["splits"][split] = len(pose_images)  # type: ignore[index]
    if sum(summary["splits"].values()) != expected:  # type: ignore[union-attr]
        raise ValueError("manifest total does not match files")
    summary["unique_images"] = len(hashes)
    summary["status"] = "valid"
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(validate(args.root), indent=2))
