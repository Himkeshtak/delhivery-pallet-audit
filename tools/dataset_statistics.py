"""Generate exact annotation counts for the committed synthetic dataset card."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


CLASS_NAMES = {
    0: "pallet",
    1: "load",
    2: "box",
    3: "stretch_wrap",
    4: "visible_box_damage",
    5: "visible_pallet_damage",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("data"))
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("data/dataset_statistics.json"))
    args = parser.parse_args()
    splits = {}
    totals: Counter[int] = Counter()
    visibility_totals: Counter[int] = Counter()
    for split in ("train", "val", "test"):
        segment_counts: Counter[int] = Counter()
        for path in sorted((args.root / "load_seg" / "labels" / split).glob("*.txt")):
            for line in path.read_text(encoding="utf-8").splitlines():
                segment_counts[int(line.split()[0])] += 1
        visibility: Counter[int] = Counter()
        pose_labels = sorted((args.root / "pallet_pose" / "labels" / split).glob("*.txt"))
        for path in pose_labels:
            fields = path.read_text(encoding="utf-8").split()
            for index in range(7, 17, 3):
                visibility[int(float(fields[index]))] += 1
        totals.update(segment_counts)
        visibility_totals.update(visibility)
        splits[split] = {
            "images": len(pose_labels),
            "pose_instances": len(pose_labels),
            "pose_keypoints": sum(visibility.values()),
            "keypoint_visibility": {
                str(key): visibility.get(key, 0) for key in (0, 1, 2)
            },
            "segmentation_instances": {
                CLASS_NAMES[key]: segment_counts.get(key, 0) for key in CLASS_NAMES
            },
        }
    output = {
        "dataset_version": "synthetic-v1",
        "splits": splits,
        "totals": {
            "images": sum(item["images"] for item in splits.values()),
            "pose_instances": sum(item["pose_instances"] for item in splits.values()),
            "pose_keypoints": sum(item["pose_keypoints"] for item in splits.values()),
            "keypoint_visibility": {
                str(key): visibility_totals.get(key, 0) for key in (0, 1, 2)
            },
            "segmentation_instances": {
                CLASS_NAMES[key]: totals.get(key, 0) for key in CLASS_NAMES
            },
        },
        "generated_by": "tools/dataset_statistics.py",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
