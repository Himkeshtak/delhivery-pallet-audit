"""Convert four-keypoint COCO annotations into leakage-safe YOLO pose splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path


def split_for_group(group: str, seed: int, train: float, val: float) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if value < train else "val" if value < train + val else "test"


def convert(coco_path: Path, images_root: Path, output: Path, seed: int = 7,
            train_fraction: float = .7, val_fraction: float = .15) -> dict[str, int]:
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images = {row["id"]: row for row in coco["images"]}
    annotations: dict[int, list[dict]] = defaultdict(list)
    for row in coco["annotations"]:
        annotations[row["image_id"]].append(row)
    counts = defaultdict(int)
    for image_id, metadata in images.items():
        group = str(metadata.get("capture_session") or metadata.get("group") or metadata["file_name"])
        split = split_for_group(group, seed, train_fraction, val_fraction)
        source = images_root / metadata["file_name"]
        if not source.exists():
            raise FileNotFoundError(source)
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_dir / source.name)
        lines = []
        width, height = float(metadata["width"]), float(metadata["height"])
        for ann in annotations[image_id]:
            x, y, w, h = (float(v) for v in ann["bbox"])
            kpts = ann["keypoints"]
            if len(kpts) != 12:
                raise ValueError(f"annotation {ann['id']} must contain four COCO keypoints")
            fields = [str(int(ann.get("category_id", 1)) - 1), f"{(x+w/2)/width:.8f}",
                      f"{(y+h/2)/height:.8f}", f"{w/width:.8f}", f"{h/height:.8f}"]
            for index in range(0, 12, 3):
                fields.extend((f"{float(kpts[index])/width:.8f}", f"{float(kpts[index+1])/height:.8f}",
                               str(int(kpts[index+2]))))
            lines.append(" ".join(fields))
        (label_dir / f"{source.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        counts[split] += 1
    manifest = {"seed": seed, "split_unit": "capture_session/group", "counts": dict(counts)}
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dict(counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("coco", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(convert(args.coco, args.images, args.output, args.seed))

