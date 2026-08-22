"""Generate deterministic COCO-keypoints pallet images for pipeline smoke tests.

These images are deliberately simple and MUST NOT be reported as real-world accuracy.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw


def generate(output: Path, count: int, seed: int) -> None:
    random.seed(seed)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for idx in range(count):
        width, height = 640, 480
        image = Image.new("RGB", (width, height), (188, 190, 186))
        draw = ImageDraw.Draw(image)
        cx, cy = random.randint(180, 460), random.randint(250, 390)
        theta = math.radians(random.uniform(-35, 35))
        length, depth = random.randint(170, 240), random.randint(70, 120)
        local = [(-length/2, -depth/2), (length/2, -depth/2), (length/2, depth/2), (-length/2, depth/2)]
        corners = []
        for x, y in local:
            corners.append((cx + x*math.cos(theta)-y*math.sin(theta), cy + x*math.sin(theta)+y*math.cos(theta)))
        draw.polygon(corners, fill=(133, 84, 42), outline=(58, 37, 21), width=4)
        for a, b in zip(corners, corners[1:] + corners[:1]):
            draw.line((a, b), fill=(57, 35, 19), width=4)
        name = f"synthetic_{idx:04d}.png"
        image.save(images_dir / name)
        flat = [v for p in corners for v in (round(p[0], 2), round(p[1], 2), 2)]
        xs, ys = [p[0] for p in corners], [p[1] for p in corners]
        images.append({"id": idx + 1, "file_name": name, "width": width, "height": height, "synthetic": True})
        annotations.append({"id": idx + 1, "image_id": idx + 1, "category_id": 1,
                            "bbox": [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)],
                            "area": length*depth, "iscrowd": 0, "num_keypoints": 4, "keypoints": flat})
    coco = {"info": {"description": "Synthetic smoke-test data; not a performance benchmark", "seed": seed},
            "images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "pallet", "keypoints": ["front_left", "front_right", "back_right", "back_left"],
                            "skeleton": [[1, 2], [2, 3], [3, 4], [4, 1]]}]}
    (output / "annotations.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    generate(args.output, args.count, args.seed)

