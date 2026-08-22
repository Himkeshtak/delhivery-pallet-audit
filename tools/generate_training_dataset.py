"""Generate deterministic domain-shifted YOLO pose and segmentation datasets.

The dataset is procedural training/evaluation evidence only. It cannot establish real warehouse
accuracy or replace physical camera calibration.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


WIDTH, HEIGHT = 320, 256
PIXELS_PER_METRE = 80.0
H_IMAGE_TO_FLOOR = np.array([
    [1 / PIXELS_PER_METRE, 0, -2.0],
    [0, -1 / PIXELS_PER_METRE, 3.2],
    [0, 0, 1],
], dtype=float)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    count: int
    domain: str


def rotated_rect(cx: float, cy: float, width: float, height: float,
                 angle: float) -> list[tuple[float, float]]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return [
        (cx + x * cosine - y * sine, cy + x * sine + y * cosine)
        for x, y in ((-width / 2, -height / 2), (width / 2, -height / 2),
                     (width / 2, height / 2), (-width / 2, height / 2))
    ]


def normalized_polygon(class_id: int, polygon: list[tuple[float, float]]) -> str:
    values = [str(class_id)]
    for x, y in polygon:
        values.extend((f"{np.clip(x / WIDTH, 0, 1):.7f}",
                       f"{np.clip(y / HEIGHT, 0, 1):.7f}"))
    return " ".join(values)


def floor_point(pixel: tuple[float, float]) -> tuple[float, float]:
    mapped = H_IMAGE_TO_FLOOR @ np.array([pixel[0], pixel[1], 1.0])
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


def background(rng: random.Random, domain: str) -> Image.Image:
    if domain == "train":
        wall = rng.choice(((185, 188, 184), (164, 170, 171), (198, 191, 177)))
        floor = rng.choice(((122, 126, 122), (142, 137, 128), (105, 111, 112)))
    elif domain == "validation":
        wall, floor = (178, 188, 197), (105, 117, 126)
    else:
        wall, floor = (116, 130, 148), (68, 78, 91)
    image = Image.new("RGB", (WIDTH, HEIGHT), wall)
    draw = ImageDraw.Draw(image)
    horizon = 82
    draw.rectangle((0, horizon, WIDTH, HEIGHT), fill=floor)
    for x in range(-WIDTH, WIDTH * 2, 55):
        draw.line((x, HEIGHT, x + 125, horizon),
                  fill=tuple(max(v - 18, 0) for v in floor), width=1)
    for y in range(horizon + 24, HEIGHT, 34):
        draw.line((0, y, WIDTH, y), fill=tuple(min(v + 14, 255) for v in floor), width=2)
    for x in (34, 286):
        draw.rectangle((x - 5, 18, x + 5, horizon + 22), fill=(74, 82, 85))
    draw.line((38, 232, 282, 232), fill=(222, 192, 50), width=3)
    return image


def render_sample(rng: random.Random, domain: str,
                  sample_id: int) -> tuple[Image.Image, dict, list[str]]:
    image = background(rng, domain)
    draw = ImageDraw.Draw(image)
    cx, cy = rng.uniform(105, 215), rng.uniform(184, 218)
    width, depth = rng.uniform(92, 132), rng.uniform(38, 58)
    angle = math.radians(rng.uniform(-38, 38))
    pallet = rotated_rect(cx, cy, width, depth, angle)
    draw.polygon([(x + 6, y + 7) for x, y in pallet], fill=(49, 52, 52))
    wood = rng.choice(((143, 92, 46), (119, 73, 38), (169, 116, 61), (92, 61, 38)))
    draw.polygon(pallet, fill=wood, outline=(48, 31, 19), width=3)
    for fraction in (-.3, 0, .3):
        ox, oy = -math.sin(angle) * depth * fraction, math.cos(angle) * depth * fraction
        a = (cx - math.cos(angle) * width * .46 + ox,
             cy - math.sin(angle) * width * .46 + oy)
        b = (cx + math.cos(angle) * width * .46 + ox,
             cy + math.sin(angle) * width * .46 + oy)
        draw.line((a, b), fill=(70, 43, 25), width=3)

    seg_lines = [normalized_polygon(0, pallet)]
    boxes: list[list[tuple[float, float]]] = []
    layers, per_layer = rng.randint(1, 3), rng.randint(2, 3)
    box_width, box_height = width * .82 / per_layer, rng.uniform(21, 29)
    for layer in range(layers):
        for column in range(per_layer):
            local_x = (column - (per_layer - 1) / 2) * box_width * 1.02
            lift = depth * .35 + layer * box_height * .72
            bx = cx + local_x * math.cos(angle) + math.sin(angle) * lift
            by = cy + local_x * math.sin(angle) - math.cos(angle) * lift
            polygon = rotated_rect(bx, by, box_width * .94, box_height, angle)
            colour = rng.choice(((181, 133, 76), (205, 171, 113),
                                 (139, 101, 62), (221, 202, 159)))
            draw.polygon(polygon, fill=colour, outline=(73, 55, 36), width=2)
            draw.line((polygon[0], polygon[2]),
                      fill=tuple(max(v - 25, 0) for v in colour), width=1)
            boxes.append(polygon)
            seg_lines.append(normalized_polygon(2, polygon))

    points = [point for polygon in boxes for point in polygon]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    load = [(min(xs), min(ys)), (max(xs), min(ys)),
            (max(xs), max(ys)), (min(xs), max(ys))]
    seg_lines.append(normalized_polygon(1, load))
    if rng.random() < .72:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).polygon(
            load, fill=(185, 225, 236, 28), outline=(204, 241, 248, 150), width=2)
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(image)
        seg_lines.append(normalized_polygon(3, load))

    if rng.random() < (.18 if domain == "test_shift" else .10):
        target = rng.choice(boxes)
        dx, dy = sum(p[0] for p in target) / 4, sum(p[1] for p in target) / 4
        damage = [(dx - 6, dy - 2), (dx - 1, dy - 6),
                  (dx + 6, dy), (dx + 1, dy + 5)]
        draw.polygon(damage, fill=(70, 39, 31))
        seg_lines.append(normalized_polygon(4, damage))
    if rng.random() < (.12 if domain == "test_shift" else .06):
        x, y = pallet[1]
        damage = [(x - 9, y - 4), (x + 3, y - 3),
                  (x + 7, y + 4), (x - 8, y + 3)]
        draw.polygon(damage, fill=(42, 29, 23))
        seg_lines.append(normalized_polygon(5, damage))

    visibility = [2, 2, 2, 2]
    if rng.random() < (.28 if domain == "test_shift" else .13):
        corner = rng.randrange(4)
        visibility[corner] = 1
        x, y = pallet[corner]
        draw.rectangle((x - 12, y - 17, x + 13, y + 13), fill=(45, 51, 57))

    noise = np.random.default_rng(10_000 + sample_id).normal(
        0, 11 if domain == "test_shift" else 5, (HEIGHT, WIDTH, 3))
    image = Image.fromarray(np.clip(np.asarray(image, dtype=np.int16) + noise,
                                    0, 255).astype(np.uint8))
    if domain == "test_shift" and sample_id % 3 == 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=.8))
    if domain == "test_shift" and sample_id % 4 == 0:
        glare = Image.new("RGBA", image.size, (0, 0, 0, 0))
        ImageDraw.Draw(glare).polygon(
            [(225, 82), (285, 82), (210, 255), (160, 255)],
            fill=(255, 255, 240, 36))
        image = Image.alpha_composite(image.convert("RGBA"), glare).convert("RGB")

    px, py = [p[0] for p in pallet], [p[1] for p in pallet]
    bbox = [min(px), min(py), max(px) - min(px), max(py) - min(py)]
    pose = ["0", f"{(bbox[0] + bbox[2] / 2) / WIDTH:.7f}",
            f"{(bbox[1] + bbox[3] / 2) / HEIGHT:.7f}",
            f"{bbox[2] / WIDTH:.7f}", f"{bbox[3] / HEIGHT:.7f}"]
    coco_keypoints = []
    for point, visible in zip(pallet, visibility):
        pose.extend((f"{point[0] / WIDTH:.7f}", f"{point[1] / HEIGHT:.7f}", str(visible)))
        coco_keypoints.extend((round(point[0], 3), round(point[1], 3), visible))
    floor = [floor_point(p) for p in pallet]
    centre = np.mean(np.asarray(floor), axis=0)
    theta = math.degrees(math.atan2(
        floor[1][1] - floor[0][1], floor[1][0] - floor[0][0]))
    record = {
        "pose_label": " ".join(pose), "bbox": bbox, "keypoints": coco_keypoints,
        "ground_truth": {"x_m": float(centre[0]), "y_m": float(centre[1]),
                         "theta_deg": theta},
        "keypoints_px": pallet,
    }
    return image, record, seg_lines


def generate(output: Path, specs: list[SplitSpec], seed: int) -> dict:
    if output.exists():
        shutil.rmtree(output)
    pose_root, seg_root = output / "pallet_pose", output / "load_seg"
    coco = {
        "info": {"description": "Procedural pallet data", "seed": seed,
                 "limitations": "Synthetic-only; not warehouse performance evidence"},
        "categories": [{"id": 1, "name": "pallet",
                        "keypoints": ["front_left", "front_right", "back_right", "back_left"]}],
        "images": [], "annotations": [],
    }
    evaluation, counts, image_id = [], {}, 1
    for split_index, spec in enumerate(specs):
        rng = random.Random(seed + split_index * 1000)
        counts[spec.name] = {"images": spec.count, "pallets": spec.count,
                             "domain": spec.domain}
        for index in range(spec.count):
            image, record, seg_lines = render_sample(rng, spec.domain, image_id)
            filename = f"{spec.name}_{index:04d}.jpg"
            for root in (pose_root, seg_root):
                image_dir = root / "images" / spec.name
                image_dir.mkdir(parents=True, exist_ok=True)
                image.save(image_dir / filename, quality=88, optimize=True)
            pose_labels = pose_root / "labels" / spec.name
            seg_labels = seg_root / "labels" / spec.name
            pose_labels.mkdir(parents=True, exist_ok=True)
            seg_labels.mkdir(parents=True, exist_ok=True)
            (pose_labels / f"{Path(filename).stem}.txt").write_text(
                record["pose_label"], encoding="utf-8")
            (seg_labels / f"{Path(filename).stem}.txt").write_text(
                "\n".join(seg_lines), encoding="utf-8")
            coco["images"].append({
                "id": image_id, "file_name": f"{spec.name}/{filename}",
                "width": WIDTH, "height": HEIGHT, "capture_session": spec.domain,
                "synthetic": True,
            })
            coco["annotations"].append({
                "id": image_id, "image_id": image_id, "category_id": 1,
                "bbox": record["bbox"], "area": record["bbox"][2] * record["bbox"][3],
                "iscrowd": 0, "num_keypoints": 4, "keypoints": record["keypoints"],
            })
            if spec.name == "test":
                evaluation.append({
                    "image": f"pallet_pose/images/test/{filename}",
                    "homography_image_to_floor": H_IMAGE_TO_FLOOR.tolist(),
                    "reprojection_rmse_px": 0.0,
                    "keypoints_px": record["keypoints_px"],
                    "confidences": [1, 1, 1, 1],
                    "ground_truth": record["ground_truth"], "domain": spec.domain,
                })
            image_id += 1
    manifest = {
        "dataset_version": "synthetic-v1", "seed": seed,
        "generator": "tools/generate_training_dataset.py",
        "license": "Generated by this repository under MIT; no third-party imagery",
        "counts": counts, "total_images": sum(item.count for item in specs),
        "held_out_difference": "test uses unseen cool/dark palette, stronger noise, blur and glare",
        "warning": "Synthetic training benchmark only; does not establish warehouse accuracy",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "annotations_coco.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    (output / "evaluation_pose_records.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8")
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "synthetic_calibration.json").write_text(json.dumps({
        "calibration_id": "synthetic-affine-v1", "image_size": [WIDTH, HEIGHT],
        "homography_image_to_floor": H_IMAGE_TO_FLOOR.tolist(),
        "reprojection_rmse_px": 0.0, "units": "metres",
        "warning": "Synthetic generator calibration; not a physical camera artefact",
    }, indent=2), encoding="utf-8")
    pose_yaml = (f"path: {pose_root.resolve().as_posix()}\ntrain: images/train\n"
                 "val: images/val\ntest: images/test\nkpt_shape: [4, 3]\n"
                 "flip_idx: [1, 0, 3, 2]\nnames:\n  0: pallet\n")
    seg_yaml = (f"path: {seg_root.resolve().as_posix()}\ntrain: images/train\n"
                "val: images/val\ntest: images/test\nnames:\n  0: pallet\n"
                "  1: load\n  2: box\n  3: stretch_wrap\n"
                "  4: visible_box_damage\n  5: visible_pallet_damage\n")
    (output / "pallet_pose.yaml").write_text(pose_yaml, encoding="utf-8")
    (output / "load_seg.yaml").write_text(seg_yaml, encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--train", type=int, default=240)
    parser.add_argument("--val", type=int, default=60)
    parser.add_argument("--test", type=int, default=60)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    specs = [
        SplitSpec("train", args.train, "train"),
        SplitSpec("val", args.val, "validation"),
        SplitSpec("test", args.test, "test_shift"),
    ]
    print(json.dumps(generate(args.output, specs, args.seed), indent=2))
