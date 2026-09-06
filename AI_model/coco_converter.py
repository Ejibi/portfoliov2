"""
utils/coco_converter.py  —  AxisDeepLabV3+

Converts a Roboflow COCO-JSON export into pre-rendered (H, W, 9) float32
numpy masks — one .npy file per image, same stem as the image file.

Run this ONCE before training if you want faster DataLoader I/O
(avoids re-rasterising polygons on every epoch).  The AxisCarDamageDataset
class in dataset.py also does this on-the-fly, so this script is optional
but recommended for large datasets.

Usage:
    python utils/coco_converter.py \
        --json  data/train/_annotations.coco.json \
        --imgs  data/train \
        --out   data/train_masks

Output layout:
    data/train_masks/
        image_001.npy    # shape (H, W, 9)  float32  binary 0/1
        image_002.npy
        ...
    data/train_masks/_category_map.json   # records name→channel mapping
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List
import pycocotools.mask as mask_util

# Default class list — must match config.py CLASSES
DEFAULT_CLASSES: List[str] = [
    "Dent", "Scratch", "Corrosion",
    "Cracked_Glass", "Shattered_Glass",
    "Panel_Misalignment", "Fragmentation","Car_Inner_texture",
    "Panel_Crumpling"
]


def build_category_map(
    coco_categories: List[dict],
    class_names:     List[str],
) -> Dict[int, int]:
    """
    Maps COCO category_id → channel index.
    Matching is case-insensitive and strips whitespace.
    Prints a warning for any COCO category that cannot be matched.
    """
    name_to_idx = {n.lower().strip(): i for i, n in enumerate(class_names)}
    cat_map: Dict[int, int] = {}
    unmatched = []

    for cat in coco_categories:
        key = cat["name"].lower().strip()
        if key in name_to_idx:
            cat_map[cat["id"]] = name_to_idx[key]
        else:
            unmatched.append(cat["name"])

    if unmatched:
        print(f"  WARNING — unmatched COCO categories (will be ignored): {unmatched}")
        print(f"  Expected class names: {class_names}")

    return cat_map


def rasterise(
    annotations: List[dict],
    height:      int,
    width:       int,
    cat_map:     Dict[int, int],
    num_classes: int,
) -> np.ndarray:
    """Polygon annotations → (H, W, C) float32 binary array."""
    masks = np.zeros((num_classes, height, width), dtype=np.float32)
    for ann in annotations:
        ch = cat_map.get(ann.get("category_id"))
        if ch is None:
            continue
        
        segmentation = ann.get("segmentation", [])
        
        if isinstance(segmentation, dict):
            # Decode the RLE string into a binary mask (0s and 1s)
            decoded_mask = mask_util.decode(segmentation)
            # Merge it into our main masks array using maximum to preserve overlaps
            masks[ch] = np.maximum(masks[ch], decoded_mask)
            continue
            
        for seg in segmentation:
            if len(seg) < 6:
                continue
            pts = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(masks[ch], [pts], 1.0)
            
    return masks.transpose(1, 2, 0)


def convert(
    json_path:   Path,
    img_dir:     Path,
    output_dir:  Path,
    class_names: List[str] = DEFAULT_CLASSES,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        coco = json.load(f)

    cat_map = build_category_map(coco["categories"], class_names)
    print(f"  Category map: {cat_map}")

    # Group annotations by image id
    ann_by_img: Dict[int, List[dict]] = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    ok = skip = 0
    for img_info in coco["images"]:
        # Find the image file (handle subdirectories in file_name)
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            # Try just the filename stem
            candidates = list(img_dir.glob(img_info["file_name"].split("/")[-1]))
            if not candidates:
                skip += 1
                continue
            img_path = candidates[0]

        anns   = ann_by_img.get(img_info["id"], [])
        masks  = rasterise(
            anns,
            img_info["height"],
            img_info["width"],
            cat_map,
            len(class_names),
        )
        out_path = output_dir / (Path(img_info["file_name"]).stem + ".npy")
        np.save(out_path, masks)
        ok += 1

    # Save category map for reference
    with open(output_dir / "_category_map.json", "w") as f:
        json.dump({"classes": class_names, "coco_cat_to_channel": cat_map}, f, indent=2)

    print(f"  Done — {ok} masks saved to {output_dir}  ({skip} skipped)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert Roboflow COCO JSON to .npy masks")
    p.add_argument("--json",    required=True, help="Path to _annotations.coco.json")
    p.add_argument("--imgs",    required=True, help="Directory containing the images")
    p.add_argument("--out",     required=True, help="Output directory for .npy masks")
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                   help="Ordered class names matching config.py CLASSES")
    args = p.parse_args()

    convert(
        json_path=Path(args.json),
        img_dir=Path(args.imgs),
        output_dir=Path(args.out),
        class_names=args.classes,
    )
