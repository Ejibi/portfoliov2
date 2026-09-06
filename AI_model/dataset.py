"""
dataset.py  —  AxisDeepLabV3+

Loads Roboflow COCO-JSON exports and builds (B, 7, H, W) binary mask tensors
so that every class channel is an independent binary segmentation target.
Overlapping classes (dent + scratch on the same pixel) are expressed naturally:
mask[0, y, x] = 1  AND  mask[1, y, x] = 1  simultaneously.

Augmentation design for car-damage:
  Geometric   — perspective, elastic, rotation: models camera angle variation.
  Photometric — HSV, CLAHE, shadow, rain:       models lighting & weather.
  Noise/blur  — ISO noise, motion blur:         models phone-camera quality.
  Coarse dropout — hides patches:               forces context reasoning.
  All spatial transforms apply identically to every mask channel.
"""

from unicodedata import name

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CLASSES
import pycocotools.mask as mask_util

# ── Augmentation pipelines ────────────────────────────────────────────────────

def train_transforms(image_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    h, w = image_size
    return A.Compose([
        # ── Spatial ──────────────────────────────────────────────────────────
        A.RandomResizedCrop(size=(h, w), scale=(0.45, 1.0),
                            ratio=(0.75, 1.33), interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), 
                 rotate=(-15, 15), shear=(-5, 5), p=0.5),
        
        # ── Photometric & Noise ──────────────────────────────────────────────
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.4),
        
        A.GaussNoise(p=0.3), 
        
        A.CoarseDropout(
            num_holes_range=(4, 8),
            hole_height_range=(20, 50),
            hole_width_range=(20, 50),
            fill=0,        
            fill_mask=None,   
            p=0.4
        ),
        # ── Outputs ──────────────────────────────────────────────────────────
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]) # additional_targets removed completely!

def val_transforms(image_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    h, w = image_size
    return A.Compose([
        A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def tta_transforms(image_size: Tuple[int,int] = (512, 512)) -> List[A.Compose]:
    """Four TTA views: original, h-flip, v-flip, brightness jitter."""
    h, w = image_size
    base = [
        A.Resize(h, w, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return [
        A.Compose(base),
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
        A.Compose([A.VerticalFlip(p=1.0)]   + base),
        A.Compose([A.RandomBrightnessContrast(
            brightness_limit=0.10, contrast_limit=0.10, p=1.0)] + base),
    ]


# ── Dataset ───────────────────────────────────────────────────────────────────

class AxisCarDamageDataset(Dataset):
    """
    Loads a Roboflow COCO-JSON export and returns:
        image : (3, H, W)  float32  — normalised RGB
        masks : (9, H, W)  float32  — one binary channel per damage class,
                                       co-occurrence fully supported.

    The COCO JSON must have category names that match CLASSES exactly.
    Run utils/coco_converter.py once to verify and remap if needed.
    """

    def __init__(
        self,
        coco_json_path: Path,
        img_dir:        Path,
        transform:      Optional[A.Compose] = None,
        class_names:    List[str]           = CLASSES,
        mask_dir:        Optional[Path]           = None,
    ):
        self.img_dir     = Path(img_dir)
        self.transform   = transform
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.mask_dir = Path(mask_dir) if mask_dir else None

        # ── Parse COCO JSON ───────────────────────────────────────────────────
        with open(coco_json_path) as f:
            coco = json.load(f)

        # Build class_name → channel_index map
        # Category names in JSON must match CLASSES (case-insensitive strip)
        self._cat_to_idx: Dict[int, int] = {}
        name_to_idx = {n.lower().strip(): i for i, n in enumerate(class_names)}
        for cat in coco["categories"]:
            key = cat["name"].lower().strip()
            if key in name_to_idx:
                self._cat_to_idx[cat["id"]] = name_to_idx[key]

        # Group annotations by image id
        ann_by_img: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

        self.samples: List[Tuple[Path, List[dict], int, int]] = []
        for img_info in coco["images"]:
            img_path = self.img_dir / img_info["file_name"]
            if not img_path.exists():
                continue
            anns = ann_by_img.get(img_info["id"], [])
            self.samples.append((
                img_path, anns,
                img_info["height"], img_info["width"],
            ))

        if not self.samples:
            raise RuntimeError(
                f"No valid images found.\n"
                f"  JSON: {coco_json_path}\n"
                f"  img_dir: {img_dir}\n"
                f"  Matched categories: {self._cat_to_idx}"
            )
            
    def __len__(self) -> int:
        return len(self.samples)

    def _build_masks(
            self,
            annotations: List[dict],
            height: int,
            width:  int,
        ) -> np.ndarray:
            """
            Rasterise polygon annotations into a (H, W, C) float32 binary array.
            """
            # Create as (C, H, W) for OpenCV memory contiguity
            masks = np.zeros((self.num_classes, height, width), dtype=np.float32)
            for ann in annotations:
                cat_id = ann.get("category_id")
                ch = self._cat_to_idx.get(cat_id)
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
                    # Draw on contiguous slice
                    cv2.fillPoly(masks[ch], [pts], 1.0)
                    
            # Transpose back to (H, W, C)
            return masks.transpose(1, 2, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, annotations, height, width = self.samples[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.mask_dir:
            npy_path = self.mask_dir / f"{img_path.stem}.npy"
            if npy_path.exists():
                masks_hwc = np.load(str(npy_path))
            else:
                masks_hwc = self._build_masks(annotations, height, width)
        else:
            masks_hwc = self._build_masks(annotations, height, width)
            
        if self.transform is not None:
            result = self.transform(image=image, mask=masks_hwc)
            image = result["image"]
            masks = result["mask"]
            if not isinstance(masks, torch.Tensor):
                masks = torch.from_numpy(masks)
            if masks.ndim == 3 and masks.shape[-1] == self.num_classes:
                masks = masks.permute(2, 0, 1)
            masks = masks.float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            masks = torch.from_numpy(masks_hwc).permute(2, 0, 1).float()

        return image, masks
