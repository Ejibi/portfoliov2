"""
predict.py  —  AxisDeepLabV3+

Inference pipeline:
  • Test-Time Augmentation (TTA) — 4 views: original, h-flip, v-flip,
    brightness jitter.  Predictions are un-flipped before averaging.
    Typically adds +0.5–2 mIoU at 4× inference cost.
  • Per-class sigmoid thresholds — tunable per class so rare types
    (panel_separation, fragmentation) can use a lower threshold.
  • 7-class colour overlay  — each class rendered with its own colour at
    configurable alpha, overlapping classes blend visually.
  • Probability map export  — saves a (H, W, 7) float32 .npy per image for
    downstream analysis (e.g. severity scoring, reporting).
  • JSON summary  — damage_pct, severity label, and per-class flags per image.

CLI:
    python predict.py --input /path/to/images --num_images --output results/ 
                      --checkpoint checkpoints/best.pt [--no-tta]
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Union
from random import shuffle

import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config  import Config, CLASSES, CLASS_COLOURS_BGR
from model   import AxisDeepLabV3Plus
from dataset import tta_transforms, val_transforms


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: Union[str, Path], device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 1. Restore the exact config used during training
    cfg = ckpt.get("cfg", Config()) 
    
    # 2. Build model dynamically based on the saved config
    model = AxisDeepLabV3Plus(
        encoder_name=cfg.encoder_name,
        encoder_weights=None,       
        num_classes=cfg.num_classes,
        use_aux_head=cfg.use_aux_head, # Replaces the hardcoded True     
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    val_m = ckpt.get("val_metrics", {})
    miou  = val_m.get("mean/iou", 0)
    print(f"Loaded  {ckpt_path}  (val mIoU {miou:.4f})")
    
    return model, cfg


# ── Single-image inference ────────────────────────────────────────────────────

@torch.inference_mode()
def predict_single(
    model:     torch.nn.Module,
    image_bgr: np.ndarray,
    cfg:       Config,
    device:    torch.device,
) -> np.ndarray:
    """
    Returns a (H, W, 7) float32 probability map in original image resolution.
    When cfg.use_tta=True, averages across 4 augmented views.
    """
    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    tfms = tta_transforms(cfg.image_size) if cfg.use_tta \
           else [val_transforms(cfg.image_size)]

    prob_sum = None

    for i, tfm in enumerate(tfms):
        tensor = tfm(image=image_rgb)["image"].unsqueeze(0).to(device)
        logits = model(tensor)                                 # (1, 9, h, w)
        prob   = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (9, h, w)

        # Un-flip before accumulating
        if i == 1:  prob = prob[:, :, ::-1].copy()   # h-flip
        if i == 2:  prob = prob[:, ::-1, :].copy()   # v-flip

        prob_sum = prob if prob_sum is None else prob_sum + prob

    prob_avg = prob_sum / len(tfms)  # (9, h, w)

    # Resize each channel back to original resolution
    result = np.stack([
        cv2.resize(prob_avg[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        for c in range(prob_avg.shape[0])
    ], axis=-1)   # (H, W, 9)

    return result.astype(np.float32)


# ── Post-processing helpers ───────────────────────────────────────────────────

def build_binary_masks(
    prob_map: np.ndarray,   # (H, W, 9)
    cfg:      Config,
) -> np.ndarray:
    """Returns (H, W, 7) uint8 binary mask (0 or 255)."""
    masks = np.zeros(prob_map.shape, dtype=np.uint8)
    for i, name in enumerate(cfg.class_names):
        thr = cfg.threshold_for(name)
        masks[:, :, i] = (prob_map[:, :, i] > thr).astype(np.uint8) * 255
    return masks


def build_overlay(
    image_bgr: np.ndarray,
    masks:     np.ndarray,   # (H, W, 7) uint8
    class_names: List[str],
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Renders all active damage classes as semi-transparent colour overlays.
    Overlapping classes blend additively (dent + scratch → both colours visible).
    """
    overlay = image_bgr.copy().astype(np.float32)
    blended = image_bgr.astype(np.float32)

    for i, name in enumerate(class_names):
        colour = CLASS_COLOURS_BGR[name]
        ch     = masks[:, :, i]
        if ch.max() == 0:
            continue
        colour_layer = np.zeros_like(image_bgr, dtype=np.float32)
        colour_layer[ch > 0] = colour
        # Only blend where this class is active
        mask_bool = ch > 0
        blended[mask_bool] = (
            (1 - alpha) * blended[mask_bool] +
            alpha * colour_layer[mask_bool]
        )

    return blended.clip(0, 255).astype(np.uint8)


def damage_summary(masks: np.ndarray, class_names: List[str]) -> dict:
    """Per-class damage percentage and severity label."""
    h, w    = masks.shape[:2]
    total_px = h * w
    per_class = {}
    for i, name in enumerate(class_names):
        pct = float(masks[:, :, i].sum()) / 255.0 / total_px * 100
        per_class[name] = {
            "damage_pct": round(pct, 3),
            "severity":   _severity(pct),
        }
    overall = float(np.any(masks > 0, axis=2).sum()) / total_px * 100
    return {
        "overall_damage_pct": round(overall, 3),
        "overall_severity":   _severity(overall),
        "per_class":          per_class,
    }


def _severity(pct: float) -> str:
    if pct < 0.5:  return "none"
    if pct < 3.0:  return "minor"
    if pct < 10.0: return "moderate"
    return "severe"


# ── Batch inference ───────────────────────────────────────────────────────────

def predict_folder(
    model:     torch.nn.Module,
    input_dir: Path,
    max_images: int,
    output_dir: Path,
    cfg:       Config,
    device:    torch.device,
) -> None:
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    for sub in ("masks", "overlays", "prob_maps"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    img_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not img_paths:
        raise FileNotFoundError(f"No images in {input_dir}")

    all_summaries = []
    shuffle(img_paths)
    img_paths = img_paths[:max_images] if max_images is not None else img_paths

    for img_path in img_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  skip (unreadable): {img_path.name}")
            continue

        prob_map = predict_single(model, bgr, cfg, device)  # (H,W,7)
        masks    = build_binary_masks(prob_map, cfg)          # (H,W,7)
        overlay  = build_overlay(bgr, masks, cfg.class_names)
        summary  = damage_summary(masks, cfg.class_names)
        summary["image"] = img_path.name

        # Save outputs
        stem = img_path.stem
        np.save(output_dir / "prob_maps" / f"{stem}_probs.npy", prob_map)
        cv2.imwrite(str(output_dir / "overlays" / f"{stem}_overlay.png"), overlay)
        # Save one PNG per class mask
        for i, name in enumerate(cfg.class_names):
            cv2.imwrite(
                str(output_dir / "masks" / f"{stem}_{name}.png"),
                masks[:, :, i],
            )

        all_summaries.append(summary)
        overall = summary["overall_damage_pct"]
        sev     = summary["overall_severity"]
        active  = [n for n in cfg.class_names
                   if summary["per_class"][n]["severity"] != "none"]
        print(f"  {img_path.name:<40s}  {overall:5.1f}%  [{sev}]  "
              f"classes: {', '.join(active) or 'none'}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nDone → {output_dir}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AxisDeepLabV3+ inference")
    p.add_argument("--input",      required=True)
    p.add_argument("--num_images", type=int, default=None, help="Limit number of images to process")
    p.add_argument("--output",     default="results")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--no-tta",     action="store_true")
    args = p.parse_args()

    cfg           = Config()
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg    = load_model(args.checkpoint, device)
    cfg.use_tta   = not args.no_tta

    predict_folder(model, Path(args.input), args.num_images, Path(args.output), cfg, device)
