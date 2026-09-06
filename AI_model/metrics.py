"""
metrics.py  —  AxisDeepLabV3+

Per-class evaluation metrics:
  • IoU  (Jaccard)
  • Dice (F1 on masks)
  • Boundary F1 — applies Sobel to both prediction and GT, then computes F1
    on edge pixels.  Specifically captures how well hairline scratches and
    panel edges are localised.

All metrics return per-class arrays so training logs can show which damage
types are improving and which need attention.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """x: (B, C, H, W) binary float -> edge magnitude map."""
    kx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    B, C, H, W = x.shape
    flat = x.reshape(B*C, 1, H, W)
    ex   = F.conv2d(flat, kx, padding=1)
    ey   = F.conv2d(flat, ky, padding=1)
    return (ex**2 + ey**2 + 1e-6).sqrt().reshape(B, C, H, W)

class MetricAccumulator:
    """
    Accumulates raw components globally across batches to prevent 
    absent-class biases from smoothing terms.
    """
    def __init__(self, class_names: List[str], threshold: float = 0.50, edge_thr: float = 0.10):
        self.class_names = class_names
        self.threshold   = threshold
        self.edge_thr    = edge_thr
        C = len(class_names)
        
        # Initialize global accumulation tracking
        self.register_buffers(C)
        self.reset()

    def register_buffers(self, C: int):
        # Keeps tracking tensors ready
        self._global_inter = torch.zeros(C)
        self._global_union = torch.zeros(C)
        self._global_total = torch.zeros(C)
        self._global_tp    = torch.zeros(C)
        self._global_fp    = torch.zeros(C)
        self._global_fn    = torch.zeros(C)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        device = logits.device
        self._global_inter = self._global_inter.to(device)
        self._global_union = self._global_union.to(device)
        self._global_total = self._global_total.to(device)
        self._global_tp    = self._global_tp.to(device)
        self._global_fp    = self._global_fp.to(device)
        self._global_fn    = self._global_fn.to(device)

        preds = (torch.sigmoid(logits) > self.threshold).float()
        B, C  = preds.shape[:2]
        
        p = preds.view(B, C, -1)
        t = targets.view(B, C, -1)
        
        # Core Segmentation Components
        inter = (p * t).sum(dim=(0, 2))
        p_sum = p.sum(dim=(0, 2))
        t_sum = t.sum(dim=(0, 2))
        
        self._global_inter += inter
        self._global_union += (p_sum + t_sum - inter)
        self._global_total += (p_sum + t_sum)
        
        # Boundary Components
        edge_pred = (_sobel_edges(preds)   > self.edge_thr).float()
        edge_gt   = (_sobel_edges(targets) > self.edge_thr).float()
        ep = edge_pred.view(B, C, -1)
        eg = edge_gt.view(B, C, -1)
        
        self._global_tp += (ep * eg).sum(dim=(0, 2))
        self._global_fp += (ep * (1 - eg)).sum(dim=(0, 2))
        self._global_fn += ((1 - ep) * eg).sum(dim=(0, 2))

    def compute(self, smooth: float = 1e-6) -> Dict[str, float]:
        # Divide globally at the end of the epoch
        iou  = ((self._global_inter + smooth) / (self._global_union + smooth)).cpu().numpy()
        dice = ((2 * self._global_inter + smooth) / (self._global_total + smooth)).cpu().numpy()
        
        prec = (self._global_tp + smooth) / (self._global_tp + self._global_fp + smooth)
        rec  = (self._global_tp + smooth) / (self._global_tp + self._global_fn + smooth)
        bf1  = (2 * prec * rec / (prec + rec + smooth)).cpu().numpy()

        result: Dict[str, float] = {}
        for i, name in enumerate(self.class_names):
            result[f"{name}/iou"]         = float(iou[i])
            result[f"{name}/dice"]        = float(dice[i])
            result[f"{name}/boundary_f1"] = float(bf1[i])

        result["mean/iou"]         = float(iou.mean())
        result["mean/dice"]        = float(dice.mean())
        result["mean/boundary_f1"] = float(bf1.mean())
        return result

    def reset(self) -> None:
        self._global_inter.zero_()
        self._global_union.zero_()
        self._global_total.zero_()
        self._global_tp.zero_()
        self._global_fp.zero_()
        self._global_fn.zero_()


def format_metrics(metrics: Dict[str, float], class_names: List[str]) -> str:
    lines = []
    for name in class_names:
        iou  = metrics.get(f"{name}/iou",         0)
        dice = metrics.get(f"{name}/dice",        0)
        bf1  = metrics.get(f"{name}/boundary_f1", 0)
        lines.append(f"  {name:<20s} IoU {iou:.3f}  Dice {dice:.3f}  BndF1 {bf1:.3f}")
    miou  = metrics.get("mean/iou",         0)
    mdice = metrics.get("mean/dice",        0)
    mbf1  = metrics.get("mean/boundary_f1", 0)
    lines.append(f"  {'MEAN':<20s} IoU {miou:.3f}  Dice {mdice:.3f}  BndF1 {mbf1:.3f}")
    return "\n".join(lines)