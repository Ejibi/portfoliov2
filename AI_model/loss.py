"""
loss.py  —  AxisDeepLabV3+

Three-component loss for 7-class multi-label segmentation of car damage:

  1. Per-class Focal Tversky  (dice_weight=0.45)
       Computes Dice independently for each of the 7 channels, then averages.
       Rare classes (panel_separation, fragmentation) get equal gradient weight
       to common ones (scratch) — critical for a long-tail damage distribution.

  2. OHEM BCE  (bce_weight=0.35)
       Per-pixel BCE loss sorted by difficulty; only the hardest `keep_ratio`
       fraction of pixels contribute to the backward pass.  Boundary pixels
       and hairline scratches naturally rise to the top of the difficulty
       ranking, concentrating gradient exactly where precision matters.

  3. Sobel Edge Loss  (edge_weight=0.20)
       Applies Sobel edge detection to both predicted probability maps and
       ground-truth masks, then penalises the L1 difference.  Makes boundary
       quality an explicit optimisation target rather than a side effect.

  DeepSupervisionLoss wraps the above to accept the
  (main_logits, aux_logits) tuple from the model during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union


# ── 1. Per-class Dice ─────────────────────────────────────────────────────────

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for highly imbalanced multi-label segmentation.
    alpha: weight of False Positives
    beta: weight of False Negatives (Set beta > alpha to penalize missed damages more)
    gamma: Focal parameter to focus on hard examples
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        B, C = probs.shape[:2]
        
        # Flatten spatial dimensions to compute per-image, per-class
        p = probs.view(B, C, -1)
        t = targets.view(B, C, -1)
        
        # Calculate True Positives, False Positives, False Negatives
        TP = (p * t).sum(dim=2)
        FP = (p * (1 - t)).sum(dim=2)
        FN = ((1 - p) * t).sum(dim=2)
        
        # Tversky Index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        # Focal Adjustment
        focal_tversky = (1.0 - tversky) ** self.gamma
        
        return focal_tversky.mean()


# ── 2. OHEM BCE ───────────────────────────────────────────────────────────────

class OHEMBCELoss(nn.Module):
    def __init__(self, keep_ratio: float = 0.70, label_smoothing: float = 0.05, pos_weight: list = None):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.eps = label_smoothing
        # Register pos_weight as a buffer so it moves to the GPU automatically
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(pos_weight))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        t = targets * (1 - self.eps) + (1 - targets) * self.eps

        pw = self.pos_weight.view(1, C, 1, 1).to(logits.device) if self.pos_weight is not None else None

        per_pixel = F.binary_cross_entropy_with_logits(
            logits, t, reduction="none", pos_weight=pw
        )

        flat_loss = per_pixel.view(B, C, -1)  # (B, C, H*W)
        total_pixels = H * W
        k = max(1, int(total_pixels * self.keep_ratio))
        
        loss_per_class = []
        for c in range(C):
            difficulty = flat_loss[:, c, :]  # (B, H*W)
            
            # Find the threshold for the hardest `k` pixels per image
            threshold = difficulty.topk(k, dim=1).values[:, -1].detach()  # (B,)
            hard_mask = (difficulty >= threshold.unsqueeze(1)).float()  # (B, H*W)
            hard_mask = hard_mask.view(B, 1, H, W)  # (B, 1, H, W)

            masked_loss = (per_pixel[:, c:c+1, :, :] * hard_mask).sum()
            num_selected = hard_mask.sum() + 1e-6
            loss_class = masked_loss / num_selected
            loss_per_class.append(loss_class)

        return torch.stack(loss_per_class).mean()


# ── 3. Sobel Edge Loss ────────────────────────────────────────────────────────

class SobelEdgeLoss(nn.Module):
    """
    Differentiable edge loss.  Applies Sobel filters to both the predicted
    probability map and the ground-truth mask (per class channel), then
    computes L1 loss on the resulting edge magnitude maps.
    """
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])
        ky = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]])
        # (1, 1, 3, 3) — applied depthwise per class channel
        self.register_buffer("kx", kx.view(1,1,3,3))
        self.register_buffer("ky", ky.view(1,1,3,3))

    def _edge_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        flat = x.reshape(B * C, 1, H, W)
            
        #Dynamically cast to BOTH the device (CUDA) and precision (float16)
        ex   = F.conv2d(flat, self.kx.to(device=flat.device, dtype=flat.dtype), padding=1)
        ey   = F.conv2d(flat, self.ky.to(device=flat.device, dtype=flat.dtype), padding=1)
            
        mag  = torch.sqrt(ex**2 + ey**2 + 1e-6)
        return mag.reshape(B, C, H, W)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs        = torch.sigmoid(logits)
        edge_pred    = self._edge_magnitude(probs)
        edge_target  = self._edge_magnitude(targets)
        return F.l1_loss(edge_pred, edge_target)


# ── Combined loss ─────────────────────────────────────────────────────────────

class AxisSegLoss(nn.Module):
    """
    Full training loss for AxisDeepLabV3+:
        total = dice_w * PerClassDice
              + bce_w  * OHEMBCE
              + edge_w * SobelEdge
    """
    def __init__(
        self,
        dice_weight:     float = 0.45,
        bce_weight:      float = 0.35,
        edge_weight:     float = 0.20,
        ohem_keep_ratio: float = 0.40,
        label_smoothing: float = 0.05,
        smooth:     float = 1.0,
        focal_tversky_alpha: float = 0.3,
        focal_tversky_beta:  float = 0.7,
        focal_tversky_gamma: float = 0.75,
        pos_weight: list = [10.0]*9
    ):
        super().__init__()
        self.w_dice = dice_weight
        self.w_bce  = bce_weight
        self.w_edge = edge_weight

        self.dice = FocalTverskyLoss(alpha=focal_tversky_alpha, beta=focal_tversky_beta, gamma=focal_tversky_gamma, smooth=smooth)
        self.bce  = OHEMBCELoss(keep_ratio=ohem_keep_ratio,
                                 label_smoothing=label_smoothing, pos_weight=pos_weight)
        self.edge = SobelEdgeLoss()

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        d = self.dice(logits, targets)
        b = self.bce(logits,  targets)
        e = self.edge(logits, targets)
        total = self.w_dice * d + self.w_bce * b + self.w_edge * e
        return total, {"dice": d.item(), "bce": b.item(), "edge": e.item()}


# ── Deep supervision wrapper ──────────────────────────────────────────────────

class DeepSupervisionLoss(nn.Module):
    """
    Handles the (main_logits, aux_logits) tuple returned by AxisDeepLabV3Plus
    during training, and plain tensor during inference.

        total = base_loss(main) + aux_weight * base_loss(aux)
    """
    def __init__(self, base_loss: nn.Module, aux_weight: float = 0.40):
        super().__init__()
        self.base = base_loss
        self.w    = aux_weight

    def forward(
        self,
        output:  Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        if isinstance(output, (tuple, list)):
            main, aux      = output
            total_m, info  = self.base(main, targets)
            total_a, _     = self.base(aux,  targets)
            total          = total_m + self.w * total_a
            return total, info
        return self.base(output, targets)
