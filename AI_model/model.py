"""
model.py  —  AxisDeepLabV3+

Architecture decisions:
  • ConvNeXt-V2-tiny backbone  — Global Response Normalisation (GRN) gives
    each spatial location awareness of the full-image channel distribution,
    partially addressing the local-receptive-field limitation of CNNs.
    FCMAE pretraining builds more holistic representations than ImageNet.

  • 7 independent output channels, activation=None  — BCEWithLogitsLoss
    treats each channel as a separate binary classifier.  Overlapping classes
    share pixels legally; no argmax or softmax is ever applied.

  • Auxiliary deep-supervision head on stage-3 (stride 16, 384 ch) —
    injects gradient signal into mid-network layers, acting as a regulariser
    and accelerating feature convergence on small datasets.

  • Dropout before the final classifier — reduces overconfidence on the
    relatively small car-damage datasets typically available (~10k images).

  • parameter_groups() exposes backbone vs head parameters separately so
    AdamW can apply a 10× smaller LR to the pretrained backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Dict, List, Tuple, Union


class AuxHead(nn.Module):
    """
    Lightweight auxiliary classification head attached to the encoder's
    second-to-last feature map (stride 16 for ConvNeXt-V2-tiny = 384 ch).
    Active only during training.
    """
    def __init__(self, in_ch: int, num_classes: int, dropout: float = 0.10):
        super().__init__()
        mid = in_ch // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class AxisDeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with a ConvNeXt-V2 backbone configured for 9-class
    multi-label (sigmoid, not softmax) car-damage segmentation.
    """

    def __init__(
        self,
        encoder_name:    str   = "tu-convnextv2_tiny",
        encoder_weights: str   = "fcmae",
        num_classes:     int   = 9,
        use_aux_head:    bool  = True,
        dropout:         float = 0.10,
    ):
        super().__init__()
        self.use_aux_head = use_aux_head

        # ── Main DeepLabV3+ ───────────────────────────────────────────────────
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,     # raw logits — sigmoid applied in loss / predict
        )

        # Inject dropout before the final 1×1 conv in the segmentation head
        self._add_head_dropout(dropout)

        # ── Auxiliary head ────────────────────────────────────────────────────
        if use_aux_head:
            # encoder.out_channels = [3, 96, 192, 384, 768] for convnextv2_tiny
            # index -2 = stage-3 = 384 channels
            aux_in_ch = self.model.encoder.out_channels[-2]
            self.aux_head = AuxHead(aux_in_ch, num_classes, dropout)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _add_head_dropout(self, p: float) -> None:
        seg_head = self.model.segmentation_head
        self.model.segmentation_head = nn.Sequential(
            nn.Dropout2d(p=p), *list(seg_head.children())
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Training  → (main_logits, aux_logits)  both at input resolution.
        Inference → main_logits only.
        """
        features       = self.model.encoder(x)
        decoder_out    = self.model.decoder(features)
        main_logits    = self.model.segmentation_head(decoder_out)

        if self.training and self.use_aux_head:
            aux_logits = self.aux_head(features[-2])
            aux_logits = F.interpolate(
                aux_logits, size=x.shape[2:],
                mode="bilinear", align_corners=False,
            )
            return main_logits, aux_logits

        return main_logits

    # ── Parameter groups ──────────────────────────────────────────────────────

    def parameter_groups(
        self,
        lr_backbone:  float,
        lr_head:      float,
        weight_decay: float,
    ) -> List[Dict]:
        """
        Returns two AdamW parameter groups:
          group 0 — encoder (pretrained backbone) → lr_backbone
          group 1 — decoder + heads (random init)  → lr_head
        """
        backbone_p, head_p = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "model.encoder" in name:
                backbone_p.append(param)
            else:
                head_p.append(param)
        return [
            {"params": backbone_p, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head_p,     "lr": lr_head,     "weight_decay": weight_decay},
        ]
