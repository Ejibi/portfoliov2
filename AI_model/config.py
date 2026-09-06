"""
config.py  —  AxisDeepLabV3+
Single source of truth for every hyperparameter and path.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# ── Class registry  (channel index = list position) ──────────────────────────
CLASSES: List[str] = [
    "Dent",               # 0
    "Scratch",            # 1
    "Corrosion",          # 2
    "Cracked_Glass",      # 3
    "Shattered_Glass",    # 4
    "Panel_Misalignment",   # 5
    "Fragmentation",      # 6
    "Car_Inner_texture",  # 7
    "Panel_Crumpling"     # 8
]

# BGR colours for OpenCV overlay  (one per class)
CLASS_COLOURS_BGR: Dict[str, Tuple[int,int,int]] = {
    "Dent":             (  0, 100, 220),
    "Scratch":          (200, 100,  10),
    "Corrosion":        ( 20,  80, 160),
    "Cracked_Glass":    (200, 200,   0),
    "Shattered_Glass":  (  0, 220, 220),
    "Panel_Misalignment": (180,  40, 130),
    "Fragmentation":    ( 40, 180,  40),
    "Car_Inner_texture": (  0, 100,  0),
    "Panel_Crumpling":  (100,   0, 100)
}


@dataclass
class Config:
    project_name: str = "AxisDeepLabV3Plus"
    # ── Paths ─────────────────────────────────────────────────────────────────
    data_root:       Path = Path("dataset")
    checkpoint_dir:  Path = Path("checkpoints")
    log_dir:         Path = Path("logs")

    # Roboflow COCO-JSON export locations
    train_json:    Path = Path("dataset/train/_annotations.coco.json")
    val_json:      Path = Path("dataset/valid/_annotations.coco.json")
    train_img_dir: Path = Path("dataset/train")
    val_img_dir:   Path = Path("dataset/valid")
    train_mask_dir: Path = Path("dataset/train_masks")
    val_mask_dir:   Path = Path("dataset/valid_masks")

    # ── Class info ────────────────────────────────────────────────────────────
    class_names: List[str] = field(default_factory=lambda: CLASSES)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    # ── Model ─────────────────────────────────────────────────────────────────
    encoder_name:    str   = "tu-convnextv2_tiny"   # GRN + FCMAE pretraining
    encoder_weights: str   = "fcmae"                # MAE-pretrained weights
    in_channels:     int   = 3
    use_aux_head:    bool  = True
    aux_weight:      float = 0.40
    decoder_dropout: float = 0.10

    # ── Input ─────────────────────────────────────────────────────────────────
    image_size: Tuple[int,int] = (512, 512)

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:  int = 8
    num_workers: int = 2
    grad_accum_steps: int = 4
    epochs:      int = 120
    seed:        int = 42

    # ── Optimiser  (differential LRs: backbone 10× smaller than head) ─────────
    lr_backbone:  float = 3e-5
    lr_head:      float = 3e-4
    weight_decay: float = 1e-4

    # ── Loss ──────────────────────────────────────────────────────────────────
    dice_weight:     float = 0.45
    bce_weight:      float = 0.35
    edge_weight:     float = 0.20
    ohem_keep_ratio: float = 0.35
    label_smoothing: float = 0.00
    pos_weight: list = field(default_factory=lambda: [
    1.5,   # Dent               (973) — very common
    1.5,   # Scratch            (672) — common
    12.0,  # Corrosion           (39) — rare, max weight
    12.0,  # Cracked_Glass       (35) — rarest, max weight
    4.0,   # Shattered_Glass    (241) — medium
    1.5,   # Panel_Misalignment (927) — very common
    1.5,   # Fragmentation      (612) — common
    2.2,   # Car_Inner_texture  (373) — slightly under-represented
    1.5,   # Panel_Crumpling    (683) — common
    ])
    
    # Focal Tversky Loss parameters ──────────────────────────────────────────────────────────────────
    focal_tversky_alpha: float = 0.25
    focal_tversky_beta:  float = 0.75
    focal_tversky_gamma: float = 0.75
    smooth:     float = 1.0
    

    # ── LR schedule ───────────────────────────────────────────────────────────
    warmup_epochs: int   = 6
    min_lr:        float = 3e-6

    # ── Regularisation ────────────────────────────────────────────────────────
    grad_clip: float = 1.0
    use_amp:   bool  = True
    use_ema:   bool  = True
    ema_decay: float = 0.99

    # ── Inference ─────────────────────────────────────────────────────────────
    use_tta:        bool  = True
    pred_threshold: float = 0.50
    # Per-class threshold overrides (tune on val after training)
    # e.g. {"scratch": 0.40, "corrosion": 0.45}
    class_thresholds: Dict[str, float] = field(default_factory=lambda: {
    "Corrosion":       0.35,
    "Cracked_Glass":   0.35,
    "Shattered_Glass": 0.45,
    })

    def threshold_for(self, cls: str) -> float:
        return self.class_thresholds.get(cls, self.pred_threshold)
