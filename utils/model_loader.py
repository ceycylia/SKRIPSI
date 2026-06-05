"""Load arsitektur model dan checkpoint .pth."""

from pathlib import Path

import torch
import torch.nn as nn
import timm
import segmentation_models_pytorch as smp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

LEAF_MODEL_PATH = MODELS_DIR / "model_segmentasi_daun_efficientnetb0_best.pth"
LESION_MODEL_PATH = MODELS_DIR / "model_segmentasi_lesi_efficientnetb0_best.pth"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "klasifikasi_penyakit_final_best.pth"

ENCODER_NAME = "efficientnet-b0"
CLASS_NAMES = ["Black_Measles", "Black_Rot", "Isariopsis_Leaf_Spot"]
CLASS_DISPLAY_NAMES = {
    "Black_Measles": "Black Measles",
    "Black_Rot": "Black Rot",
    "Isariopsis_Leaf_Spot": "Isariopsis Leaf Spot",
}
NUM_CLASSES = len(CLASS_NAMES)
DROPOUT = 0.50


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_unet(encoder_weights=None):
    """U-Net EfficientNet-B0 (notebook segmentasi daun & lesi)."""
    return smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )


class EfficientNetB0Classifier(nn.Module):
    """Classifier EfficientNet-B0 (notebook klasifikasi)."""

    def __init__(self, num_classes, dropout):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, images):
        features = self.backbone(images)
        return self.classifier(features)


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_state_dict(model, checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def load_leaf_model(device=None):
    device = device or get_device()
    model = build_unet(encoder_weights=None).to(device)
    checkpoint = _load_checkpoint(LEAF_MODEL_PATH, device)
    _load_state_dict(model, checkpoint)
    model.eval()
    return model


def load_lesion_model(device=None):
    device = device or get_device()
    model = build_unet(encoder_weights=None).to(device)
    checkpoint = _load_checkpoint(LESION_MODEL_PATH, device)
    _load_state_dict(model, checkpoint)
    model.eval()
    return model


def load_classifier_model(device=None):
    device = device or get_device()
    model = EfficientNetB0Classifier(NUM_CLASSES, DROPOUT).to(device)
    checkpoint = _load_checkpoint(CLASSIFIER_MODEL_PATH, device)
    _load_state_dict(model, checkpoint)
    model.eval()
    return model


@torch.no_grad()
def load_all_models(device=None):
    """Load ketiga model sekaligus."""
    device = device or get_device()
    return {
        "device": device,
        "leaf": load_leaf_model(device),
        "lesion": load_lesion_model(device),
        "classifier": load_classifier_model(device),
    }
