"""Pipeline inference: segmentasi daun → lesi → severity → klasifikasi."""

import torch
import numpy as np

from utils.preprocessing import (
    letterbox_resize,
    postprocess_leaf_mask,
    apply_leaf_mask,
    prepare_segmentation_tensor,
    prepare_classification_tensor,
    overlay_mask,
    pil_to_rgb_array,
    SEG_IMG_SIZE,
)
from utils.severity import compute_severity, is_healthy, severity_category
from utils.model_loader import CLASS_NAMES, CLASS_DISPLAY_NAMES

LEAF_THRESHOLD = 0.45
LESION_THRESHOLD = 0.40


@torch.no_grad()
def predict_leaf_mask(leaf_model, image_256, device):
    """Prediksi mask daun pada citra 256x256."""
    tensor = prepare_segmentation_tensor(image_256).unsqueeze(0).to(device)
    logits = leaf_model(tensor)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (prob >= LEAF_THRESHOLD).astype(np.uint8)
    return postprocess_leaf_mask(mask), prob


@torch.no_grad()
def predict_lesion_mask(lesion_model, masked_image, device):
    """Prediksi mask lesi pada citra daun masked (256x256)."""
    tensor = prepare_segmentation_tensor(masked_image).unsqueeze(0).to(device)
    logits = lesion_model(tensor)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (prob >= LESION_THRESHOLD).astype(np.uint8)
    return mask, prob


@torch.no_grad()
def predict_disease(classifier_model, masked_image, device):
    """Klasifikasi penyakit dari citra masked."""
    tensor = prepare_classification_tensor(masked_image).unsqueeze(0).to(device)
    logits = classifier_model(tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(np.argmax(probabilities))

    return {
        "pred_idx": pred_idx,
        "pred_class": CLASS_NAMES[pred_idx],
        "pred_class_display": CLASS_DISPLAY_NAMES[CLASS_NAMES[pred_idx]],
        "confidence": float(probabilities[pred_idx]),
        "probabilities": {
            CLASS_DISPLAY_NAMES[name]: float(probabilities[idx])
            for idx, name in enumerate(CLASS_NAMES)
        },
    }


def run_diagnosis(image_input, models):
    """
    Jalankan pipeline diagnosis lengkap.

    Returns dict berisi citra visualisasi, mask, severity, dan hasil klasifikasi.
    """
    device = models["device"]
    leaf_model = models["leaf"]
    lesion_model = models["lesion"]
    classifier_model = models["classifier"]

    original_rgb = pil_to_rgb_array(image_input)
    image_256 = letterbox_resize(original_rgb, size=SEG_IMG_SIZE)

    leaf_mask, leaf_prob = predict_leaf_mask(leaf_model, image_256, device)
    masked_image = apply_leaf_mask(image_256, leaf_mask)

    lesion_mask_raw, lesion_prob = predict_lesion_mask(
        lesion_model, masked_image, device
    )
    lesion_mask = (lesion_mask_raw & leaf_mask).astype(np.uint8)

    severity = compute_severity(lesion_mask, leaf_mask)
    category = severity_category(severity)
    healthy = is_healthy(severity)

    result = {
        "original_rgb": original_rgb,
        "image_256": image_256,
        "leaf_mask": leaf_mask,
        "leaf_prob": leaf_prob,
        "masked_image": masked_image,
        "lesion_mask": lesion_mask,
        "lesion_prob": lesion_prob,
        "severity": severity,
        "severity_category": category,
        "is_healthy": healthy,
        "leaf_overlay": overlay_mask(image_256, leaf_mask, color=(0, 200, 0)),
        "lesion_overlay": overlay_mask(image_256, lesion_mask, color=(255, 60, 60)),
        "classification": None,
    }

    if not healthy:
        result["classification"] = predict_disease(
            classifier_model, masked_image, device
        )

    return result
