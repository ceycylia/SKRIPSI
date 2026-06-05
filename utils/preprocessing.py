"""Preprocessing citra sesuai notebook training."""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms

# Konstanta dari notebook
SEG_IMG_SIZE = 256
CLS_IMG_SIZE = 224
LEAF_PAD_VALUE = 255
MASKED_BG_VALUE = 0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

seg_val_transform = A.Compose([
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

cls_eval_transform = transforms.Compose([
    transforms.Resize((CLS_IMG_SIZE, CLS_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
])


def letterbox_resize(image, mask=None, size=SEG_IMG_SIZE, image_pad=LEAF_PAD_VALUE):
    """Letterbox resize dengan padding putih (notebook segmentasi daun/lesi)."""
    h, w = image.shape[:2]

    scale = min(size / w, size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    image_canvas = np.full((size, size, 3), image_pad, dtype=np.uint8)

    top = (size - new_h) // 2
    left = (size - new_w) // 2
    image_canvas[top : top + new_h, left : left + new_w] = resized_image

    if mask is None:
        return image_canvas

    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    mask_canvas = np.zeros((size, size), dtype=np.uint8)
    mask_canvas[top : top + new_h, left : left + new_w] = resized_mask

    return image_canvas, mask_canvas


def fill_holes(mask):
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape

    padded = np.pad(mask, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    flood = padded.copy()
    flood_mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 1)

    holes = 1 - flood
    filled = np.clip(padded + holes, 0, 1)

    return filled[1:-1, 1:-1].astype(np.uint8)


def count_components(mask):
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    return max(num_labels - 1, 0), labels, stats, centroids


def select_leaf_component(mask, min_area_ratio=0.002):
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape
    center = np.array([w / 2, h / 2])

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    candidates = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        area_ratio = area / (h * w)

        if area_ratio < min_area_ratio:
            continue

        cx, cy = centroids[label]
        dist = np.linalg.norm(np.array([cx, cy]) - center) / np.linalg.norm(center)
        score = area_ratio - 0.35 * dist
        candidates.append((score, label))

    if len(candidates) == 0:
        return mask

    _, best_label = max(candidates, key=lambda x: x[0])
    return (labels == best_label).astype(np.uint8)


def clean_leaf_mask(mask):
    mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = fill_holes(mask)
    return mask.astype(np.uint8)


def postprocess_leaf_mask(mask):
    """Post-processing mask daun (notebook segmentasi lesi)."""
    mask = (mask > 0).astype(np.uint8)
    n_components, _, _, _ = count_components(mask)

    if n_components <= 1:
        return mask

    mask = select_leaf_component(mask)
    mask = clean_leaf_mask(mask)
    return mask.astype(np.uint8)


def apply_leaf_mask(image, leaf_mask, bg_value=MASKED_BG_VALUE):
    """Buat citra daun dengan background hitam."""
    leaf_mask_bool = leaf_mask > 0
    result = np.full_like(image, bg_value, dtype=np.uint8)
    result[leaf_mask_bool] = image[leaf_mask_bool]
    return result


def pil_to_rgb_array(image):
    """Konversi PIL Image ke numpy RGB."""
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    return image


def prepare_segmentation_tensor(image_rgb):
    """Siapkan tensor untuk model segmentasi (256x256, sudah letterbox)."""
    transformed = seg_val_transform(image=image_rgb)
    return transformed["image"]


def prepare_classification_tensor(masked_image_rgb):
    """Siapkan tensor untuk model klasifikasi (224x224)."""
    pil_image = Image.fromarray(masked_image_rgb)
    return cls_eval_transform(pil_image)


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.45):
    """Overlay mask berwarna pada citra untuk visualisasi."""
    overlay = image.copy()
    mask_bool = mask > 0
    color_arr = np.array(color, dtype=np.uint8)
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * color_arr
    ).astype(np.uint8)
    return overlay
