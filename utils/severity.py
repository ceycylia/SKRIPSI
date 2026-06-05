"""Perhitungan severity dan kategori keparahan."""

HEALTHY_THRESHOLD = 0.5


def compute_severity(lesion_mask, leaf_mask):
    """Hitung severity (%) = piksel lesi / piksel daun * 100."""
    leaf_pixels = int((leaf_mask > 0).sum())
    if leaf_pixels == 0:
        return 0.0

    lesion_pixels = int(((lesion_mask > 0) & (leaf_mask > 0)).sum())
    return lesion_pixels / leaf_pixels * 100.0


def is_healthy(severity):
    return severity <= HEALTHY_THRESHOLD


def severity_category(severity):
    """Kategori keparahan sesuai spesifikasi sistem."""
    if severity <= HEALTHY_THRESHOLD:
        return "Sehat"
    if severity <= 25.0:
        return "Ringan"
    if severity <= 50.0:
        return "Sedang"
    return "Berat"
