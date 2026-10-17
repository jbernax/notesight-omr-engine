"""
Runtime safety patches for homr.staff_parsing.

Prevents OpenCV cv2.resize zero-dimension assertions and inverted/negative
staff bounding-box regions when processing degraded or tightly spaced scans.

Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
"""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def safe_calculate_region(staff: Any, regions: Any) -> np.ndarray:
    """Calculate staff bounding region with guaranteed positive width and height."""
    x_min = staff.min_x - 2 * staff.average_unit_size
    x_max = staff.max_x + 2 * staff.average_unit_size
    y_min = max(
        staff.min_y - 4 * staff.average_unit_size,
        regions.get_start_of_closest_staff_above(staff.min_y),
    )
    y_max = min(
        staff.max_y + 4 * staff.average_unit_size,
        regions.get_start_of_closest_staff_below(staff.max_y),
    )
    if y_max <= y_min:
        y_min = staff.min_y - 4 * staff.average_unit_size
        y_max = max(y_min + 1, staff.max_y + 4 * staff.average_unit_size)
    if x_max <= x_min:
        x_min = staff.min_x
        x_max = max(x_min + 1, staff.max_x + 2 * staff.average_unit_size)
    return np.array([int(x_min), int(y_min), max(int(x_min) + 1, int(x_max)), max(int(y_min) + 1, int(y_max))])


def safe_get_tr_omr_canvas_size(
    image_shape: tuple[int, ...],
    margin_top: int = 0,
    margin_bottom: int = 0,
    max_height: int = 192,
    max_width: int = 1280,
) -> np.ndarray:
    """Compute TrOMR target canvas dimensions while clamping zero/negative inputs to >= 1 px."""
    try:
        from homr.transformer.configs import default_config
        max_height = int(default_config.max_height)
        max_width = int(default_config.max_width)
    except Exception:
        pass

    tr_omr_max_height_with_margin = max(1, max_height - margin_top - margin_bottom)
    tr_omr_ratio = float(tr_omr_max_height_with_margin) / float(max(1, max_width))

    height, width = image_shape[:2]
    safe_height = max(1, int(height))
    safe_width = max(1, int(width))

    if safe_height / safe_width > tr_omr_ratio:
        new_shape = [
            max(1, int(safe_width / safe_height * tr_omr_max_height_with_margin)),
            tr_omr_max_height_with_margin,
        ]
    else:
        new_shape = [
            max_width,
            max(1, int(safe_height / safe_width * max_width)),
        ]
    return np.array(new_shape)


def safe_center_image_on_canvas(
    image: np.ndarray,
    canvas_size: np.ndarray,
    margin_top: int = 0,
    margin_bottom: int = 0,
) -> np.ndarray:
    """Center a cropped staff image on a fixed TrOMR canvas without failing on empty slices."""
    import homr.staff_parsing as sp

    tr_omr_max_height = int(sp.tr_omr_max_height)
    tr_omr_max_width = int(sp.tr_omr_max_width)
    is_grayscale = image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)

    if is_grayscale:
        new_image = np.full((tr_omr_max_height, tr_omr_max_width), 255, dtype=np.uint8)
    else:
        new_image = np.full((tr_omr_max_height, tr_omr_max_width, 3), 255, dtype=np.uint8)

    if image.shape[0] <= 0 or image.shape[1] <= 0:
        return new_image

    safe_target = (
        min(tr_omr_max_width, max(1, int(canvas_size[0]))),
        min(tr_omr_max_height, max(1, int(canvas_size[1]))),
    )
    resized = cv2.resize(image, safe_target)

    x_offset = 0
    tr_omr_max_height_with_margin = tr_omr_max_height - margin_top - margin_bottom
    y_offset = max(0, (tr_omr_max_height_with_margin - resized.shape[0]) // 2 + margin_top)

    end_y = min(tr_omr_max_height, y_offset + resized.shape[0])
    end_x = min(tr_omr_max_width, x_offset + resized.shape[1])
    new_image[y_offset:end_y, x_offset:end_x] = resized[: end_y - y_offset, : end_x - x_offset]
    return new_image


def apply_homr_safety_patches() -> bool:
    """Bind runtime safety functions onto homr.staff_parsing in memory."""
    try:
        import homr.staff_parsing as sp

        sp._calculate_region = safe_calculate_region  # type: ignore[attr-defined]
        sp.get_tr_omr_canvas_size = safe_get_tr_omr_canvas_size  # type: ignore[assignment]
        sp.center_image_on_canvas = safe_center_image_on_canvas  # type: ignore[assignment]
        return True
    except Exception:
        return False
