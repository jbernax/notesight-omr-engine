"""
Standalone homr transcription runner with staff bounding-box layout capture.

Writes <image>.musicxml and <image>.layout.json containing the staff boxes of every
system in original page pixel coordinates.

Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import numpy as np

from notesight_omr_engine.patches import apply_homr_safety_patches


def transcribe_image(image_path: Path | str, filter_narrow_staffs: bool | None = None) -> Path:
    """
    Run homr OMR inference on a single page image and write both <image>.musicxml
    and <image>.layout.json.

    Args:
        image_path: Path to the input page image (.png, .jpg, etc.).
        filter_narrow_staffs: If True, discard staff fragments < 40% of median staff width.
            Defaults to checking OMR_FILTER_NARROW_STAFFS environment variable if None.

    Returns:
        Path to the generated .musicxml file.
    """
    apply_homr_safety_patches()

    import homr.main as homr_main
    import homr.staff_detection as staff_detection
    import homr.staff_parsing as staff_parsing
    from homr.main import ProcessingConfig, XmlGeneratorArguments, process_image

    target_image = Path(image_path)
    if filter_narrow_staffs is None:
        filter_narrow_staffs = os.getenv("OMR_FILTER_NARROW_STAFFS", "false").lower() in ("1", "true")

    frame: dict[str, float] = {"x": 0.0, "y": 0.0, "crop_w": 0.0, "crop_h": 0.0}
    layout: dict[str, Any] = {"systems": []}

    original_detect_staff = staff_detection.detect_staff
    original_homr_detect_staff = getattr(homr_main, "detect_staff", original_detect_staff)
    original_autocrop = homr_main.autocrop
    original_parse_staffs = homr_main.parse_staffs

    def _filter_narrow_staffs_wrapper(*args: Any, **kwargs: Any) -> Any:
        staffs = original_detect_staff(*args, **kwargs)
        if staffs and len(staffs) > 1:
            median_width = float(np.median([s.max_x - s.min_x for s in staffs]))
            staffs = [s for s in staffs if (s.max_x - s.min_x) > 0.4 * median_width]
        return staffs

    def _recording_autocrop(img: Any) -> Any:
        cropped = original_autocrop(img)
        if cropped is not img and hasattr(cropped, "__array_interface__") and hasattr(img, "__array_interface__"):
            offset = cropped.__array_interface__["data"][0] - img.__array_interface__["data"][0]
            y, remainder = divmod(offset, img.strides[0])
            frame["x"], frame["y"] = float(remainder // img.strides[1]), float(y)
        frame["crop_w"], frame["crop_h"] = float(cropped.shape[1]), float(cropped.shape[0])
        return cropped

    def _recording_parse_staffs(debug: Any, staffs: Any, image: Any, *args: Any, **kwargs: Any) -> Any:
        systems = staff_parsing._ensure_same_number_of_staffs(staffs, image)
        sx = frame["crop_w"] / image.shape[1] if frame["crop_w"] else 1.0
        sy = frame["crop_h"] / image.shape[0] if frame["crop_h"] else 1.0
        for system in systems:
            layout["systems"].append(
                {
                    "staves": [
                        {
                            "part": part_index,
                            "min_x": float(frame["x"] + staff.min_x * sx),
                            "max_x": float(frame["x"] + staff.max_x * sx),
                            "min_y": float(frame["y"] + staff.min_y * sy),
                            "max_y": float(frame["y"] + staff.max_y * sy),
                            "is_grandstaff": bool(staff.is_grandstaff),
                        }
                        for part_index, staff in enumerate(system.staffs)
                    ]
                }
            )
        return original_parse_staffs(debug, staffs, image, *args, **kwargs)

    try:
        if filter_narrow_staffs:
            homr_main.detect_staff = _filter_narrow_staffs_wrapper
        homr_main.autocrop = _recording_autocrop
        homr_main.parse_staffs = _recording_parse_staffs

        config = ProcessingConfig(False, False, False, False, -1, False, False, False)
        process_image(str(target_image), config, XmlGeneratorArguments(False, None, None))
        target_image.with_suffix(".layout.json").write_text(json.dumps(layout), encoding="utf-8")
        return target_image.with_suffix(".musicxml")
    finally:
        homr_main.detect_staff = original_homr_detect_staff
        homr_main.autocrop = original_autocrop
        homr_main.parse_staffs = original_parse_staffs
