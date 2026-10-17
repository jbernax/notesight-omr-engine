"""
notesight-omr-engine
Copyright (C) 2026 NoteSight Contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import annotations

__version__ = "0.1.0"

from notesight_omr_engine.patches import apply_homr_safety_patches
from notesight_omr_engine.runner import transcribe_image

__all__ = [
    "__version__",
    "apply_homr_safety_patches",
    "transcribe_image",
]
