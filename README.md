# notesight-omr-engine

A standalone, stateless Optical Music Recognition (OMR) engine package and CLI tool built on top of [`homr`](https://github.com/liebharc/homr) (SegNet + TrOMR ONNX pipeline).

This package is open-source software licensed under the **GNU Affero General Public License v3.0 or later (`AGPL-3.0-or-later`)** — see [LICENSE](./LICENSE).

---

## Features

- **Zero-Dimension & Inverted Region Safety Guards**: Applies runtime safety wrappers (`notesight_omr_engine.patches`) around `homr.staff_parsing` to prevent `cv2.resize` zero-dimension assertions and negative staff bounding-box coordinates on noisy or tightly cropped scans.
- **System & Staff Layout Coordinate Extraction**: Outputs `<image>.layout.json` alongside `<image>.musicxml` containing the detected staff bounding boxes in original page pixel coordinates for every system.
- **Spurious Staff Fragment Filtering**: Optional `--filter-narrow-staffs` flag (or `OMR_FILTER_NARROW_STAFFS=true` environment variable) to discard narrow ossia/header fragments (`< 40%` of median staff width).
- **Clean Process Boundary**: Designed to run as an isolated CLI subprocess (`notesight-omr <image_path>`) with zero application/database dependencies.

---

## Installation

```bash
pip install ./packages/notesight-omr-engine

# Pre-download ONNX model checkpoints
notesight-omr --init
```

## Usage

```bash
# Transcribe a single page image -> writes page_1.musicxml and page_1.layout.json
notesight-omr path/to/page_1.png

# Enable narrow staff fragment filtering
notesight-omr --filter-narrow-staffs path/to/page_1.png
```
