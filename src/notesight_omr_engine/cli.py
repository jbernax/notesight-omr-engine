"""
CLI entrypoint for notesight-omr-engine.

Usage:
  notesight-omr [--filter-narrow-staffs] <image_path>
  notesight-omr --init
  notesight-omr --version

Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from notesight_omr_engine import __version__
from notesight_omr_engine.runner import transcribe_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="notesight-omr",
        description="Standalone Optical Music Recognition CLI powered by homr (AGPL-3.0-or-later).",
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Path to the input score page image (.png, .jpg, .tiff).",
    )
    parser.add_argument(
        "--filter-narrow-staffs",
        action="store_true",
        default=None,
        help="Discard spurious staff fragments narrower than 40%% of median staff width.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Pre-download homr ONNX neural network checkpoints.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print notesight-omr-engine version and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"notesight-omr-engine {__version__}")
        return 0

    if args.init:
        proc = subprocess.run([sys.executable, "-m", "homr.main", "--init"], check=False)
        return proc.returncode

    if not args.image_path:
        parser.print_help(sys.stderr)
        return 2

    image_file = Path(args.image_path)
    if not image_file.exists():
        print(f"Error: Input image not found: {image_file}", file=sys.stderr)
        return 1

    transcribe_image(image_file, filter_narrow_staffs=args.filter_narrow_staffs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
