#!/usr/bin/env python3
"""Prepare the workpiece KITTI-like stereo sequence for pySLAM.

The source sequence has a vertical stereo baseline in image coordinates. pySLAM's
KITTI stereo frontend expects rectified horizontal disparity, so this script
rotates both cameras counter-clockwise and optionally shifts the right image in y
to reduce residual row mismatch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IDENTITY_KITTI_POSE = "1 0 0 0 0 1 0 0 0 0 1 0\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="/home/gh/pointcloud_ws/data_new/Workpiece_Scan_20260515_kitti",
        help="Source KITTI-like sequence containing image_0, image_1 and times.txt.",
    )
    parser.add_argument(
        "--dst",
        default="/home/gh/work/pyslam/data/workpiece_kitti_vertical_baseline",
        help="Destination KITTI wrapper root.",
    )
    parser.add_argument(
        "--no-y-align",
        action="store_true",
        help="Only rotate images; do not apply per-frame right-image y alignment.",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=500,
        help="Number of ORB matches used to estimate y alignment.",
    )
    return parser.parse_args()


def estimate_right_y_shift(left: np.ndarray, right: np.ndarray, max_matches: int) -> tuple[float, int]:
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right

    orb = cv2.ORB_create(nfeatures=4000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kps_left, des_left = orb.detectAndCompute(gray_left, None)
    kps_right, des_right = orb.detectAndCompute(gray_right, None)
    if des_left is None or des_right is None:
        return 0.0, 0

    matches = sorted(matcher.match(des_left, des_right), key=lambda m: m.distance)[:max_matches]
    y_diffs: list[float] = []
    for match in matches:
        point_left = kps_left[match.queryIdx].pt
        point_right = kps_right[match.trainIdx].pt
        disparity_x = point_left[0] - point_right[0]
        if 0 < disparity_x < 250:
            y_diffs.append(point_left[1] - point_right[1])

    if not y_diffs:
        return 0.0, len(matches)
    return float(np.median(y_diffs)), len(y_diffs)


def shift_y(image: np.ndarray, shift: float) -> np.ndarray:
    height, width = image.shape[:2]
    transform = np.array([[1, 0, 0], [0, 1, shift]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def write_identity_poses(dst_root: Path, count: int) -> None:
    poses_dir = dst_root / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    (poses_dir / "00.txt").write_text(IDENTITY_KITTI_POSE * count)


def main() -> None:
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_seq = dst_root / "sequences" / "00"
    dst_left = dst_seq / "image_0"
    dst_right = dst_seq / "image_1"
    dst_left.mkdir(parents=True, exist_ok=True)
    dst_right.mkdir(parents=True, exist_ok=True)

    left_paths = sorted((src_root / "image_0").glob("*.png"))
    if not left_paths:
        raise FileNotFoundError(f"No PNG frames found in {src_root / 'image_0'}")

    manifest_rows = ["frame shift_y_pixels match_count\n"]
    for left_path in left_paths:
        right_path = src_root / "image_1" / left_path.name
        left = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
        right = cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED)
        if left is None:
            raise RuntimeError(f"Failed to read {left_path}")
        if right is None:
            raise RuntimeError(f"Failed to read {right_path}")

        left_rotated = cv2.rotate(left, cv2.ROTATE_90_COUNTERCLOCKWISE)
        right_rotated = cv2.rotate(right, cv2.ROTATE_90_COUNTERCLOCKWISE)

        shift = 0.0
        match_count = 0
        if not args.no_y_align:
            shift, match_count = estimate_right_y_shift(
                left_rotated, right_rotated, args.max_matches
            )
            right_rotated = shift_y(right_rotated, shift)

        cv2.imwrite(str(dst_left / left_path.name), left_rotated)
        cv2.imwrite(str(dst_right / left_path.name), right_rotated)
        manifest_rows.append(f"{left_path.name} {shift:.6f} {match_count}\n")

    (dst_seq / "times.txt").write_text((src_root / "times.txt").read_text())
    write_identity_poses(dst_root, len(left_paths))
    (dst_root / "vertical_baseline_manifest.txt").write_text("".join(manifest_rows))

    sample = cv2.imread(str(dst_left / left_paths[0].name), cv2.IMREAD_UNCHANGED)
    print(f"Wrote {len(left_paths)} frames to {dst_root}")
    print(f"Sample rotated frame shape: {sample.shape}")
    print(f"Manifest: {dst_root / 'vertical_baseline_manifest.txt'}")


if __name__ == "__main__":
    main()
