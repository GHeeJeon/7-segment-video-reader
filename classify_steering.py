"""
classify_steering.py

frames30_pts_steer(이미 크롭된 핸들 UI 프레임들)을 입력으로 받아
인디케이터의 중심점을 모멘트로 계산하고(L/0/R) 결과를 CSV로 저장합니다.

- 입력: frames_dir (예: .../frames30_pts_steer)  # img_%010d.png
- 출력:
    - work_dir/_steer_result.csv
    - (옵션) work_dir/_steer_overlay/  # 샘플 디버그 이미지

요구 패키지: opencv-python, numpy
(CSV는 표준 라이브러리로 저장)

사용 예:
    python classify_steering.py --in-dir "./1/1-1/frames30_pts_steer" --work-dir "./1/1-1" --fps 30 --overlay

"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


IMG_GLOB = "img_*.png"
IMG_NUM_RE = re.compile(r"img_(\d+)\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class SteeringConfig:
    # 모멘트 임계 처리
    thresh_value: int = 200
    thresh_max: int = 255

    # 상태 판정 파라미터 (크롭된 steer 이미지 좌표계 기준)
    zero_center_x: int = 29   # 기준선 x (px)
    x_point_bias: int = 3     # 감지점 x 보정 (+)
    tolerance: int = 6        # 0 판정 허용 범위

    # 디버그 오버레이 저장
    overlay_border: int = 5
    overlay_every_n: int = 30  # 30프레임마다 1장 저장 (기본 30fps면 1초마다)


def _sorted_frame_paths(frames_dir: Path) -> List[Path]:
    paths = sorted(frames_dir.glob(IMG_GLOB))
    # 파일명이 img_%010d.png 패턴이면, 숫자 기준으로 정렬이 안전합니다.
    def key(p: Path) -> int:
        m = IMG_NUM_RE.search(p.name)
        return int(m.group(1)) if m else 0
    return sorted(paths, key=key)


def _analyze_one(img_bgr: np.ndarray, cfg: SteeringConfig) -> Tuple[Optional[int], Optional[int], str]:
    """단일 steer 프레임 분석: (cx_raw, cx_calibrated, state) 반환"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, cfg.thresh_value, cfg.thresh_max, cv2.THRESH_BINARY)

    M = cv2.moments(thresh)
    if M["m00"] <= 0:
        return None, None, "N/A"

    cx_raw = int(M["m10"] / M["m00"])
    cx_cal = cx_raw + cfg.x_point_bias

    if cx_cal < (cfg.zero_center_x - cfg.tolerance):
        state = "L"
    elif cx_cal > (cfg.zero_center_x + cfg.tolerance):
        state = "R"
    else:
        state = "0"

    return cx_raw, cx_cal, state


def _save_overlay(img_bgr: np.ndarray, cx_cal: Optional[int], state: str, cfg: SteeringConfig, out_path: Path) -> None:
    h, w = img_bgr.shape[:2]

    if state == "0":
        border_color = (0, 255, 0)
    elif state == "R":
        border_color = (0, 0, 255)
    elif state == "L":
        border_color = (255, 0, 0)
    else:
        border_color = (128, 128, 128)

    debug_img = cv2.copyMakeBorder(
        img_bgr,
        cfg.overlay_border, cfg.overlay_border, cfg.overlay_border, cfg.overlay_border,
        cv2.BORDER_CONSTANT,
        value=border_color
    )

    # 기준선 표시
    x0 = cfg.zero_center_x + cfg.overlay_border
    cv2.line(debug_img, (x0, 0), (x0, h + cfg.overlay_border * 2), (255, 255, 255), 1)

    # 감지점 표시
    if cx_cal is not None:
        cv2.circle(debug_img, (cx_cal + cfg.overlay_border, (h // 2) + cfg.overlay_border), 3, (0, 255, 255), -1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), debug_img)


def analyze_steering_frames(
    frames_dir: Path,
    work_dir: Path,
    fps: float,
    cfg: SteeringConfig = SteeringConfig(),
    overlay: bool = False,
) -> Path:
    """frames_dir 내 img_*.png를 순회하며 steer 상태를 CSV로 저장하고 결과 CSV 경로를 반환"""
    frames_dir = frames_dir.resolve()
    work_dir = work_dir.resolve()

    if not frames_dir.is_dir():
        raise RuntimeError(f"frames_dir 폴더를 찾을 수 없습니다: {frames_dir}")

    paths = _sorted_frame_paths(frames_dir)
    if not paths:
        raise RuntimeError(f"프레임 이미지를 찾지 못했습니다: {frames_dir}/{IMG_GLOB}")

    out_csv = work_dir / "_steer_result.csv"
    overlay_dir = work_dir / "_steer_overlay"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "time_sec", "cx_raw", "cx_calibrated", "state", "img_path"])

        for i, p in enumerate(paths):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                # 빈/손상 파일 방어
                w.writerow([i, i / fps, "", "", "ERR_EMPTY", str(p)])
                continue

            cx_raw, cx_cal, state = _analyze_one(img, cfg)
            w.writerow([i, i / fps, "" if cx_raw is None else cx_raw, "" if cx_cal is None else cx_cal, state, str(p)])

            if overlay and (i % cfg.overlay_every_n == 0):
                out_img = overlay_dir / f"sample_f{i:06d}_{state}.png"
                _save_overlay(img, cx_cal, state, cfg, out_img)

    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="frames30_pts_steer 기반 핸들(steering) UI 분석")
    ap.add_argument("--in-dir", required=True, help="입력 frames 폴더 (예: .../frames30_pts_steer)")
    ap.add_argument("--work-dir", required=True, help="출력 저장 기준 폴더 (예: 영상 작업 폴더)")
    ap.add_argument("--fps", type=float, required=True, help="프레임 FPS (time_sec 계산용)")

    ap.add_argument("--overlay", action="store_true", help="샘플 디버그 이미지 저장")
    ap.add_argument("--overlay-every", type=int, default=30, help="오버레이 저장 주기 (기본 30)")

    ap.add_argument("--thresh", type=int, default=200, help="threshold 값 (기본 200)")
    ap.add_argument("--zero-center-x", type=int, default=29, help="0 기준 x(px) (기본 29)")
    ap.add_argument("--x-bias", type=int, default=3, help="감지점 x 보정(px) (기본 +3)")
    ap.add_argument("--tol", type=int, default=6, help="0 판정 허용범위(px) (기본 6)")

    args = ap.parse_args()

    cfg = SteeringConfig(
        thresh_value=args.thresh,
        zero_center_x=args.zero_center_x,
        x_point_bias=args.x_bias,
        tolerance=args.tol,
        overlay_every_n=args.overlay_every,
    )

    out_csv = analyze_steering_frames(
        frames_dir=Path(args.in_dir),
        work_dir=Path(args.work_dir),
        fps=args.fps,
        cfg=cfg,
        overlay=args.overlay,
    )

    print(f"완료: {out_csv}")


if __name__ == "__main__":
    main()
