"""
classify_steering.py

frames30_pts_steer(이미 크롭된 핸들 UI 프레임들)을 입력으로 받아
픽셀 중심점을 steering 각도로 변환하여(±60도 범위, 0.5도 단위) CSV로 저장합니다.

변환 원리:
- 이미지 가로: 58픽셀 = -540도 ~ +540도 범위 (전체 1080도)
- 중앙(0도): x = 27픽셀
- 각도 = (픽셀 - 27) × (1080 / 58)
- 양자화: 0.5도 단위
- 필터: ±60도 범위만 유효

- 입력: frames_dir (예: .../frames30_pts_steer)  # img_%010d.png
- 출력:
    - work_dir/_steer_result.csv (컬럼: frame_idx, time_sec, cx_raw, angle_deg_raw, angle_deg_quantized_05, img_path)
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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    zero_center_x: int = 26   # 기준선 x (px)
    x_point_bias: int = 0     # 감지점 x 보정 (0으로 수정)
    tolerance: int = 6        # 0 판정 허용 범위

    # 디버그 오버레이 저장
    overlay_border: int = 5
    overlay_every_n: int = 1   # 모든 프레임 저장


def _sorted_frame_paths(frames_dir: Path) -> List[Path]:
    paths = sorted(frames_dir.glob(IMG_GLOB))
    # 파일명이 img_%010d.png 패턴이면, 숫자 기준으로 정렬이 안전합니다.
    def key(p: Path) -> int:
        m = IMG_NUM_RE.search(p.name)
        return int(m.group(1)) if m else 0
    return sorted(paths, key=key)


def _analyze_one(img_bgr: np.ndarray, cfg: SteeringConfig) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """단일 steer 프레임 분석: (cx_raw, steering_angle_deg, steering_angle_quantized) 반환
    
    픽셀-각도 변환:
    - 이미지 가로 58픽셀 = -540도 ~ +540도 (전체 1080도)
    - 중앙(0도) = x=27 픽셀
    - 1픽셀 ≈ 18.62도
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, cfg.thresh_value, cfg.thresh_max, cv2.THRESH_BINARY)

    M = cv2.moments(thresh)
    if M["m00"] <= 0:
        return None, None, None

    cx_raw = int(M["m10"] / M["m00"])

    # 보정: 사용자가 지정한 x_point_bias(px)를 측정값에 적용
    cx_corrected = cx_raw + cfg.x_point_bias

    # 픽셀을 각도로 변환
    # 58픽셀이 1080도(±540도)를 나타냄
    # 1픽셀 = 1080/58 ≈ 18.62도
    # 각도 = (보정된 픽셀 - 중앙) × (1080 / 58)
    degrees_per_pixel = 1080.0 / 58.0  # 약 18.62 도/픽셀
    angle_deg = (cx_corrected - cfg.zero_center_x) * degrees_per_pixel

    # 0.5도 단위로 양자화
    angle_quantized = round(angle_deg / 0.5) * 0.5

    # ±60도 범위 필터 (범위 초과 시 None으로 표시)
    if abs(angle_quantized) > 60.0:
        angle_quantized = None

    return cx_raw, angle_deg, angle_quantized


def _save_overlay(img_bgr: np.ndarray, cx_raw: Optional[int], px_offset: Optional[float], cfg: SteeringConfig, out_path: Path) -> None:
    h, w = img_bgr.shape[:2]

    # 오프셋(px)에 따라 테두리 색상 결정 (BGR)
    if px_offset is None:
        border_color = (128, 128, 128)  # 회색: 감지 실패
    elif px_offset == 0:
        border_color = (0, 255, 0)      # 초록색: N (중앙)
    elif px_offset < -5:
        border_color = (255, 0, 0)      # 파란색: LE (왼쪽 크게)
    elif px_offset < 0:
        border_color = (255, 255, 0)    # 청록색: L (왼쪽)
    elif px_offset > 5:
        border_color = (0, 0, 255)      # 빨간색: RE (오른쪽 크게)
    else: # 0 < px_offset <= 5
        border_color = (0, 255, 255)    # 노란색: R (오른쪽)

    debug_img = cv2.copyMakeBorder(
        img_bgr,
        cfg.overlay_border, cfg.overlay_border, cfg.overlay_border, cfg.overlay_border,
        cv2.BORDER_CONSTANT,
        value=border_color
    )
    # 기준선(0도) 표시
    x0 = cfg.zero_center_x + cfg.overlay_border
    cv2.line(debug_img, (x0, 0), (x0, h + cfg.overlay_border * 2), (255, 255, 255), 1)

    # 감지점 표시: 측정값에 x_point_bias를 적용한 위치에 노란 점 표시
    if cx_raw is not None:
        cx_display = cx_raw + cfg.x_point_bias + cfg.overlay_border
        cv2.circle(debug_img, (cx_display, (h // 2) + cfg.overlay_border), 2, (0, 255, 255), -1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # PNG 압축 레벨 0: 무압축 저장 (58×19px 소형 이미지라 압축 오버헤드 > 절감 효과)
    cv2.imwrite(str(out_path), debug_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


# ─────────────────────────────────────────────────────────────
# 병렬 로딩 헬퍼
# ─────────────────────────────────────────────────────────────

def _load_one_frame(path: Path) -> Optional[np.ndarray]:
    """디스크에서 이미지 한 장을 읽어 반환합니다.
    
    반환값:
        - 정상: BGR ndarray
        - 실패(파일 없음 / 빈 파일): None
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    return img


def _load_frames_parallel(paths: List[Path], max_workers: int = 4) -> List[Optional[np.ndarray]]:
    """이미지 목록을 스레드 풀로 병렬 로딩합니다.

    pool.map()을 사용하므로 반환 순서는 입력(paths) 순서와 동일합니다.
    (로딩이 완료되는 순서가 달라도 결과는 항상 paths[0]→imgs[0] 순서)

    args:
        paths      : 로딩할 파일 경로 리스트 (정렬된 순서)
        max_workers: 동시에 디스크를 읽을 스레드 수 (기본값 4)
    returns:
        paths와 같은 순서의 ndarray 리스트 (실패 시 해당 위치 None)
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        imgs = list(pool.map(_load_one_frame, paths))
    return imgs


# ─────────────────────────────────────────────────────────────
# 단일 프레임 분석 헬퍼
# ─────────────────────────────────────────────────────────────

def _analyze_frame(img: np.ndarray, i: int, fps: float, p: Path,
                   cfg: SteeringConfig, keep_img: bool) -> Dict:
    """이미지 1장을 분석하여 결과 딕셔너리를 반환합니다.

    분석 내용:
        - 핸들 UI 픽셀 중심(cx_raw) 검출
        - 중앙(zero_center_x) 대비 픽셀 오프셋 계산
        - 위치 라벨 생성 (예: N_0, L_-3, R_+5)

    args:
        img      : 분석할 BGR 이미지
        i        : 프레임 인덱스 (파일 순서 번호)
        fps      : 초당 프레임 수 (시간 계산용)
        p        : 원본 파일 경로 (csv 기록용)
        cfg      : 분석 설정값
        keep_img : True면 결과에 img를 포함 (오버레이 저장용)
    returns:
        분석 결과를 담은 딕셔너리
    """
    cx_raw, _, _ = _analyze_one(img, cfg)

    # 픽셀 오프셋 계산 (중앙 0 기준, 왼쪽 음수 / 오른쪽 양수)
    px_offset = None
    if cx_raw is not None:
        cx_corrected = cx_raw + cfg.x_point_bias
        px_offset = cx_corrected - cfg.zero_center_x

    # 위치 라벨 생성 (파일명·CSV에 사용)
    if px_offset is None:  pos_str, off_val = "X", "null"
    elif px_offset == 0:   pos_str, off_val = "N", "0"
    elif px_offset < 0:    pos_str, off_val = "L", str(px_offset)
    else:                  pos_str, off_val = "R", f"+{px_offset}"
    steer_label = f"{pos_str}_{off_val}"

    return {
        "frame_idx": i,
        "time_sec":  i / fps,
        "cx_raw":    cx_raw,
        "px_offset": px_offset,
        "steer_label": steer_label,
        "img_path":  str(p),
        "error":     False,
        "img":       img if keep_img else None,
    }


# ─────────────────────────────────────────────────────────────
# 병렬 오버레이 저장 헬퍼
# ─────────────────────────────────────────────────────────────

def _save_overlay_batch(
    save_queue: List[Tuple[np.ndarray, Optional[int], Optional[float], SteeringConfig, Path]],
    max_workers: int = 4,
) -> None:
    """오버레이 이미지를 스레드 풀로 병렬 저장합니다.

    분석이 완전히 끝난 뒤 save_queue에 쌓인 항목을 한꺼번에 씁니다.
    각 파일은 독립적이므로 저장 순서가 뒤바뀌어도 결과에 영향이 없습니다.
    (파일명에 프레임 번호가 명시되어 있기 때문)

    args:
        save_queue  : (img, cx_raw, px_offset, cfg, out_path) 튜플 리스트
        max_workers : 동시에 디스크에 쓸 스레드 수 (기본값 4)
    """
    def _write(args):
        img, cx_raw, px_offset, cfg, out_path = args
        _save_overlay(img, cx_raw, px_offset, cfg, out_path)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pool.map(_write, save_queue)


# ─────────────────────────────────────────────────────────────
# 메인 분석 함수
# ─────────────────────────────────────────────────────────────

def analyze_steering_frames(
    frames_dir: Path,
    work_dir: Path,
    fps: float,
    cfg: SteeringConfig = SteeringConfig(),
    overlay: bool = False,
) -> Path:
    """frames_dir 내 img_*.png를 분석하여 CSV로 저장하고 결과 CSV 경로를 반환합니다.

    처리 순서:
        1) 이미지 병렬 로딩  (_load_frames_parallel)
        2) 프레임별 분석     (_analyze_frame)
        3) 노이즈 감지       (직전·직후 프레임 비교)
        4) 3D-LNR / SRR 계산
        5) CSV 저장
        6) 오버레이 이미지 병렬 저장 (_save_overlay_batch), overlay=True 시
    """
    frames_dir = frames_dir.resolve()
    work_dir   = work_dir.resolve()

    if not frames_dir.is_dir():
        raise RuntimeError(f"frames_dir 폴더를 찾을 수 없습니다: {frames_dir}")

    paths = _sorted_frame_paths(frames_dir)
    if not paths:
        raise RuntimeError(f"프레임 이미지를 찾지 못했습니다: {frames_dir}/{IMG_GLOB}")

    out_csv    = work_dir / "_steer_result.csv"
    overlay_dir = work_dir / "_steer_overlay"

    # ── 1단계: 병렬 로딩 ──────────────────────────────────────
    # 스레드 풀이 paths 순서를 보장하여 imgs[i] == paths[i]
    imgs = _load_frames_parallel(paths)

    # ── 2단계: 순서대로 분석 ──────────────────────────────────
    # 분석 자체는 이전 프레임 결과에 의존하지 않으므로
    # 로딩만 병렬화해도 병목이 크게 줄어듭니다.
    results = []
    for i, (p, img) in enumerate(zip(paths, imgs)):
        if img is None:
            # 이미지 로딩 실패 → 에러 레코드 추가
            results.append({
                "frame_idx": i, "time_sec": i / fps,
                "cx_raw": None, "px_offset": None,
                "steer_label": "ERR", "img_path": str(p),
                "error": True, "img": None,
            })
            continue

        res = _analyze_frame(img, i, fps, p, cfg, keep_img=overlay)
        results.append(res)

    # (2단계: 노이즈 감지 로직은 그대로 유지)
    for i in range(len(results)):
        res = results[i]
        if res.get("error") or res["px_offset"] is None:
            res["is_noise"] = "N"
            continue
        if i == 0 or i == len(results) - 1:
            res["is_noise"] = "N"
            continue
        prev_res = results[i-1]
        next_res = results[i+1]
        if prev_res.get("error") or next_res.get("error") or \
           prev_res["px_offset"] is None or next_res["px_offset"] is None:
            res["is_noise"] = "N"
            continue
        cur_val = res["px_offset"]
        if cur_val != prev_res["px_offset"] and cur_val != next_res["px_offset"]:
            res["is_noise"] = "Y"
        else:
            res["is_noise"] = "N"

    # 3단계: 3D-LNR 및 SRR 계산 준비
    def get_3d_lnr(val, n):
        if val is None: return ""
        if val == 0: return "N"
        if 0 < val <= n: return "R"
        if -n <= val < 0: return "L"
        if val > n: return "RE"  # Right Exceed
        if val < -n: return "LE" # Left Exceed
        return ""

    for res in results:
        res["lnr_vals"] = {}
        val = res["px_offset"]
        for n in range(1, 6):
            res["lnr_vals"][n] = get_3d_lnr(val, n)

    intentional_window = 60
    side_durations = []
    current_side = None
    duration = 0
    for i in range(len(results)):
        val = results[i]["px_offset"]
        side = "N"
        if val is not None:
            if val < 0: side = "L"
            elif val > 0: side = "R"
        if side == current_side and side != "N":
            duration += 1
        else:
            current_side, duration = side, 1
        side_durations.append(duration)
    
    future_durations = [0] * len(results)
    current_side, duration = None, 0
    for i in range(len(results)-1, -1, -1):
        val = results[i]["px_offset"]
        side = "N"
        if val is not None:
            if val < 0: side = "L"
            elif val > 0: side = "R"
        if side == current_side and side != "N":
            duration += 1
        else:
            current_side, duration = side, 1
        future_durations[i] = duration

    for n in range(1, 6):
        last_extreme = None
        direction = 0
        for i in range(len(results)):
            res = results[i]
            res[f"srr_check_{n}"] = "N"
            res[f"srr_58d_{n}"] = "N"
            if i == 0 or res.get("error") or res["px_offset"] is None:
                if res["px_offset"] is not None: last_extreme = res["px_offset"]
                continue
            prev_res = results[i-1]
            if prev_res.get("error") or prev_res["px_offset"] is None:
                last_extreme = res["px_offset"]
                continue
            cur_val = res["px_offset"]
            prev_val = prev_res["px_offset"]
            cur_lnr, prev_lnr = res["lnr_vals"][n], prev_res["lnr_vals"][n]
            is_intentional_trans = (prev_lnr == "LE" and cur_lnr == "L") or \
                                   (prev_lnr == "RE" and cur_lnr == "R") or \
                                   (prev_lnr == "L" and cur_lnr == "LE") or \
                                   (prev_lnr == "R" and cur_lnr == "RE")
            if cur_lnr != prev_lnr and not is_intentional_trans:
                res[f"srr_check_{n}"] = "Y"
            is_long_corner = (side_durations[i] >= intentional_window) or (future_durations[i] >= intentional_window)
            if not is_intentional_trans and not is_long_corner:
                diff = cur_val - prev_val
                if diff != 0:
                    new_dir = 1 if diff > 0 else -1
                    if last_extreme is None:
                        last_extreme, direction = prev_val, new_dir
                    if new_dir != direction:
                        if abs(cur_val - last_extreme) >= n:
                            res[f"srr_58d_{n}"] = "Y"
                            last_extreme, direction = cur_val, new_dir
                    else:
                        if (direction == 1 and cur_val > last_extreme) or (direction == -1 and cur_val < last_extreme):
                            last_extreme = cur_val

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["frame_idx", "time_sec", "cx_raw", "steer_px_offset", "steer_label", "is_noise"]
        for n in range(1, 6):
            header.extend([f"lnr_{n}px", f"3d_srr_{n}px", f"58d_srr_{n}px"])
        header.append("img_path")
        w.writerow(header)

        for res in results:
            if res.get("error"):
                row = [res["frame_idx"], f"{res['time_sec']:.3f}", "", "", "ERR_EMPTY", "N"] + [""] * 15 + [res["img_path"]]
                w.writerow(row)
                continue
            cx_raw_str = "" if res["cx_raw"] is None else res["cx_raw"]
            px_off_str = "" if res["px_offset"] is None else res["px_offset"]
            row = [res["frame_idx"], f"{res['time_sec']:.3f}", cx_raw_str, px_off_str, res["steer_label"], res["is_noise"]]
            for n in range(1, 6):
                row.extend([res["lnr_vals"][n], res[f"srr_check_{n}"], res[f"srr_58d_{n}"]])
            row.append(res["img_path"])
            w.writerow(row)

    # ── 6단계: 오버레이 이미지 병렬 저장 ──────────────────────────────
    # 분석(CSV 기록)이 완전히 끝난 뒤, save_queue를 한꺼번에 병렬 저장합니다.
    # 각 파일은 독립적이므로 저장 완료 순서가 달라도 결과에 영향이 없습니다.
    if overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        save_queue = [
            (
                res["img"],
                res["cx_raw"],
                res["px_offset"],
                cfg,
                overlay_dir / f"f{res['frame_idx']:06d}_{res['steer_label']}.png",
            )
            for res in results
            if res["img"] is not None
        ]
        _save_overlay_batch(save_queue)

    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="frames30_pts_steer 기반 핸들(steering) UI 분석")
    ap.add_argument("--in-dir", required=True, help="입력 frames 폴더 (예: .../frames30_pts_steer)")
    ap.add_argument("--work-dir", required=True, help="출력 저장 기준 폴더 (예: 영상 작업 폴더)")
    ap.add_argument("--fps", type=float, required=True, help="프레임 FPS (time_sec 계산용)")

    ap.add_argument("--overlay", action="store_true", help="샘플 디버그 이미지 저장")
    ap.add_argument("--overlay-every", type=int, default=30, help="오버레이 저장 주기 (기본 30)")

    ap.add_argument("--thresh", type=int, default=200, help="threshold 값 (기본 200)")
    ap.add_argument("--zero-center-x", type=int, default=26, help="0 기준 x(px) (기본 26)")
    ap.add_argument("--x-bias", type=int, default=0, help="감지점 x 보정(px) (기본 0)")
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
