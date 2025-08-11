"""
7세그 속도 + 시간 엑셀 내보내기 (필터 적용 + 종료 전 0구간 제거)

`classify_sevenseg.py`가 생성한 분류 CSV를 읽어 시간 열을 포함한 엑셀 파일을 작성합니다.

엑셀에는 **다음 구간만** 저장합니다.
- 시작: 속도가 0 → ≥1 로 **처음 전이되는 프레임**(t=0)
- 끝: 시작 이후 **첫 검은 화면**(블랙 프레임)이 나타나기 **바로 직전의 마지막 "비(非)0 속도" 프레임**
  - 즉, 블랙 프레임 직전의 **연속된 0 속도 구간은 제외**합니다(완전 정차 후 종료 구간 제거).
- 블랙 프레임이 없을 경우: 파일 끝까지 포함(이때는 말단 0 구간을 강제 제거하지 않음).

시간 규칙:
- FPS(기본 30) 기준으로 프레임당 1/FPS씩 증가
- 저장된 마지막 프레임까지 시간값을 기록

사용 예:
    python export_speed_to_excel.py \
        --cls_csv _cls_result.csv \
        --images_dir ../frames30_pts \
        --out_xlsx _speed_time.xlsx \
        --fps 30
"""
from __future__ import annotations
import os
import argparse
import math
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm

# --------------------- 검은 화면(블랙 프레임) 판정 ---------------------

def is_black_image(path: str, mean_t: float = 6.0, std_t: float = 6.0) -> bool:
    """`path`의 이미지를 사실상 '검은 화면'으로 볼지 여부를 반환합니다."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # 파일이 없으면 보수적으로 '검은 화면 아님'으로 처리
        return False
    m = float(img.mean())
    s = float(img.std())
    return (m < mean_t) and (s < std_t)

# --------------------- 핵심 로직 ---------------------

def export_speed_xlsx(
    cls_csv_path: str = "_cls_result.csv",
    images_dir: str = "../frames30_pts",
    out_xlsx: str = "_speed_time.xlsx",
    fps: int = 30,
    black_mean_t: float = 6.0,
    black_std_t: float = 6.0,
) -> str:
    """분류 CSV를 읽어 시간 열을 포함한 엑셀 파일을 생성합니다.

    저장 규칙:
    - 시작 인덱스(start_idx): 속도가 0에서 ≥1로 전이되는 최초 프레임
    - 종료 인덱스(trimmed_end): 시작 이후 첫 블랙 프레임 직전에서 **우측 말단의 연속 0 속도 구간**을 제거한 뒤 남는 마지막 비0 프레임
    - 저장 범위: [start_idx, trimmed_end]를 기본으로 하되,
        * start_idx 바로 앞 프레임이 0이면 **선행 0 한 프레임 포함**
        * trimmed_end 바로 다음 프레임이 0이고 블랙 이전이면 **말단 0 한 프레임 포함**
    """
    if not os.path.isfile(cls_csv_path):
        raise FileNotFoundError(f"CSV not found: {cls_csv_path}")

    df = pd.read_csv(cls_csv_path)
    required = {"filename", "pred_number"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing from CSV: {sorted(missing)}")

    # 원본 처리 순서 유지(행 인덱스 리셋)
    df = df.reset_index(drop=True)

    # 속도 열 정수화(유효하지 않은 값은 0으로 대체) 및 음수 방지
    speeds = pd.to_numeric(df["pred_number"], errors="coerce").fillna(0).astype(int)
    speeds = speeds.clip(lower=0)

    # 검은 화면 여부 계산(시작 이후 최초 True를 종료 기준으로 사용)
    img_paths = [os.path.join(images_dir, fn) for fn in df["filename"].astype(str)]

    if tqdm is not None:
        blacks = []
    for p in tqdm(img_paths, desc="블랙 판정", unit="img", dynamic_ncols=True):
        blacks.append(is_black_image(p, mean_t=black_mean_t, std_t=black_std_t))
    else:
        blacks = [is_black_image(p, mean_t=black_mean_t, std_t=black_std_t) for p in img_paths]


    # 시작 인덱스: speed가 ≥1이고, 직전이 0인 최초 지점
    start_idx: Optional[int] = None
    for i in range(len(speeds)):
        if speeds.iloc[i] >= 1 and (i == 0 or speeds.iloc[i - 1] == 0):
            start_idx = i
            break

    # 종료 기준: 시작 이후 최초의 블랙 프레임 (없으면 None)
    black_idx: Optional[int] = None
    if start_idx is not None:
        for j in range(start_idx, len(blacks)):
            if blacks[j]:
                black_idx = j
                break

    # 경과 시간 계산(전 구간 대비로 미리 계산해도 되지만, 슬라이스에서 사용)
    elapsed = [math.nan] * len(df)
    if start_idx is not None:
        # 블랙이 없으면 파일 끝까지 시간 증가
        last_idx_for_time = (black_idx - 1) if black_idx is not None else (len(df) - 1)
        for i in range(start_idx, last_idx_for_time + 1):
            elapsed[i] = (i - start_idx) / float(fps)

    # 전체 프레임 테이블 구성
    all_rows = pd.DataFrame({
        "idx": np.arange(len(df), dtype=int),
        "filename": df["filename"],
        "speed": speeds,
        "is_black": blacks,
        "time_s": elapsed,
    })

    # ---------------- 필터: 시작~(블랙 직전 말단 0 제거) + 경계 0 포함 ----------------
    if start_idx is None:
        out = all_rows.iloc[0:0].copy()
        range_note = "no_start_found"
        trimmed_end = None
        trailing_zeros = 0
        lead0_included = 0
        trail0_included = 0
    else:
        # 블랙 여부에 따른 우측 경계 설정
        right_limit = black_idx if black_idx is not None else len(df)
        # 우측 말단에서 연속된 0 속도 구간 길이 계산 (시작 이전은 보지 않음)
        j = right_limit - 1
        trailing_zeros = 0
        while j >= start_idx and speeds.iloc[j] == 0:
            trailing_zeros += 1
            j -= 1
        # 트리밍 규칙: 블랙이 존재할 때만 말단 0 구간 제거
        if black_idx is not None:
            trimmed_end = j
        else:
            trimmed_end = right_limit - 1

        # 기본 저장 범위
        start_save = start_idx
        end_save = trimmed_end
        lead0_included = 0
        trail0_included = 0

        # 선행 0 한 프레임 포함
        if start_idx > 0 and speeds.iloc[start_idx - 1] == 0:
            start_save = start_idx - 1
            lead0_included = 1

        # 말단 0 한 프레임 포함(블랙이 있는 경우에만, 블랙 바로 직전의 0)
        if black_idx is not None:
            if (trimmed_end + 1) < right_limit and speeds.iloc[trimmed_end + 1] == 0:
                end_save = trimmed_end + 1
                trail0_included = 1

        if end_save < start_save:
            # 유효 범위가 없으면 빈 구간
            out = all_rows.iloc[0:0].copy()
        else:
            out = all_rows.iloc[start_save:end_save + 1].reset_index(drop=True)
        range_note = f"{start_save}-{end_save}"

    # 메타데이터 시트 구성
    meta_rows = [
        {"key": "fps", "value": fps},
        {"key": "start_idx", "value": -1 if start_idx is None else int(start_idx)},
        {"key": "black_idx", "value": -1 if black_idx is None else int(black_idx)},
        {"key": "trimmed_end_idx", "value": -1 if (start_idx is None or trimmed_end is None) else int(trimmed_end)},
        {"key": "trailing_zero_count_before_black", "value": trailing_zeros},
        {"key": "leading_zero_included", "value": lead0_included},
        {"key": "trailing_zero_included", "value": trail0_included},
        {"key": "black_mean_t", "value": black_mean_t},
        {"key": "black_std_t", "value": black_std_t},
        {"key": "saved_range", "value": range_note},
        {"key": "images_dir", "value": images_dir},
    ]
    meta = pd.DataFrame(meta_rows, columns=["key", "value"])

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
        out.to_excel(xw, index=False, sheet_name="speed_time")
        meta.to_excel(xw, index=False, sheet_name="meta")

    return out_xlsx

# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser(description="Export speeds with time to Excel (active interval, trim trailing zeros before black).")
    ap.add_argument("--cls_csv", default="_cls_result.csv", help="분류 CSV 경로")
    ap.add_argument("--images_dir", default="../frames30_pts", help="원본 이미지 폴더(블랙 판정용)")
    ap.add_argument("--out_xlsx", default="_speed_time.xlsx", help="출력 엑셀 경로")
    ap.add_argument("--fps", type=int, default=30, help="FPS (기본 30)")
    ap.add_argument("--black_mean_t", type=float, default=6.0, help="블랙 판정용 평균 밝기 임계(0..255)")
    ap.add_argument("--black_std_t", type=float, default=6.0, help="블랙 판정용 표준편차 임계(0..255)")
    args = ap.parse_args()

    out = export_speed_xlsx(
        cls_csv_path=args.cls_csv,
        images_dir=args.images_dir,
        out_xlsx=args.out_xlsx,
        fps=args.fps,
        black_mean_t=args.black_mean_t,
        black_std_t=args.black_std_t,
    )
    print(f"Wrote: {out}")

if __name__ == "__main__":
    main()
