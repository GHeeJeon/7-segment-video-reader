"""
7세그 속도 + 시간 엑셀 내보내기 (필터 적용 + 종료 전 0구간 제거 + 메트릭 상단 1회 표기)

- classify_sevenseg.py가 생성한 분류 CSV를 읽어 시간 열을 포함한 엑셀 파일을 작성합니다.
- 엑셀에는 다음 구간만 저장합니다.
  * 시작: 속도가 0 → ≥1 로 처음 전이되는 프레임(t=0)
  * 끝: 시작 이후 첫 검은 화면(블랙 프레임) 바로 직전의 마지막 "비0 속도" 프레임
    - 즉, 블랙 프레임 직전의 연속된 0 속도 구간은 제외(완전 정차 후 종료 구간 제거)
  * 블랙 프레임이 없을 경우: 파일 끝까지 포함(이때는 말단 0 구간을 강제 제거하지 않음)

시간 규칙:
- FPS(기본 30) 기준으로 프레임당 1/FPS씩 증가
- 저장된 마지막 프레임까지 시간값을 기록

상단 메트릭(한 번만 표시):
- 총 주행 시간, 평균속력, 총 과속 시간, 총 과속 거리(전체), 총 과속 거리(넘은 거리만),
  속도 표준 편차, 타겟 속도 표준 편차, 총 과속 횟수 (엑셀)
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
    debug: bool = False,
) -> str:
    """분류 CSV를 읽어 시간 열을 포함한 엑셀 파일을 생성합니다."""

    if not os.path.isfile(cls_csv_path):
        raise FileNotFoundError(f"CSV not found: {cls_csv_path}")

    # CSV 로드
    df = pd.read_csv(cls_csv_path)
    # 방어: Unnamed 인덱스열/중복열 제거
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:")]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df = df.reset_index(drop=True)

    required = {"filename", "pred_number"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing from CSV: {sorted(missing)}")

    # 속도 정수화(음수 방지)
    speeds = pd.to_numeric(df["pred_number"], errors="coerce").fillna(0).astype(int).clip(lower=0)

    # 이미지 경로
    img_paths = [os.path.join(images_dir, fn) for fn in df["filename"].astype(str)]

    # 블랙 프레임 판정 (단일 패스)
    iterable = tqdm(img_paths, desc="블랙 판정", unit="img", dynamic_ncols=True) if tqdm is not None else img_paths
    blacks = [is_black_image(p, mean_t=black_mean_t, std_t=black_std_t) for p in iterable]

    if len(blacks) != len(df):
        raise RuntimeError(f"[BUG] blacks len {len(blacks)} != df len {len(df)}")

    if debug:
        print(f"[DEBUG] rows={len(df)}, blacks(first 5)={blacks[:5]}")

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

    # 경과 시간 계산
    elapsed = [math.nan] * len(df)
    if start_idx is not None:
        # 블랙이 없으면 파일 끝까지 시간 증가
        last_idx_for_time = (black_idx - 1) if black_idx is not None else (len(df) - 1)
        for i in range(start_idx, last_idx_for_time + 1):
            elapsed[i] = (i - start_idx) / float(fps)

    # 전체 프레임 테이블
    all_rows = pd.DataFrame({
        "idx": np.arange(len(df), dtype=int),
        "filename": df["filename"],
        "speed": speeds,
        "is_black": blacks,
        "time_s": elapsed,
    })

    # ---------------- 필터: 시작~(블랙 직전 말단 0 제거) + 경계 0 포함 ----------------
    trailing_zeros = 0
    lead0_included = 0
    trail0_included = 0
    trimmed_end: Optional[int] = None
    range_note = "init"

    if start_idx is None:
        out = all_rows.iloc[0:0].copy()
        range_note = "no_start_found"
    else:
        # 블랙 여부에 따른 우측 경계 설정
        right_limit = black_idx if black_idx is not None else len(df)
        # 우측 말단에서 연속된 0 속도 구간 길이 계산 (시작 이전은 보지 않음)
        j = right_limit - 1
        while j >= start_idx and speeds.iloc[j] == 0:
            trailing_zeros += 1
            j -= 1
        # 트리밍 규칙: 블랙이 존재할 때만 말단 0 구간 제거
        trimmed_end = j if black_idx is not None else (right_limit - 1)

        # 기본 저장 범위
        start_save = start_idx
        end_save = trimmed_end

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

    # ---------------- 메트릭 계산(상단 1회 표기용) ----------------
    spd = out["speed"].astype(float) if not out.empty else pd.Series([], dtype=float)
    t = out["time_s"] if not out.empty else pd.Series([], dtype=float)

    # (1) 총 주행 시간(초): time_s의 최대값
    total_time = float(t.max()) if not t.empty and t.notna().any() else 0.0

    # (2) 평균속력: (속도들의 합 / 총 주행 시간)
    avg_speed = float(spd.sum() / total_time) if total_time > 0 else float("nan")

    # 과속 관련(임계 50)
    over_mask = spd > 50.0
    over_cnt = int(over_mask.sum())

    # (3) 총 과속 시간(초): (50 초과 프레임 개수 / FPS)
    over_speed_time = float(over_cnt / float(fps))

    # (4) 총 과속 거리(전체): sum( speed / 108000 ) for speed>50
    total_over_speed_distance = float((spd[over_mask] / 108000.0).sum())

    # (5) 총 과속 거리(넘은 거리만): sum( (speed-50) / 108000 ) for speed>50
    part_over_speed_distance = float(((spd[over_mask] - 50.0) / 108000.0).sum())

    # (6) 속도 표준 편차(요청식): mean(|mean(speed) - speed|)
    if not spd.empty:
        mean_spd = float(spd.mean())
        std_like = float((np.abs(mean_spd - spd)).mean())
    else:
        std_like = float("nan")

    # (7) 타겟 속도 표준 편차: mean(|50 - speed|)
    target_dev = float((np.abs(50.0 - spd)).mean()) if not spd.empty else float("nan")

    # (8) 총 과속 횟수
    over_speed_count = over_cnt

    # speed_time 시트용 상단 메트릭(한글 컬럼명, 1행에 1번만 표시)
    metrics_df = pd.DataFrame([{
        "총 주행 시간": total_time,
        "평균속력": avg_speed,
        "총 과속 시간": over_speed_time,
        "총 과속 거리(전체)": total_over_speed_distance,
        "총 과속 거리(넘은 거리만)": part_over_speed_distance,
        "속도 표준 편차": std_like,
        "타겟 속도 표준 편차": target_dev,
        "총 과속 횟수 (엑셀)": over_speed_count,
    }])

    # per-frame 데이터는 기본 컬럼만 유지
    out_base_cols = ["idx", "filename", "speed", "is_black", "time_s"]
    out = out.loc[:, out_base_cols]

    # ---------------- 메타데이터 시트 ----------------
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
        # 메트릭(영문 키로도 메타에 보관하고 싶으면 아래 유지)
        {"key": "total_time_s", "value": total_time},
        {"key": "average_speed", "value": avg_speed},
        {"key": "over_speed_time_s", "value": over_speed_time},
        {"key": "total_over_speed_distance", "value": total_over_speed_distance},
        {"key": "part_over_speed_distance", "value": part_over_speed_distance},
        {"key": "std_like", "value": std_like},
        {"key": "target_speed_deviation", "value": target_dev},
        {"key": "over_speed_count", "value": over_speed_count},
    ]
    meta = pd.DataFrame(meta_rows, columns=["key", "value"])

    # ---------------- 엑셀 저장 ----------------
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
        # speed_time 시트: (1) 상단 메트릭 1행 (2) 빈 줄 하나 (3) per-frame 테이블
        metrics_df.to_excel(xw, index=False, sheet_name="speed_time", startrow=0)
        startrow = metrics_df.shape[0] + 2  # 빈 줄 하나
        out.to_excel(xw, index=False, sheet_name="speed_time", startrow=startrow)

        # meta 시트
        meta.to_excel(xw, index=False, sheet_name="meta")

    if debug:
        print(f"[DEBUG] Wrote: {out_xlsx}  (rows={len(out)})")

    return out_xlsx


# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser(description="Export speeds with time to Excel (active interval, trim trailing zeros before black) + top metrics.")
    ap.add_argument("--cls_csv", default="_cls_result.csv", help="분류 CSV 경로")
    ap.add_argument("--images_dir", default="../frames30_pts", help="원본 이미지 폴더(블랙 판정용)")
    ap.add_argument("--out_xlsx", default="_speed_time.xlsx", help="출력 엑셀 경로")
    ap.add_argument("--fps", type=int, default=30, help="FPS (기본 30)")
    ap.add_argument("--black_mean_t", type=float, default=6.0, help="블랙 판정용 평균 밝기 임계(0..255)")
    ap.add_argument("--black_std_t", type=float, default=6.0, help="블랙 판정용 표준편차 임계(0..255)")
    ap.add_argument("--debug", action="store_true", help="디버그 출력")
    args = ap.parse_args()

    out = export_speed_xlsx(
        cls_csv_path=args.cls_csv,
        images_dir=args.images_dir,
        out_xlsx=args.out_xlsx,
        fps=args.fps,
        black_mean_t=args.black_mean_t,
        black_std_t=args.black_std_t,
        debug=args.debug,
    )
    print(f"Wrote: {out}")

if __name__ == "__main__":
    main()
