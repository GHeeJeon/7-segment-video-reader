
# 7-seg speed + time exporter (CSV-only, no image reading)
# - Reads _cls_result.csv only.
# - Range:
#   * start: first index where speed (pred_number) >= 1
#   * end: last 0 right before the first -1 (black) block after start
#     (i.e., the 0 frame just before 0 -> -1 transition)
#   * if no -1 exists: include until the end of file
# - Removed image-based black detection and any OpenCV calls.

from __future__ import annotations
import argparse
import math
from typing import Optional

import numpy as np
import pandas as pd


def export_speed_xlsx_csv_only(
    cls_csv_path: str = "_cls_result.csv",
    out_xlsx: str = "_speed_time.xlsx",
    fps: int = 30,
    debug: bool = False,
) -> str:
    if not pd.io.common.file_exists(cls_csv_path):
        raise FileNotFoundError(f"CSV not found: {cls_csv_path}")

    # CSV load
    df = pd.read_csv(cls_csv_path)
    # Tidy up
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:")]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df = df.reset_index(drop=True)

    required = {"filename", "pred_number"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing from CSV: {sorted(missing)}")

    # Keep -1 values to detect black block from CSV alone
    speeds = pd.to_numeric(df["pred_number"], errors="coerce").fillna(-1).astype(int)

    # start: first >=1
    start_idx: Optional[int] = None
    for i, v in enumerate(speeds):
        if v >= 1:
            start_idx = i
            break

    # first black (-1) after start
    black_idx: Optional[int] = None
    if start_idx is not None:
        for j in range(start_idx, len(speeds)):
            if speeds.iloc[j] == -1:
                black_idx = j
                break

    # end: last 0 before first -1 (pattern "0 then -1")
    end_on_last_0_idx: Optional[int] = None
    if start_idx is not None:
        # Find the LAST 0 -> -1 transition by scanning from the end
        for j in range(len(speeds) - 2, -1, -1):
            if (speeds.iloc[j] == 0) and (speeds.iloc[j + 1] == -1):
                end_on_last_0_idx = j
                break

    # decide final range
    if start_idx is None:
        out = pd.DataFrame(columns=["idx", "filename", "speed", "is_black", "time_s"])
        range_note = "no_start_found"
        last_idx_for_time = None
    else:
        if black_idx is None:
            start_save = start_idx
            end_save = len(df) - 1
        else:
            if end_on_last_0_idx is not None:
                start_save = start_idx
                end_save = end_on_last_0_idx
            else:
                start_save = start_idx
                end_save = max(start_idx, black_idx - 1)

        # time_s: 0.. based on fps
        elapsed = [math.nan] * len(df)
        for k in range(start_save, end_save + 1):
            elapsed[k] = (k - start_save) / float(fps)

        is_black_series = (speeds == -1).astype(bool)
        all_rows = pd.DataFrame({
            "idx": np.arange(len(df), dtype=int),
            "filename": df["filename"].astype(str),
            "speed": speeds,
            "is_black": is_black_series,
            "time_s": elapsed,
        })

        out = all_rows.iloc[start_save:end_save + 1].reset_index(drop=True)
        range_note = f"{start_save}-{end_save}"
        last_idx_for_time = end_save

    # metrics
    if out.empty:
        total_time = 0.0
        spd = pd.Series([], dtype=float)
        t = pd.Series([], dtype=float)
    else:
        spd = out["speed"].astype(float)
        t = out["time_s"].astype(float)
        total_time = float(t.max()) if t.notna().any() else 0.0

    avg_speed = float(spd.sum() / total_time) if total_time > 0 else float("nan")
    over_mask = spd > 50.0
    over_cnt = int(over_mask.sum())
    over_speed_time = float(over_cnt / float(fps))
    total_over_speed_distance = float((spd[over_mask] / 108000.0).sum())
    part_over_speed_distance = float(((spd[over_mask] - 50.0) / 108000.0).sum())
    if not spd.empty:
        mean_spd = float(spd.mean())
        std_like = float((abs(mean_spd - spd)).mean())
        target_dev = float((abs(50.0 - spd)).mean())
    else:
        std_like = float("nan")
        target_dev = float("nan")
    over_speed_count = over_cnt

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

    meta_rows = [
        {"key": "fps", "value": fps},
        {"key": "start_idx", "value": -1 if start_idx is None else int(start_idx)},
        {"key": "black_idx", "value": -1 if black_idx is None else int(black_idx)},
        {"key": "end_on_last_0_idx", "value": -1 if end_on_last_0_idx is None else int(end_on_last_0_idx)},
        {"key": "saved_range", "value": range_note},
        {"key": "last_idx_for_time", "value": -1 if last_idx_for_time is None else int(last_idx_for_time)},
        {"key": "csv_rows", "value": int(len(df))},
    ]
    meta = pd.DataFrame(meta_rows, columns=["key", "value"])

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
        metrics_df.to_excel(xw, index=False, sheet_name="speed_time", startrow=0)
        startrow = metrics_df.shape[0] + 2
        out.to_excel(xw, index=False, sheet_name="speed_time", startrow=startrow)
        meta.to_excel(xw, index=False, sheet_name="meta")

    if debug:
        print(f"[DEBUG] wrote {out_xlsx} rows={len(out)} start_idx={start_idx} black_idx={black_idx} end0={end_on_last_0_idx}")

    return out_xlsx


def main():
    ap = argparse.ArgumentParser(description="Export speeds with time to Excel using CSV only (trim trailing 0s before first -1 block).")
    ap.add_argument("--cls_csv", default="_cls_result.csv", help="분류 CSV 경로")
    ap.add_argument("--out_xlsx", default="_speed_time.xlsx", help="출력 엑셀 경로")
    ap.add_argument("--fps", type=int, default=30, help="FPS (기본 30)")
    ap.add_argument("--debug", action="store_true", help="디버그 출력")
    args = ap.parse_args()

    out = export_speed_xlsx_csv_only(
        cls_csv_path=args.cls_csv,
        out_xlsx=args.out_xlsx,
        fps=args.fps,
        debug=args.debug,
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
