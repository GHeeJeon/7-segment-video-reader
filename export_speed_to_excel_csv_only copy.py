import os
import math
import argparse
import pandas as pd
import numpy as np

def export_speed_xlsx_csv_only(
    cls_csv_path: str = "_cls_result.csv",
    out_xlsx_path: str = "_speed_time.xlsx",
    fps: int = 30,
    debug: bool = False,
) -> str:
    if not os.path.isfile(cls_csv_path):
        raise FileNotFoundError(f"CSV not found: {cls_csv_path}")

    df = pd.read_csv(cls_csv_path)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df = df.reset_index(drop=True)

    required = {"filename", "pred_number"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    speeds = pd.to_numeric(df["pred_number"], errors="coerce").fillna(0).astype(int)

    # 말단 연속된 -1 제거
    end_idx = len(speeds) - 1
    last_valid_idx = end_idx
    while last_valid_idx >= 0 and speeds.iloc[last_valid_idx] == -1:
        last_valid_idx -= 1
    black_idx = last_valid_idx + 1 if last_valid_idx < end_idx else None

    # 시작: 0 → 1 이상 전이되는 지점 찾기
    start_idx = None
    for i in range(1, len(speeds)):
        if speeds.iloc[i] >= 1 and speeds.iloc[i - 1] == 0:
            start_idx = i
            break

    if start_idx is None:
        print("No valid start index found")
        return out_xlsx_path

    # 시작 프레임 바로 앞 0 포함
    start_save = start_idx - 1 if start_idx > 0 else start_idx

    # 종료: 블랙 직전 0 제거 (단 1프레임 유지)
    right_limit = black_idx if black_idx is not None else len(df)
    j = right_limit - 1
    while j >= start_idx and speeds.iloc[j] == 0:
        j -= 1
    trimmed_end = j
    end_save = trimmed_end + 1 if trimmed_end + 1 < right_limit and speeds.iloc[trimmed_end + 1] == 0 else trimmed_end

    # 시간 계산
    time_s = [None] * len(df)
    for i in range(start_idx, end_save + 1):
        time_s[i] = (i - start_idx) / float(fps)

    df["speed"] = speeds
    df["time_s"] = time_s
    df["is_black"] = (speeds == -1).astype(bool)

    out = df.iloc[start_save:end_save + 1].reset_index(drop=True)

    # 통계 메트릭
    spd = out["speed"].astype(float)
    t = out["time_s"]
    total_time = float(t.max()) if t.notna().any() else 0.0
    avg_speed = float(spd.sum() / total_time) if total_time > 0 else float("nan")
    over_mask = spd > 50.0
    over_cnt = int(over_mask.sum())
    over_speed_time = float(over_cnt / float(fps))
    total_over_speed_distance = float((spd[over_mask] / 108000.0).sum())
    part_over_speed_distance = float(((spd[over_mask] - 50.0) / 108000.0).sum())
    mean_spd = float(spd.mean()) if not spd.empty else float("nan")
    std_like = float((abs(mean_spd - spd)).mean()) if not spd.empty else float("nan")
    target_dev = float((abs(50.0 - spd)).mean()) if not spd.empty else float("nan")

    metrics_df = pd.DataFrame([{
        "총 주행 시간": total_time,
        "평균속력": avg_speed,
        "총 과속 시간": over_speed_time,
        "총 과속 거리(전체)": total_over_speed_distance,
        "총 과속 거리(넘은 거리만)": part_over_speed_distance,
        "속도 표준 편차": std_like,
        "타겟 속도 표준 편차": target_dev,
        "총 과속 횟수 (엑셀)": over_cnt,
    }])

    # 메타 시트
    meta_df = pd.DataFrame([
        {"key": "fps", "value": fps},
        {"key": "start_idx", "value": start_idx},
        {"key": "black_idx", "value": black_idx},
        {"key": "start_save", "value": start_save},
        {"key": "end_save", "value": end_save},
        {"key": "csv_rows", "value": len(df)},
        {"key": "exported_rows", "value": len(out)},
    ])

    # 엑셀 저장
    with pd.ExcelWriter(out_xlsx_path, engine="xlsxwriter") as xw:
        metrics_df.to_excel(xw, index=False, sheet_name="speed_time", startrow=0)
        out.to_excel(xw, index=False, sheet_name="speed_time", startrow=metrics_df.shape[0] + 2)
        meta_df.to_excel(xw, index=False, sheet_name="meta")

    if debug:
        print(f"[DEBUG] Exported rows: {len(out)}")
        print(f"[DEBUG] Start: {df.iloc[start_save]['filename']} (speed={df.iloc[start_save]['speed']})")
        print(f"[DEBUG] End: {df.iloc[end_save]['filename']} (speed={df.iloc[end_save]['speed']})")

    return out_xlsx_path


def main():
    ap = argparse.ArgumentParser(description="7-segment speed CSV to Excel exporter")
    ap.add_argument("--cls_csv", default="_cls_result.csv", help="입력 CSV 경로")
    ap.add_argument("--out_xlsx", default="_speed_time.xlsx", help="출력 XLSX 경로")
    ap.add_argument("--fps", type=int, default=30, help="FPS (기본값: 30)")
    ap.add_argument("--debug", action="store_true", help="디버그 출력")

    args = ap.parse_args()

    out = export_speed_xlsx_csv_only(
        cls_csv_path=args.cls_csv,
        out_xlsx_path=args.out_xlsx,
        fps=args.fps,
        debug=args.debug,
    )

    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
