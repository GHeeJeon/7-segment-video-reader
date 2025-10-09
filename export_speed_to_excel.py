"""
7세그 속도 + 시간 엑셀 내보내기 (필터 적용 + 종료 전 0구간 제거 + 메트릭 상단 1회 표기 + check 컬럼 + 급가속/급감속 감지)

- classify_sevenseg.py가 생성한 분류 CSV를 읽어 시간 열을 포함한 엑셀 파일을 작성합니다.
- 엑셀에는 다음 구간만 저장합니다.
  * 시작: 속도가 0 → ≥1 로 처음 전이되는 프레임(t=0), 단 그 직전 0 프레임도 포함
  * 끝: 시작 이후 첫 블랙(-1) 프레임 바로 직전의 마지막 "비0 속도" 프레임
    - 블랙 직전 연속된 0 구간은 제외 (마지막 0 한 프레임은 유지)
  * 블랙 프레임이 없을 경우: 파일 끝까지 포함 (말단 0 구간 강제 제거 X)
- check 컬럼: 단발 튐(앞뒤가 같은데 가운데만 다른 경우)을 Y로 마킹
- 급가속/급감속 컬럼: 1초(30프레임) 전 대비 ±10km/h 이상 변화 시 Y
"""

import os
import argparse
import numpy as np
import pandas as pd

# ===== 설정 상수 =====
OVER_SPEED_KMH = 60.0  # 과속 기준(km/h)

def is_spike_among_plateaus(speeds: list[int], i: int) -> bool:
    """단발 튐 검출: 앞뒤가 같고 현재만 다른 경우"""
    if i <= 0 or i >= len(speeds) - 1:
        return False
    before, now, after = speeds[i - 1], speeds[i], speeds[i + 1]

    # (1) 가운데만 튄 경우
    if now != before and now != after and before == after:
        return True

    # (2) 정상 추세 (가속/감속)
    if (before < now < after) or (before > now > after):
        return False

    # (3) 앞뒤 평균에서 크게 튄 경우 (옵션)
    diff = abs(now - (before + after) / 2)
    if diff >= 10:
        return True

    return False


def export_speed_xlsx(
    cls_csv_path: str = "_cls_result.csv",
    out_xlsx_path: str = "_speed_time.xlsx",
    fps: int = 30,
    debug: bool = False,
    show_tech_cols: bool = False,
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

    # ----- 구간 설정 -----
    # 말단 연속된 -1 제거
    end_idx = len(speeds) - 1
    last_valid_idx = end_idx
    while last_valid_idx >= 0 and speeds.iloc[last_valid_idx] == -1:
        last_valid_idx -= 1
    black_idx = last_valid_idx + 1 if last_valid_idx < end_idx else None

    # 시작 프레임 찾기 (0→≥1 전이)
    start_idx = None
    for i in range(1, len(speeds)):
        if speeds.iloc[i] >= 1 and speeds.iloc[i - 1] == 0:
            start_idx = i
            break
    if start_idx is None:
        print("No valid start index found")
        return out_xlsx_path

    start_save = start_idx
    right_limit = black_idx if black_idx is not None else len(df)

    # 블랙 직전 0 제거
    j = right_limit - 1
    while j >= start_idx and speeds.iloc[j] == 0:
        j -= 1
    end_save = j

    # ----- 시간 및 데이터 정리 -----
    time_s = [None] * len(df)
    for i in range(start_idx, end_save + 1):
        time_s[i] = (i - start_idx) / float(fps)

    df["speed"] = speeds
    df["time_s"] = time_s
    df["is_black"] = (speeds == -1).astype(bool)
    out = df.iloc[start_save:end_save + 1].reset_index(drop=True)
    out["time_s"] = pd.to_numeric(out["time_s"], errors="coerce").ffill().fillna(0.0)
    out["is_black"] = out["speed"].eq(-1)

    # ----- check 컬럼 -----
    speeds_list = out["speed"].tolist()
    out["check"] = ["Y" if is_spike_among_plateaus(speeds_list, i) else "N" for i in range(len(speeds_list))]

    # ---------- 통계 메트릭 ----------
    spd = out["speed"].astype(float)
    total_time = float((end_save - start_save + 1) / fps)
    avg_speed_requested = float(spd.sum() / (total_time * fps)) if total_time > 0 else float("nan")
    
    # ---------------- 과속 기준 ----------------
    basic_mask = spd > OVER_SPEED_KMH        # 기본 과속: 60 초과
    sustained_mask = np.zeros_like(basic_mask, dtype=bool)  # 1초(30프레임) 이상 지속

    if basic_mask.any():
        run_len = 0
        for i, flag in enumerate(basic_mask):
            if flag:
                run_len += 1
                if run_len >= fps:           # 1초 이상 지속이면 True
                    sustained_mask[i] = True
            else:
                run_len = 0

    # ---------------- 기본 과속 기준 통계 ----------------
    over_frame_cnt = int(basic_mask.sum())
    prev_basic = np.roll(basic_mask, 1)
    prev_basic[0] = False
    basic_start = basic_mask & (~prev_basic)
    basic_segments = int(basic_start.sum())                   # 기본 과속 구간 수
    over_speed_time = float(over_frame_cnt / float(fps))
    den = float(fps) * 3600.0
    total_over_speed_distance = float((spd[basic_mask] / den).sum())
    part_over_speed_distance = float(((spd[basic_mask] - OVER_SPEED_KMH) / den).sum())

    # ---------------- 1초(30프레임) 이상 지속 과속 기준 ----------------
    prev_sustained = np.roll(sustained_mask, 1)
    prev_sustained[0] = False
    sustained_start = sustained_mask & (~prev_sustained)
    sustained_segments = int(sustained_start.sum())

    # ---------------- 편차 ----------------
    mean_spd = float(avg_speed_requested) if not np.isnan(avg_speed_requested) else float("nan")
    std_pop = float(np.sqrt(((spd - mean_spd) ** 2).mean())) if not spd.empty else float("nan")
    target_rmse = float(np.sqrt(((OVER_SPEED_KMH - spd) ** 2).mean())) if not spd.empty else float("nan")

    # ---------------- 50~60km/h 구간 ----------------
    mask_50_60 = (spd >= 50) & (spd <= 60)
    time_50_60 = float(mask_50_60.sum() / float(fps))
    ratio_50_60 = float((time_50_60 / total_time * 100)) if total_time > 0 else float("nan")

    # ---------------- 급가속 / 급감속 ----------------
    delta_v_1s = spd - spd.shift(fps)
    accel_mask = delta_v_1s >= 10.0
    decel_mask = delta_v_1s <= -10.0
    accel_mask.iloc[:fps] = False
    decel_mask.iloc[:fps] = False

    accel_start = accel_mask & (~np.roll(accel_mask, 1))
    decel_start = decel_mask & (~np.roll(decel_mask, 1))
    accel_start.iloc[0] = False
    decel_start.iloc[0] = False

    accel_segments = int(accel_start.sum())
    decel_segments = int(decel_start.sum())

    out["rapid_accel"] = ["Y" if flag else "N" for flag in accel_mask]
    out["rapid_decel"] = ["Y" if flag else "N" for flag in decel_mask]

    # ---------------- 30프레임 이전 속도 ----------------
    speed_30frames_ago = spd.shift(fps)
    out["speed_30f_ago"] = speed_30frames_ago
    # 급가속도 급감속도 아닌 경우 빈 값으로 표시
    out.loc[~(accel_mask | decel_mask), "speed_30f_ago"] = None

    # ----- 메트릭 테이블 -----
    metrics_df = pd.DataFrame([{
        "총 주행 시간(s)": total_time,
        "평균속력 [km/h]": avg_speed_requested,
        "총 과속 시간(s)": over_speed_time,
        "총 과속 거리(전체)(km)": total_over_speed_distance,
        "총 과속 거리(초과분만)(km)": part_over_speed_distance,
        "속도 표준편차(정의) [km/h]": std_pop,
        "Target RMSE(target=60) [km/h]": target_rmse,
        "과속 프레임 수(>60)": over_frame_cnt,
        "과속 구간(기본, >60)": basic_segments,
        "과속 구간(1초 이상 지속, >60)": sustained_segments,
        "급가속 횟수(Δv≥+10)": accel_segments,
        "급감속 횟수(Δv≤-10)": decel_segments,
        "50~60km/h 주행 시간(s)": time_50_60,
        "50~60km/h 비율(%)": ratio_50_60,
    }])

    # ---------------- 메타 시트 ----------------
    meta_df = pd.DataFrame([
        {"key": "fps", "value": fps},
        {"key": "overspeed_kmh", "value": OVER_SPEED_KMH},
        {"key": "start_idx", "value": start_idx},
        {"key": "black_idx", "value": black_idx},
        {"key": "start_save", "value": start_save},
        {"key": "end_save", "value": end_save},
        {"key": "csv_rows", "value": len(df)},
        {"key": "exported_rows", "value": len(out)},
        {"key": "show_tech_cols", "value": show_tech_cols},
    ])

    # ----- 클라이언트용 컬럼 선택 -----
    if show_tech_cols:
        # 기술 컬럼 포함 (모든 컬럼 표시)
        base_cols = ["filename", "num_digits", "pred_number", "preds", "confs", "dists", "states_per_digit"]
        user_cols = ["speed", "time_s", "is_black", "check", "rapid_accel", "rapid_decel", "speed_30f_ago"]
        
        # CSV에 실제로 존재하는 컬럼만 선택
        available_base = [col for col in base_cols if col in out.columns]
        all_cols = available_base + user_cols
        out_client = out[all_cols].copy()
        
        # 한글 컬럼명 변경 (기술 컬럼은 그대로, 사용자 컬럼만 한글화)
        rename_dict = {
            "speed": "속도(km/h)",
            "time_s": "시간(초)",
            "is_black": "블랙프레임",
            "check": "노이즈감지",
            "rapid_accel": "급가속_YN",
            "rapid_decel": "급감속_YN",
            "speed_30f_ago": "30프레임전_속도(km/h)"
        }
    else:
        # 기본 모드: 클라이언트용 컬럼만
        out_client = out[["filename", "speed", "time_s", "is_black", "check", "rapid_accel", "rapid_decel", "speed_30f_ago"]].copy()
        rename_dict = {
            "filename": "파일명",
            "speed": "속도(km/h)",
            "time_s": "시간(초)",
            "is_black": "블랙프레임",
            "check": "노이즈감지",
            "rapid_accel": "급가속_YN",
            "rapid_decel": "급감속_YN",
            "speed_30f_ago": "30프레임전_속도(km/h)"
        }
    
    out_client = out_client.rename(columns=rename_dict)

    # ----- 엑셀 저장 -----
    with pd.ExcelWriter(out_xlsx_path, engine="xlsxwriter") as writer:
        metrics_df.to_excel(writer, index=False, sheet_name="speed_time", startrow=0)
        out_startrow = metrics_df.shape[0] + 2
        out_client.to_excel(writer, index=False, sheet_name="speed_time", startrow=out_startrow)
        meta_df.to_excel(writer, index=False, sheet_name="meta")

    if debug:
        print(f"[DEBUG] Exported rows: {len(out)}")
        print(f"[DEBUG] Start: {df.iloc[start_save]['filename']}")
        print(f"[DEBUG] End: {df.iloc[end_save]['filename']}")
        print(f"[DEBUG] Show tech columns: {show_tech_cols}")

    return out_xlsx_path


def main():
    ap = argparse.ArgumentParser(description="7-segment speed CSV to Excel exporter")
    ap.add_argument("-c", "--cls_csv", default="_cls_result.csv", help="입력 CSV 경로")
    ap.add_argument("-o", "--out_xlsx", default="_speed_time.xlsx", help="출력 XLSX 경로")
    ap.add_argument("-f", "--fps", type=int, default=30, help="FPS (기본값: 30)")
    ap.add_argument("-d", "--debug", action="store_true", help="디버그 출력")
    ap.add_argument(
    "-a", "--all-cols",
    action="store_true",
    help="모든 컬럼 표시 (num_digits, pred_number, preds, confs, dists, states_per_digit 포함)")
    
    args = ap.parse_args()

    out = export_speed_xlsx(
        cls_csv_path=args.cls_csv,
        out_xlsx_path=args.out_xlsx,
        fps=args.fps,
        debug=args.debug,
        show_tech_cols=args.all_cols,
    )

    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()