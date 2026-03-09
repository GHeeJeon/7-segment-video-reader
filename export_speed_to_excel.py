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
    steer_csv_path: str = None,
) -> str:
    if not os.path.isfile(cls_csv_path):
        raise FileNotFoundError(f"CSV not found: {cls_csv_path}")

    df = pd.read_csv(cls_csv_path)
    # ... (existing logic for df processing)
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
        # 속력 변화가 없어도 전체 구간을 저장하도록 함
        start_idx = 0
        start_save = 0
        end_save = len(speeds) - 1
    else:
        start_save = start_idx
        right_limit = black_idx if black_idx is not None else len(df)

        # 블랙 직전 0 제거
        j = right_limit - 1
        while j >= start_idx and speeds.iloc[j] == 0:
            j -= 1
        end_save = j
    
    # end_save가 start_save보다 작아지는 것 방지
    if end_save < start_save:
        end_save = start_save

    # ----- 시간 및 데이터 정리 -----
    out = pd.DataFrame()
    if len(df) > 0:
        time_s = [None] * len(df)
        for i in range(max(0, start_idx), min(len(df), end_save + 1)):
            time_s[i] = (i - start_idx) / float(fps)

        df["speed"] = speeds
        df["time_s"] = time_s
        df["is_black"] = (speeds == -1).astype(bool)
        out = df.iloc[start_save:end_save + 1].reset_index(drop=True)
        if "time_s" in out.columns:
            out["time_s"] = pd.to_numeric(out["time_s"], errors="coerce").ffill().fillna(0.0)
        out["is_black"] = out["speed"].eq(-1)
    # ----- check 컬럼 -----
    if not out.empty:
        speeds_list = out["speed"].tolist()
        out["check"] = ["Y" if is_spike_among_plateaus(speeds_list, i) else "N" for i in range(len(speeds_list))]
    else:
        out["check"] = []
        # Ensure 'speed' and 'time_s' columns exist for downstream processing if 'out' is empty
        # This is a minimal set to prevent KeyErrors later when accessing these columns
        out["speed"] = []
        out["time_s"] = []
        out["is_black"] = []
        out["rapid_accel"] = []
        out["rapid_decel"] = []
        out["speed_30f_ago"] = []


    # ---------- 통계 메트릭 ----------
    spd = out["speed"].astype(float) if "speed" in out.columns else pd.Series(dtype=float)
    total_time = float((end_save - start_save + 1) / fps) if end_save >= start_save else 0.0
    avg_speed_requested = float(spd.sum() / (total_time * fps)) if total_time > 0 and len(spd) > 0 else 0.0
    
    # ---------------- 과속 기준 ----------------
    OVER_SPEED_KMH = 60
    basic_mask = spd > OVER_SPEED_KMH
    sustained_mask = pd.Series([False] * len(spd))

    if not out.empty and basic_mask.any():
        run_len = 0
        for i, flag in enumerate(basic_mask):
            if flag:
                run_len += 1
                if run_len >= fps:
                    sustained_mask.iloc[i] = True
            else:
                run_len = 0

    # ---------------- 기본 과속 기준 통계 ----------------
    over_frame_cnt = int(basic_mask.sum())
    prev_basic = basic_mask.shift(1).fillna(False)
    basic_start = basic_mask & (~prev_basic)
    basic_segments = int(basic_start.sum())
    over_speed_time = float(over_frame_cnt / float(fps))
    den = float(fps) * 3600.0
    total_over_speed_distance = float((spd[basic_mask] / den).sum()) if not spd.empty else 0.0
    part_over_speed_distance = float(((spd[basic_mask] - OVER_SPEED_KMH) / den).sum()) if not spd.empty else 0.0

    # ---------------- 1초(30프레임) 이상 지속 과속 기준 ----------------
    prev_sustained = sustained_mask.shift(1).fillna(False)
    sustained_start = sustained_mask & (~prev_sustained)
    sustained_segments = int(sustained_start.sum())

    # ---------------- 편차 ----------------
    mean_spd = float(avg_speed_requested)
    std_pop = float(np.sqrt(((spd - mean_spd) ** 2).mean())) if not spd.empty else 0.0
    target_rmse = float(np.sqrt(((OVER_SPEED_KMH - spd) ** 2).mean())) if not spd.empty else 0.0

    # ---------------- 40~60km/h 구간 ----------------
    mask_40_60 = (spd >= 40) & (spd <= 60)
    time_40_60 = float(mask_40_60.sum() / float(fps))
    ratio_40_60 = float((time_40_60 / total_time * 100)) if total_time > 0 else 0.0

    # ---------------- 급가속 / 급감속 ----------------
    delta_v_1s = spd - spd.shift(fps)
    accel_mask = delta_v_1s >= 10.0
    decel_mask = delta_v_1s <= -10.0
    if len(accel_mask) > fps:
        accel_mask.iloc[:fps] = False
        decel_mask.iloc[:fps] = False
    
    accel_start = accel_mask & (~accel_mask.shift(1).fillna(False))
    decel_start = decel_mask & (~decel_mask.shift(1).fillna(False))
    
    rapid_accel_count = int(accel_start.sum())
    rapid_decel_count = int(decel_start.sum())

    if not out.empty:
        out["rapid_accel"] = ["Y" if flag else "N" for flag in accel_mask]
        out["rapid_decel"] = ["Y" if flag else "N" for flag in decel_mask]
        out["speed_30f_ago"] = spd.shift(fps)
        out.loc[~(accel_mask | decel_mask), "speed_30f_ago"] = None
    else:
        out["rapid_accel"] = []
        out["rapid_decel"] = []
        out["speed_30f_ago"] = []

    # ---------- 메트릭 테이블 통합 ----------
    metrics_df = pd.DataFrame([{
        "전체 프레임 수": len(df),
        "속력 감지 시작": start_save,
        "속력 감지 종료": end_save,
        "총 소요 시간(초)": total_time,
        "평균 속력(km/h)": avg_speed_requested,
        "최대 속력(km/h)": out["speed"].max() if not out.empty else 0,
        "60km/h 초과 시간(초)": over_speed_time,
        "60km/h 초과 구간 수": basic_segments,
        "1초이상 60km/h 초과 구간 수": sustained_segments,
        "급가속 횟수(10km/h/s)": rapid_accel_count,
        "급감속 횟수(10km/h/s)": rapid_decel_count,
        "RMSE(60km/h 기준)": target_rmse,
        "표준편차": std_pop,
        "40~60km/h 비율(%)": ratio_40_60,
        "총 과속 거리(전체)(km)": total_over_speed_distance,
        "총 과속 거리(초과분만)(km)": part_over_speed_distance
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

    # ---------------- Steering 데이터 읽기 ----------------
    steer_df = None
    if steer_csv_path and os.path.isfile(steer_csv_path):
        try:
            steer_df = pd.read_csv(steer_csv_path)
        except Exception as e:
            print(f"[경고] Steering CSV 읽기 실패: {e}")

    # ----- 엑셀 저장 -----
    with pd.ExcelWriter(out_xlsx_path, engine="xlsxwriter") as writer:
        metrics_df.to_excel(writer, index=False, sheet_name="speed_time", startrow=0)
        out_startrow = metrics_df.shape[0] + 2
        out_client.to_excel(writer, index=False, sheet_name="speed_time", startrow=out_startrow)
        meta_df.to_excel(writer, index=False, sheet_name="meta")
        if steer_df is not None:
            # 1. Steering 통계 계산 (Y 카운트)
            steer_metrics = {}
            for n in range(1, 6):
                c3d = f"3d_srr_{n}px"
                c58 = f"58d_srr_{n}px"
                if c3d in steer_df.columns:
                    steer_metrics[f"3D-SRR({n}px) 발생횟수"] = (steer_df[c3d] == "Y").sum()
                if c58 in steer_df.columns:
                    steer_metrics[f"58D-SRR({n}px) 발생횟수"] = (steer_df[c58] == "Y").sum()
            
            steer_metrics_df = pd.DataFrame([steer_metrics])

            # 2. Steering 컬럼 한글화 및 데이터 정리
            steer_rename = {
                "frame_idx": "프레임번호",
                "time_sec": "시간(초)",
                "cx_raw": "감지중심X_raw",
                "steer_px_offset": "핸들UI_이동px(0기준)",
                "steer_label": "핸들위치_라벨",
                "is_noise": "노이즈의심_YN",
                "lnr_1px": "3D-LNR(1px)", "3d_srr_1px": "3D-SRR_Check(1px)", "58d_srr_1px": "58D-SRR_Check(1px)",
                "lnr_2px": "3D-LNR(2px)", "3d_srr_2px": "3D-SRR_Check(2px)", "58d_srr_2px": "58D-SRR_Check(2px)",
                "lnr_3px": "3D-LNR(3px)", "3d_srr_3px": "3D-SRR_Check(3px)", "58d_srr_3px": "58D-SRR_Check(3px)",
                "lnr_4px": "3D-LNR(4px)", "3d_srr_4px": "3D-SRR_Check(4px)", "58d_srr_4px": "58D-SRR_Check(4px)",
                "lnr_5px": "3D-LNR(5px)", "3d_srr_5px": "3D-SRR_Check(5px)", "58d_srr_5px": "58D-SRR_Check(5px)"
            }
            
            steer_client = steer_df.rename(columns=steer_rename)
            available_cols = [v for k, v in steer_rename.items() if v in steer_client.columns]
            steer_client = steer_client[available_cols]
            
            # 3. 데이터 쓰기 (통계 상단 배치)
            sheet_name = "steering_result"
            steer_metrics_df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=0)
            steer_data_startrow = steer_metrics_df.shape[0] + 2
            steer_client.to_excel(writer, index=False, sheet_name=sheet_name, startrow=steer_data_startrow)


    if debug:
        print(f"[DEBUG] Exported rows: {len(out)}")
        print(f"[DEBUG] Start: {df.iloc[start_save]['filename']}")
        print(f"[DEBUG] End: {df.iloc[end_save]['filename']}")
        print(f"[DEBUG] Show tech columns: {show_tech_cols}")

    return out_xlsx_path


def main():
    ap = argparse.ArgumentParser(description="7-segment speed CSV to Excel exporter")
    ap.add_argument("-f", "--fps", type=int, default=30, help="FPS (기본값: 30)")
    ap.add_argument("-a", "--all-cols", action="store_true", help="모든 컬럼 표시 (num_digits, pred_number, preds, confs, dists, states_per_digit 포함)")

    args = ap.parse_args()

    out = export_speed_xlsx(
        cls_csv_path="_cls_result.csv",
        out_xlsx_path="_speed_time.xlsx",
        fps=args.fps,
        debug=False,
        show_tech_cols=args.all_cols,
    )

    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()