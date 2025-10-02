"""
7세그 속도 + 시간 엑셀 내보내기 (필터 적용 + 종료 전 0구간 제거 + 메트릭 상단 1회 표기 + check 컬럼 추가)

- classify_sevenseg.py가 생성한 분류 CSV를 읽어 시간 열을 포함한 엑셀 파일을 작성합니다.
- 엑셀에는 다음 구간만 저장합니다.
  * 시작: 속도가 0 → ≥1 로 처음 전이되는 프레임(t=0), 단 그 직전 0 프레임도 포함
  * 끝: 시작 이후 첫 블랙(-1) 프레임 바로 직전의 마지막 "비0 속도" 프레임
    - 블랙 직전 연속된 0 구간은 제외 (마지막 0 한 프레임은 유지)
  * 블랙 프레임이 없을 경우: 파일 끝까지 포함 (말단 0 구간 강제 제거 X)
- check 컬럼: 단발 튐(앞뒤가 같은데 가운데만 다른 경우)을 Y로 마킹
"""

import os
import argparse
import numpy as np
import pandas as pd

# ===== 과속 임계 =====
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

    # 시작 프레임
    start_save = start_idx

    # 종료: 블랙 직전 0 제거
    right_limit = black_idx if black_idx is not None else len(df)
    j = right_limit - 1
    while j >= start_idx and speeds.iloc[j] == 0:
        j -= 1
    trimmed_end = j
    end_save = trimmed_end

    # 시간 계산
    time_s = [None] * len(df)
    for i in range(start_idx, end_save + 1):
        time_s[i] = (i - start_idx) / float(fps)

    df["speed"] = speeds               # km/h (정수)
    df["time_s"] = time_s              # seconds
    df["is_black"] = (speeds == -1).astype(bool)

    out = df.iloc[start_save:end_save + 1].reset_index(drop=True)

    # time_s 보정
    out["time_s"] = pd.to_numeric(out["time_s"], errors="coerce").ffill().fillna(0.0)
    
    out["is_black"] = out["speed"].eq(-1)

    # check 컬럼 추가
    speeds_list = out["speed"].tolist()
    check_col = ["Y" if is_spike_among_plateaus(speeds_list, i) else "N" for i in range(len(speeds_list))]
    out["check"] = check_col

    # ---------- 통계 메트릭 ----------
    spd = out["speed"].astype(float)     # km/h
    t = out["time_s"].astype(float)      # s

    # 총 주행 시간: (시작 ~ 끝) / fps(예: 30)
    total_time = float((end_save - start_save + 1) / fps)

    # 참고: 등간격 샘플 평균(권장)
    # avg_speed_mean = float(spd.mean()) if not spd.empty else float("nan")

    # 평균속력 (요청식): ∑속도 / 총 주행 시간
    avg_speed_requested = float(spd.sum() / (total_time * fps)) if total_time > 0 else float("nan")

    # ---------------- 새로운 과속 정의 ----------------
    # 60km/h 초과 여부 마스크
    raw_mask = spd > 60.0    
    
    # 연속 run-length 계산
    over_mask = np.zeros_like(raw_mask, dtype=bool)
    if raw_mask.any():
        run_len = 0
        for i, flag in enumerate(raw_mask):
            if flag:
                run_len += 1
                if run_len >= fps:  # 30프레임(1초) 이상이면 과속 인정
                    over_mask[i] = True
            else:
                run_len = 0

    # (1) 과속 프레임 수
    over_frame_cnt = int(over_mask.sum())

    # (2) 과속 '구간' 횟수 (연속 1초 이상 구간 단위)
    prev_over = np.roll(over_mask, 1)
    prev_over[0] = False
    over_start = over_mask & (~prev_over)
    over_segments = int(over_start.sum())

    # (3) 총 과속 시간(s): 과속 프레임 수 / fps
    over_speed_time = float(over_frame_cnt / float(fps))

    # (4) 총 과속 거리(전체): ∑ v/(fps*3600) [km]
    den = float(fps) * 3600.0
    total_over_speed_distance = float((spd[over_mask] / den).sum())

    # (5) 총 과속 거리(초과분만, 기준 60km/h): ∑ (v-60)/(fps*3600)
    part_over_speed_distance = float(((spd[over_mask] - 60.0) / den).sum())

    # ---------------- 편차 계산 ----------------
    mean_spd = float(avg_speed_requested) if not np.isnan(avg_speed_requested) else float("nan")

    # (6) 표준편차(모집단)
    std_pop = float(np.sqrt(((spd - mean_spd) ** 2).mean())) if not spd.empty else float("nan")

    # (7) 타깃 속도 RMSE (60 기준)
    target_rmse = float(np.sqrt(((60.0 - spd) ** 2).mean())) if not spd.empty else float("nan")

    # 🔹 50~60km/h 주행 시간 및 비율
    mask_50_60 = (spd >= 50) & (spd <= 60)
    time_50_60 = float(mask_50_60.sum() / float(fps))  # 초 단위
    ratio_50_60 = float((time_50_60 / total_time * 100)) if total_time > 0 else float("nan")


    # 메트릭 표 (가독성을 위해 단위 병기)
    metrics_df = pd.DataFrame([{
        "총 주행 시간(s)": total_time,
        "평균속력 [km/h]": avg_speed_requested,
        "총 과속 시간(s)": over_speed_time,
        "총 과속 거리(전체)(km)": total_over_speed_distance,
        "총 과속 거리(초과분만)(km)": part_over_speed_distance,
        "속도 표준편차(정의) [km/h]": std_pop,
        "Target RMSE(target=60) [km/h]": target_rmse,
        "과속 프레임 수(>60)": over_frame_cnt,
        "과속 횟수(구간, >60)": over_segments,
        "50~60km/h 주행 시간(s)": time_50_60,
        "50~60km/h 비율(%)": ratio_50_60,
    }])

    # 메타 시트
    meta_df = pd.DataFrame([
        {"key": "fps", "value": fps},
        {"key": "overspeed_kmh", "value": OVER_SPEED_KMH},
        {"key": "start_idx", "value": start_idx},
        {"key": "black_idx", "value": black_idx},
        {"key": "start_save", "value": start_save},
        {"key": "end_save", "value": end_save},
        {"key": "csv_rows", "value": len(df)},
        {"key": "exported_rows", "value": len(out)},
    ])

    # 엑셀 저장
    with pd.ExcelWriter(out_xlsx_path, engine="xlsxwriter") as writer:
        metrics_df.to_excel(writer, index=False, sheet_name="speed_time", startrow=0)
        out_startrow = metrics_df.shape[0] + 2
        out.to_excel(writer, index=False, sheet_name="speed_time", startrow=out_startrow)
        meta_df.to_excel(writer, index=False, sheet_name="meta")

        # notes: time_s 헤더 오른쪽부터 가로로 표기
        ws = writer.sheets["speed_time"]
        header_row = out_startrow  # out의 헤더가 써진 행
        # time_s 컬럼 찾기
        try:
            time_s_col = out.columns.get_loc("time_s")
        except KeyError:
            time_s_col = len(out.columns)  # 안전장치: 못 찾으면 맨 끝 기준

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

    out = export_speed_xlsx(
        cls_csv_path=args.cls_csv,
        out_xlsx_path=args.out_xlsx,
        fps=args.fps,
        debug=args.debug,
    )

    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
