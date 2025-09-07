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

    # 총 주행 시간: 유효구간 길이 = max(time) - min(time)
    total_time = float(t.max() - t.min()) if t.notna().any() else 0.0

    # 평균속력 (요청식): ∑속도 / 총 주행 시간
    avg_speed_requested = float(spd.sum() / total_time) if total_time > 0 else float("nan")

    # 참고: 등간격 샘플 평균(권장)
    avg_speed_mean = float(spd.mean()) if not spd.empty else float("nan")

    # 과속 마스크
    over_mask = spd > OVER_SPEED_KMH

    # 과속 프레임 수(참고)
    over_frame_cnt = int(over_mask.sum())

    # 과속 '구간' 횟수: v>OVER_SPEED_KMH 이 새로 시작되는 순간의 개수
    # (이전 프레임이 OVER_SPEED_KMH 이하이고 현재가 OVER_SPEED_KMH 초과이면 카운트)
    prev_over = (spd.shift(1, fill_value=0) > OVER_SPEED_KMH)
    over_start = over_mask & (~prev_over)
    over_segments = int(over_start.sum())

    # 총 과속 시간(s): 과속 프레임 수 / fps
    over_speed_time = float(over_frame_cnt / float(fps))

    # 거리 환산 상수: v[km/h] -> v/(fps*3600) [km/frame]
    den = float(fps) * 3600.0

    # 총 과속 거리(전체): ∑(v>OVER_SPEED_KMH) v / (fps*3600)
    total_over_speed_distance = float((spd[over_mask] / den).sum())

    # 총 과속 거리(초과분만): ∑(v>OVER_SPEED_KMH) (v-OVER_SPEED_KMH) / (fps*3600)
    part_over_speed_distance = float(((spd[over_mask] - OVER_SPEED_KMH) / den).sum())

    # 편차들
    mean_spd = float(avg_speed_mean) if not np.isnan(avg_speed_mean) else float("nan")

    # 표준정의 표준편차(모집단)
    std_pop = float(np.sqrt(((spd - mean_spd) ** 2).mean())) if not spd.empty else float("nan")

    # (참고) 타깃 RMSE
    target_rmse = float(np.sqrt(((OVER_SPEED_KMH - spd) ** 2).mean())) if not spd.empty else float("nan")

    # 메트릭 표 (가독성을 위해 단위 병기)
    metrics_df = pd.DataFrame([{
        "총 주행 시간(s)": total_time,
        "평균속력(요청식: ∑v / T) [km/h]": avg_speed_requested,
        "평균속력(권장: mean) [km/h]": avg_speed_mean,
        "총 과속 시간(s)": over_speed_time,
        "총 과속 거리(전체)(km)": total_over_speed_distance,
        "총 과속 거리(초과분만)(km)": part_over_speed_distance,
        "속도 표준편차(정의) [km/h]": std_pop,
        "Target RMSE(target=60) [km/h]": target_rmse,
        "과속 프레임 수(>60)": over_frame_cnt,
        "과속 횟수(구간, >60)": over_segments,
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

        # notes = [
        #     f"total time : {total_time:.2f} s",
        #     f"average speed : {avg_speed_requested:.3f} km/h (속도합/총시간)",
        #     f"Over speed time : {over_speed_time:.2f} s (과속 프레임 {over_frame_cnt}개 / fps {fps})",
        #     f"total over speed distance : {total_over_speed_distance:.3f} km (∑ v/(fps·3600), v>)",
        #     f"part over speed distance : {part_over_speed_distance:.3f} km (∑ (OVER_SPEED_KMH)/(fps·3600), v>OVER_SPEED_KMH)",
        #     f"over speed count : {over_segments} 회 (v>OVER_SPEED_KMH 구간 시작 횟수)",
        # ]

        # notes = [
        # "is_black: True → 블랙 프레임, False → 정상 프레임",
        # "check: Y → 단발 튐(앞뒤 같고 가운데만 다른 경우), N → 정상",
        # ]

        # base_col = time_s_col + 1
        # for k, text in enumerate(notes):
        #     ws.write(header_row, base_col + k, text)

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
