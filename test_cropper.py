import cv2
import os
import numpy as np

VIDEO_PATH = "./source/sample_name/1/1-1.mp4"
SAVE_DIR = "./steering_analysis_30fps"
FRAME_INTERVAL = 1

ROI_STEER_PX = (1015, 945, 1075, 965)  # (x1, y1, x2, y2)

ZERO_CENTER_X = 29
X_POINT_BIAS = 3
TOLERANCE = 6

os.makedirs(SAVE_DIR, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"오류: 영상을 불러올 수 없습니다: {VIDEO_PATH}")
    raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"영상 원본 FPS: {fps} | 총 프레임: {total_frames}")

# ✅ 첫 프레임으로 해상도 확보 + ROI 비율 계산
ret, first = cap.read()
if not ret:
    print("오류: 첫 프레임을 읽지 못했습니다.")
    raise SystemExit(1)

ih, iw = first.shape[:2]
x1, y1, x2, y2 = ROI_STEER_PX
w = x2 - x1
h = y2 - y1

# 정규화(비율) 값
x_ratio = x1 / iw
y_ratio = y1 / ih
w_ratio = w / iw
h_ratio = h / ih

print(f"해상도: iw={iw}, ih={ih}")
print(f"ROI_STEER 비율: x={x_ratio:.8f}, y={y_ratio:.8f}, w={w_ratio:.8f}, h={h_ratio:.8f}")
print(f"ffmpeg crop 예시: crop=iw*{w_ratio:.8f}:ih*{h_ratio:.8f}:iw*{x_ratio:.8f}:ih*{y_ratio:.8f}")

# ✅ 다시 처음부터 읽기 위해 되감기
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_idx = 0
analyzed_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_INTERVAL == 0:
        ih, iw = frame.shape[:2]

        # ✅ 비율 → 픽셀 ROI로 환산 (매 프레임 동일 해상도면 항상 같은 값)
        x1p = int(round(iw * x_ratio))
        y1p = int(round(ih * y_ratio))
        wp  = int(round(iw * w_ratio))
        hp  = int(round(ih * h_ratio))
        x2p = x1p + wp
        y2p = y1p + hp

        steer_crop = frame[y1p:y2p, x1p:x2p].copy()
        h_crop, w_crop = steer_crop.shape[:2]

        gray = cv2.cvtColor(steer_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        M = cv2.moments(thresh)
        state = "N/A"
        cx_calibrated = -1

        if M["m00"] > 0:
            cx_raw = int(M["m10"] / M["m00"])
            cx_calibrated = cx_raw + X_POINT_BIAS

            if cx_calibrated < (ZERO_CENTER_X - TOLERANCE):
                state = "L"
            elif cx_calibrated > (ZERO_CENTER_X + TOLERANCE):
                state = "R"
            else:
                state = "0"

        if analyzed_count % 30 == 0:
            border_color = (0, 255, 0) if state == "0" else ((0, 0, 255) if state == "R" else (255, 0, 0))
            debug_img = cv2.copyMakeBorder(steer_crop, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=border_color)

            cv2.line(debug_img, (ZERO_CENTER_X + 5, 0), (ZERO_CENTER_X + 5, h_crop + 10), (255, 255, 255), 1)
            if cx_calibrated != -1:
                cv2.circle(debug_img, (cx_calibrated + 5, (h_crop // 2) + 5), 3, (0, 255, 255), -1)

            cv2.imwrite(f"{SAVE_DIR}/sample_f{frame_idx}_{state}.png", debug_img)
            print(f"Saving Sample -> Frame: {frame_idx} | State: {state} | ROI(px)=({x1p},{y1p})~({x2p},{y2p})")

        analyzed_count += 1

    frame_idx += 1
    if frame_idx % 300 == 0:
        print(f"진행 중... {frame_idx}/{total_frames} 프레임 처리 완료")

cap.release()
print(f"\n총 {analyzed_count}개 프레임 분석 완료.")
