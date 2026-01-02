import cv2
import os
import numpy as np

# ==========================================
# 1. 환경 설정 및 보정값 적용
# ==========================================
VIDEO_PATH = "./source/sample_name/1/1-1.mp4"
SAVE_DIR = "./steering_analysis_30fps" 
# 30fps 추출을 위해 간격을 1프레임으로 설정 (영상 자체가 30fps 기준일 때)
# 만약 영상이 60fps라면 2로 설정하여 30fps 효과를 낼 수 있습니다.
FRAME_INTERVAL = 1 

ROI_STEER = (942, 965, 1015, 1075)

# 피드백 반영 보정값
ZERO_CENTER_X = 29  # 기존 30에서 왼쪽으로 1px 이동 (-1)
X_POINT_BIAS = 3    # 노란 점(감지점)을 오른쪽으로 3px 보정 (+3)
TOLERANCE = 6       # '0' 상태 판정 허용 범위

# ==========================================
# 2. 초기화
# ==========================================
os.makedirs(SAVE_DIR, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"오류: 영상을 불러올 수 없습니다: {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"영상 원본 FPS: {fps} | 총 프레임: {total_frames}")

# ==========================================
# 3. 전체 프레임 분석 루프 (30fps 단위)
# ==========================================
frame_idx = 0
analyzed_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 설정한 프레임 간격마다 분석 수행
    if frame_idx % FRAME_INTERVAL == 0:
        # [A] ROI 크롭
        y1, y2, x1, x2 = ROI_STEER
        steer_crop = frame[y1:y2, x1:x2].copy()
        h, w = steer_crop.shape[:2]

        # [B] 인디케이터 위치 감지 및 보정 적용
        gray = cv2.cvtColor(steer_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        M = cv2.moments(thresh)
        state = "N/A"
        cx_calibrated = -1

        if M["m00"] > 0:
            # 원본 중심점에서 +3px 오른쪽으로 보정
            cx_raw = int(M["m10"] / M["m00"])
            cx_calibrated = cx_raw + X_POINT_BIAS
            
            # 보정된 좌표 기준 L / 0 / R 판별
            if cx_calibrated < (ZERO_CENTER_X - TOLERANCE):
                state = "L"
            elif cx_calibrated > (ZERO_CENTER_X + TOLERANCE):
                state = "R"
            else:
                state = "0"

        # [C] 30fps 결과 출력 (모든 이미지를 저장하면 용량이 크므로 콘솔 위주로 확인)
        # 확인을 위해 1초(30프레임)마다 한 장씩만 샘플 저장
        if analyzed_count % 30 == 0:
            border_color = (0, 255, 0) if state == "0" else ((0, 0, 255) if state == "R" else (255, 0, 0))
            debug_img = cv2.copyMakeBorder(steer_crop, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=border_color)
            
            # 보정된 기준선(29px) 및 보정된 점 표시
            cv2.line(debug_img, (ZERO_CENTER_X + 5, 0), (ZERO_CENTER_X + 5, h + 10), (255, 255, 255), 1)
            if cx_calibrated != -1:
                cv2.circle(debug_img, (cx_calibrated + 5, (h//2) + 5), 3, (0, 255, 255), -1)

            cv2.imwrite(f"{SAVE_DIR}/sample_f{frame_idx}_{state}.png", debug_img)
            print(f"Saving Sample -> Frame: {frame_idx} | State: {state}")

        analyzed_count += 1

    frame_idx += 1
    # 진행 상황 출력
    if frame_idx % 300 == 0:
        print(f"진행 중... {frame_idx}/{total_frames} 프레임 처리 완료")

cap.release()
print(f"\n총 {analyzed_count}개 프레임 분석 완료.")