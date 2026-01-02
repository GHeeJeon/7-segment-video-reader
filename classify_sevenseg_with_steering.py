# 7세그 숫자 분류(디스큐 포함, 2자리 고정 레이아웃)
# 개선점:
#  - [A] 상단 가로 세그먼트('A') 임계 강화(특히 일의 자리)
#  - [B] 배경이 충분히 어두우면 HEX 기준으로 강제 #000000 처리(블랙 클리핑)
#  - [C] 시각화 저장을 선택적으로 실행 가능
import os, glob, cv2, numpy as np, argparse
from collections import namedtuple

# ===== 입출력 =====
IN_DIR   = r"./frames30_pts"  # 입력 폴더
OUT_CSV  = r"_cls_result.csv"
VIS_DIR  = r"_cls_overlay"


# ============================================================
# ===== Steering(조향) ROI 캡쳐 + L/0/R 인식 (추가 기능) =====
#  - 기존 classify_sevenseg.py의 입출력/전처리/옵션(overlay) 흐름은 그대로 유지
#  - 각 프레임(이미지)에서 ROI를 크롭해 밝은 인디케이터 점의 중심을 구하고,
#    보정값을 적용해 L / 0 / R 상태를 판정합니다.
#  - 디버그 샘플 저장은 STEER_DEBUG_ENABLE로 제어합니다.
# ============================================================

# ROI (y1, y2, x1, x2)  ※ 사용자가 제공한 값 그대로
ROI_STEER = (942, 965, 1015, 1075)

# 보정값(사용자 코드 그대로)
ZERO_CENTER_X = 29   # 기존 30에서 왼쪽으로 1px 이동(-1)
X_POINT_BIAS  = 3    # 감지점(cx)을 오른쪽으로 3px 보정(+3)
TOLERANCE     = 6    # '0' 판정 허용 범위

# 이진화 기준(사용자 코드 그대로)
STEER_THRESH_VALUE = 200

# 디버그 샘플 저장(선택)
STEER_DEBUG_ENABLE = True
STEER_DEBUG_DIR    = "./steering_analysis_30fps"
STEER_DEBUG_EVERY_N = 30   # 분석된 이미지 N장마다 1장 저장(=30fps 기준 1초마다)

def _clamp_roi(roi, H, W):
    y1, y2, x1, x2 = roi
    y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
    x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
    return y1, y2, x1, x2

def _save_steer_debug(crop, state, cx_calibrated, tag: str):
    os.makedirs(STEER_DEBUG_DIR, exist_ok=True)

    h, w = crop.shape[:2]
    if state == "0":
        border_color = (0, 255, 0)
    elif state == "R":
        border_color = (0, 0, 255)
    else:
        border_color = (255, 0, 0)

    debug_img = cv2.copyMakeBorder(crop, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=border_color)

    # 기준선(테두리 +5 보정)
    cv2.line(
        debug_img,
        (ZERO_CENTER_X + 5, 0),
        (ZERO_CENTER_X + 5, h + 10),
        (255, 255, 255),
        1
    )

    # 감지점 표시
    if cx_calibrated is not None and cx_calibrated >= 0:
        cv2.circle(debug_img, (cx_calibrated + 5, (h // 2) + 5), 3, (0, 255, 255), -1)

    out_path = os.path.join(STEER_DEBUG_DIR, f"sample_{tag}_{state}.png")
    cv2.imwrite(out_path, debug_img)

def analyze_steering(bgr, save_debug=False, debug_tag: str = "0000"):
    """
    Returns:
      steer_state: "L"|"0"|"R"|"N/A"
      steer_cx_raw: int (ROI 기준) or -1
      steer_cx_cal: int (보정 적용) or -1
    """
    H, W = bgr.shape[:2]
    y1, y2, x1, x2 = _clamp_roi(ROI_STEER, H, W)
    if y2 <= y1 or x2 <= x1:
        return "N/A", -1, -1

    crop = bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return "N/A", -1, -1

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, STEER_THRESH_VALUE, 255, cv2.THRESH_BINARY)

    M = cv2.moments(thresh)
    if M["m00"] <= 0:
        state = "N/A"
        cx_raw = -1
        cx_cal = -1
        if save_debug:
            _save_steer_debug(crop, state, None, debug_tag)
        return state, cx_raw, cx_cal

    cx_raw = int(M["m10"] / M["m00"])
    cx_cal = cx_raw + X_POINT_BIAS

    if cx_cal < (ZERO_CENTER_X - TOLERANCE):
        state = "L"
    elif cx_cal > (ZERO_CENTER_X + TOLERANCE):
        state = "R"
    else:
        state = "0"

    if save_debug:
        _save_steer_debug(crop, state, cx_cal, debug_tag)

    return state, cx_raw, cx_cal
# ===== 전처리 =====
INVERT           = False
CLAHE_CLIP       = 2.0

# --- 배경 감산 + Otsu ---
BG_MED_K         = 31
USE_CLAHE        = True
USE_OTSU         = True

# --- [B방법] 어두운 배경 블랙 클리핑(HEX 기준) ---
BG_BLACK_CLIP_USE = True
BG_BLACK_CLIP_HEX = "#a0a0a0"   # 이 값(또는 더 밝게: "#303030" 등) 이하의 픽셀은 0으로 클리핑

# 형태학: 끊김 보정 + 미세 잡음 제거
K_CLOSE          = (3,3)
CLOSE_ITERS      = 1
OPEN_ITERS       = 1

# ===== 핵심 박스(고정 박스 사용) =====
USE_FIXED_BOX    = True
BOX_PAD_W_FR     = 0.04     # 좌우 패딩(비율)
BOX_PAD_H_FR     = 0.08     # 상하 패딩(비율)
USE_PIXEL_BOX    = False
BOX_W_PX         = 0
BOX_H_PX         = 0

# ===== 세그먼트 임계 =====
SEG_T_BASE       = 0.50
SEG_T_VERT_DELTA = -0.06
SEG_T_G_DELTA    = +0.18
SEG_T_E_EXTRA    = -0.04
SEG_T_C_EXTRA    = -0.04

# --- [A방법] 'A' 세그먼트(가로 상단) 엄격화 옵션 ---
SEG_T_A_EXTRA          = 0.12   # 모든 자리 공통 A 가산 임계
SEG_T_A_ONES_EXTRA     = 0.10   # 일의 자리 A 추가 가산 임계(더 엄격)
A_ON_FR_USE_MEDIAN     = True   # A의 on-비율 계산에 max 대신 median 사용(가느다란 노이즈 무시)

# ===== 디스큐 설정 =====
DESKEW_USE_FIXED_ROTATION = True
DESKEW_FIXED_ROT_DEG      = 4.0
DESKEW_MIN_AREA           = 50
DESKEW_MAX_DEG            = 6.0
DESKEW_DAMPING            = 0.35
DESKEW_NEAR_VERT_DEG      = 12
DESKEW_HOUGH_CANNY_1      = 50
DESKEW_HOUGH_CANNY_2      = 150
DESKEW_HOUGH_THRESH       = 20
DESKEW_MIN_LINE_LEN_FR    = 0.35
DESKEW_MAX_LINE_GAP       = 3
DESKEW_ROT_BIAS_DEG       = 0

# ===== 멀티-디지트 분할 파라미터(기존) =====
MAX_DIGITS         = 3
MIN_DIGIT_W_FR     = 0.18
MAX_DIGIT_W_FR     = 0.50
MIN_GAP_FR         = 0.02
PROJ_SMOOTH_K      = 7
VALLEY_T_FR        = 0.12
ACTIVE_COL_T_FR    = 0.05
RIGHT_ALIGN        = True

# ===== 2자리 고정 레이아웃: 파란 박스의 오른쪽 2/3만 사용 =====
DIGIT_PRESENCE_ON_FR = 0.02   # tens 존재 판정 임계
TWO_DIGIT_USE_FIXED_RIGHT_2_3 = True

# ===== 세그먼트 위치(핵심 박스 내부 상대좌표) =====
Seg = namedtuple("Seg", "x0 y0 x1 y1")
SEGS = {
    "A": Seg(0.26, 0.17, 0.46, 0.23),
    "B": Seg(0.74, 0.27, 0.93, 0.44),
    "C": Seg(0.74, 0.63, 0.93, 0.77),
    "D": Seg(0.35, 0.85, 0.65, 0.91),
    "E": Seg(0.12, 0.63, 0.28, 0.77),
    "F": Seg(0.12, 0.27, 0.28, 0.44),
    "G": Seg(0.35, 0.52, 0.65, 0.56),
}
SEG_ORDER = ["A","B","C","D","E","F","G"]

# 표준 패턴(A,B,C,D,E,F,G)
DIGIT_PATTERNS = {
    0: (1,1,1,1,1,1,0),
    1: (0,1,1,0,0,0,0),
    2: (1,1,0,1,1,0,1),
    3: (1,1,1,1,0,0,1),
    4: (0,1,1,0,0,1,1),
    5: (1,0,1,1,0,1,1),
    6: (1,0,1,1,1,1,1),
    7: (1,1,1,0,0,0,0),
    8: (1,1,1,1,1,1,1),
    9: (1,1,1,1,0,1,1),
}

# ---------- 유틸 ----------
def _hex_to_gray(hexstr: str) -> int:
    """'#RRGGBB' → 0..255 회색(루마) 임계로 변환."""
    hs = hexstr.strip().lstrip("#")
    if len(hs) != 6:
        return 0
    r = int(hs[0:2], 16); g = int(hs[2:4], 16); b = int(hs[4:6], 16)
    y = 0.299*r + 0.587*g + 0.114*b
    return int(round(np.clip(y, 0, 255)))

def _weighted_median(angles, weights):
    idx = np.argsort(angles)
    a = np.array(angles)[idx]; w = np.array(weights)[idx]
    cw = np.cumsum(w) / np.sum(w)
    k = np.searchsorted(cw, 0.5)
    return float(a[min(k, len(a)-1)])

# ---------- 전처리 ----------
def preprocess(bgr):
    """
    회색조 → (옵션)CLAHE → [B]블랙클리핑(HEX) → 배경추정(미디안) → 감산 → Otsu → 모폴로지.
    """
    g  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # (옵션) CLAHE
    if USE_CLAHE:
        g = cv2.createCLAHE(CLAHE_CLIP, (8,8)).apply(g)

    # [B] 어두운 배경은 강제로 0(#000000)으로 클리핑
    if BG_BLACK_CLIP_USE:
        thr = _hex_to_gray(BG_BLACK_CLIP_HEX)
        # thr 이하 → 0
        g = g.copy()
        g[g <= thr] = 0

    # 배경 추정 및 감산
    bg  = cv2.medianBlur(g, BG_MED_K)
    sub = cv2.subtract(g, bg)

    # 빈 프레임(노이즈만 있는 경우) 빠른 탈출을 원하면 여기서 분기 가능
    if USE_OTSU:
        _, bw = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        bw = cv2.adaptiveThreshold(sub, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 2)
    if INVERT:
        bw = 255 - bw

    k = cv2.getStructuringElement(cv2.MORPH_RECT, K_CLOSE)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITERS)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  k, iterations=OPEN_ITERS)
    return bw

# ---------- 디스큐 ----------
def deskew(bgr, bw):
    H, W = bw.shape
    if DESKEW_USE_FIXED_ROTATION:
        rot = float(DESKEW_FIXED_ROT_DEG)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, 1.0)
        bgr_rot = cv2.warpAffine(bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        bw_rot  = cv2.warpAffine(bw,  M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return bgr_rot, bw_rot

    # (동적 모드 생략: 필요시 기존 코드 이식)
    return bgr, bw

# ---------- 핵심 박스 ----------
def _fixed_bbox_fraction(bw):
    H, W = bw.shape
    x0 = int(W * BOX_PAD_W_FR); x1 = W - 1 - int(W * BOX_PAD_W_FR)
    y0 = int(H * BOX_PAD_H_FR); y1 = H - 1 - int(H * BOX_PAD_H_FR)
    x0 = max(0, min(x0, W-2)); x1 = max(x0+1, min(x1, W-1))
    y0 = max(0, min(y0, H-2)); y1 = max(y0+1, min(y1, H-1))
    return x0, y0, x1, y1

def _fixed_bbox_pixel(bw):
    H, W = bw.shape
    w = BOX_W_PX if BOX_W_PX > 0 else W
    h = BOX_H_PX if BOX_H_PX > 0 else H
    w = min(w, W); h = min(h, H)
    x0 = (W - w)//2; x1 = x0 + w - 1
    y0 = (H - h)//2; y1 = y0 + h - 1
    return x0, y0, x1, y1

def core_box(bw):
    if USE_FIXED_BOX:
        if USE_PIXEL_BOX: return _fixed_bbox_pixel(bw)
        return _fixed_bbox_fraction(bw)
    # 동적 모드가 필요하면 기존 구현을 붙이세요.
    H, W = bw.shape
    return 0, 0, W-1, H-1

# ---------- 2자리 고정 보라색 박스(우측 2/3) ----------
def fixed_two_digit_boxes(core):
    """
    파란 박스(core)를 3등분한 뒤, 왼쪽(백의 자리)을 버리고 오른쪽 2칸만 사용.
    tens: 가운데 칸(정중앙)
    ones: 오른쪽 칸(우측 벽 밀착, 항상 포함)
    """
    cx0, cy0, cx1, cy1 = core
    Wc = cx1 - cx0 + 1; Hc = cy1 - cy0 + 1
    fw = max(1, Wc // 3)  # 각 칸 폭
    y0, y1 = cy0, cy1

    # ones: 오른쪽 칸
    ones_x1 = cx1
    ones_x0 = max(cx0, ones_x1 - fw + 1)

    # tens: 가운데 칸(겹침 없이 배치)
    tens_x0 = cx0 + (Wc - fw) // 2
    tens_x1 = tens_x0 + fw - 1

    # tens 존재 판정(희미하면 제외)
    boxes = []
    boxes.append(("tens", (tens_x0, y0, tens_x1, y1)))
    boxes.append(("ones", (ones_x0, y0, ones_x1, y1)))
    return boxes

# ---------- 세그먼트 on/off ----------
def segment_states(bw, box, place: str = "ones"):
    """
    place: 'ones' | 'tens'  → A 세그먼트 임계 조정에 사용
    """
    cx0, cy0, cx1, cy1 = box
    Wc = max(1, cx1 - cx0 + 1)
    Hc = max(1, cy1 - cy0 + 1)

    # 기본 임계
    t_base = SEG_T_BASE
    t_vert = np.clip(t_base + SEG_T_VERT_DELTA, 0, 1)
    t_g    = np.clip(t_base + SEG_T_G_DELTA,   0, 1)
    t_e    = np.clip(t_vert + SEG_T_E_EXTRA,   0, 1)
    t_c    = np.clip(t_vert + SEG_T_C_EXTRA,   0, 1)

    # A 세그먼트 가산(자리별)
    a_extra = SEG_T_A_EXTRA + (SEG_T_A_ONES_EXTRA if place == "ones" else 0.0)
    T = {"A":np.clip(t_base + a_extra, 0, 1), "B":t_vert, "C":t_c, "D":t_base, "E":t_e, "F":t_vert, "G":t_g}

    states = []
    for key in SEG_ORDER:
        s = SEGS[key]

        if key == "A" and (place == "tens" or place == "ones"):
            s = Seg(s.x0 + 0.03, s.y0, s.x1 + 0.15, s.y1)

        x0 = cx0 + int(s.x0 * Wc); x1 = cx0 + int(s.x1 * Wc)
        y0 = cy0 + int(s.y0 * Hc); y1 = cy0 + int(s.y1 * Hc)
        x0 = max(0, min(bw.shape[1]-1, x0)); x1 = max(0, min(bw.shape[1]-1, x1))
        y0 = max(0, min(bw.shape[0]-1, y0)); y1 = max(0, min(bw.shape[0]-1, y1))
        if x1 <= x0 or y1 <= y0:
            states.append(0); 
            continue

        if key in ("A","D","G"):  # 가로 세그: 세로로 3분할
            thirds = np.linspace(y0, y1, 4, dtype=int)
            vals = []
            for k in range(3):
                yy0, yy1 = thirds[k], thirds[k+1]
                if yy1 <= yy0: continue
                roi = bw[yy0:yy1, x0:x1]
                vals.append((roi > 0).mean())
            # A만 median 사용 가능
            on_frac = (np.median(vals) if (A_ON_FR_USE_MEDIAN and key == "A") else (max(vals) if vals else 0.0)) if vals else 0.0
        else:                      # 세로 세그: 가로로 3분할
            thirds = np.linspace(x0, x1, 4, dtype=int)
            vals = []
            for k in range(3):
                xx0, xx1 = thirds[k], thirds[k+1]
                if xx1 <= xx0: continue
                roi = bw[y0:y1, xx0:xx1]
                vals.append((roi > 0).mean())
            on_frac = max(vals) if vals else 0.0

        states.append(1 if on_frac >= T[key] else 0)

    return tuple(states)

# ---------- 매칭 ----------
def match_digit(states):
    if sum(states) <= 1:
        return -1, 0.0, 7

    best_d, best_dist = None, 99
    for d, pat in DIGIT_PATTERNS.items():
        dist = sum(ps ^ qs for ps, qs in zip(states, pat))
        if dist < best_dist:
            best_dist = dist
            best_d = d

    conf = 1.0 - (best_dist / 7.0)
    if best_dist == 0:
        return best_d, 1.0, 0
    else:
        return -1, conf, best_dist

# ---------- 시각화 ----------
def draw_overlay_multi(bgr, bw, core, pair_boxes, per_digit):
    vis = bgr.copy()
    cx0, cy0, cx1, cy1 = core
    # 핵심 박스(파란색)
    cv2.rectangle(vis, (cx0,cy0), (cx1,cy1), (255,0,0), 1)
    for (place, dbox), (pred, conf, dist, states) in zip(pair_boxes, per_digit):
        dx0, dy0, dx1, dy1 = dbox
        # 자릿수 박스(보라색)
        cv2.rectangle(vis, (dx0,dy0), (dx1,dy1), (255,0,255), 1)
        # 세그먼트
        Wc = dx1 - dx0 + 1; Hc = dy1 - dy0 + 1
        for j, key in enumerate(SEG_ORDER):
            s = SEGS[key]
            x0 = dx0 + int(s.x0 * Wc); x1 = dx0 + int(s.x1 * Wc)
            y0 = dy0 + int(s.y0 * Hc); y1 = dy0 + int(s.y1 * Hc)
            color = (0,255,0) if states[j] == 1 else (0,0,255)
            cv2.rectangle(vis, (x0,y0), (x1,y1), color, 1)
        label = f"{place}:{pred}({conf:.2f})"
        cv2.putText(vis, label, (dx0, max(10, dy0-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(vis, label, (dx0, max(10, dy0-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return vis

# ---------- 메인 ----------
def main(overlay=None):
    # 명령줄 인자 파싱 (직접 실행될 때만)
    if overlay is None:
        parser = argparse.ArgumentParser(description='7-segment 숫자 분류')
        parser.add_argument('-o', '--overlay', action='store_true',
                            help='인식 결과 오버레이 이미지 저장')
        args = parser.parse_args()
        overlay = args.overlay

    # overlay 활성화 시에만 폴더 생성
    if overlay:
        os.makedirs(VIS_DIR, exist_ok=True)

    patterns = ("*.[Pp][Nn][Gg]", "*.[Jj][Pp][Gg]", "*.[Jj][Pp][Ee][Gg]")
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(IN_DIR, pat)))
    paths = sorted(dict.fromkeys(paths))  # 안전하게 중복 제거

    rows = ["filename,num_digits,pred_number,preds,confs,dists,states_per_digit,steer_state,steer_cx_raw,steer_cx_calibrated\n"]
    ok = 0

    for i, p in enumerate(paths):
        bgr0 = cv2.imread(p)
        if bgr0 is None:
            continue
        # [Steering] ROI 캡쳐 + L/0/R 판정 (추가)
        # - ROI 좌표는 원본 프레임(bgr0) 기준이므로 디스큐/전처리 이전에 수행
        steer_save = (STEER_DEBUG_ENABLE and (ok % STEER_DEBUG_EVERY_N == 0))
        steer_tag = f"{i:04d}"
        steer_state, steer_cx_raw, steer_cx_cal = analyze_steering(bgr0, save_debug=steer_save, debug_tag=steer_tag)


        # 0) 전처리
        bw0 = preprocess(bgr0)

        # 1) 디스큐
        bgr, bw = deskew(bgr0, bw0)

        # 2) 핵심 박스
        core = core_box(bw)

        # 3) 두 자리 고정 박스(우측 2/3)
        pair_boxes = fixed_two_digit_boxes(core)

        # 4) 자리별 분류
        per_digit = []
        preds, confs, dists, states_dump = [], [], [], []
        digit_states = {}
        
        for place, dbox in pair_boxes:
            states = segment_states(bw, dbox, place=place)
            pred, conf, dist = match_digit(states)

            digit_states[place] = states  # ← 세그먼트 원형 저장

            per_digit.append((pred, conf, dist, states))
            preds.append(str(pred))
            confs.append(f"{conf:.3f}")
            dists.append(str(dist))
            states_dump.append(f"{place}:{''.join(map(str,states))}")

        # 5) pred_number 조립 (특수 조건 처리 포함)
        # 조건: tens 세그먼트가 모두 0이고, ones 는 유효하면 → ones만 사용
        tens_states = digit_states.get("tens", (0,)*7)
        ones_pred = int(preds[1]) if preds[1] != "-1" else -1
        tens_pred = int(preds[0]) if preds[0] != "-1" else -1

        if all(s == 0 for s in tens_states) and ones_pred >= 0:
            pred_number = ones_pred
        elif ones_pred == -1 or tens_pred == -1:
            pred_number = -1
        else:
            pred_number = tens_pred * 10 + ones_pred

        # 6) 시각화 저장 (옵션이 활성화된 경우에만)
        if overlay:
            vis = draw_overlay_multi(bgr, bw, core, pair_boxes, per_digit)
            cv2.imwrite(os.path.join(VIS_DIR, f"{i:04d}_{pred_number}.png"), vis)

        # 7) CSV 저장
        preds_str = '"' + " ".join(preds) + '"'
        rows.append(
            f"{os.path.basename(p)},{len(preds)},{pred_number},"
            f"{preds_str},{' '.join(confs)},{' '.join(dists)},"
            f"{'|'.join(states_dump)},{steer_state},{steer_cx_raw},{steer_cx_cal}\n"
        )
        ok += 1

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.writelines(rows)
    
    if overlay:
        print(f"완료: {ok}장 분류 → {OUT_CSV} / 오버레이: {VIS_DIR}")
    else:
        print(f"완료: {ok}장 분류 → {OUT_CSV}")

if __name__ == "__main__":
    main()