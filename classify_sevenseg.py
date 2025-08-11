# classify_sevenseg.py — 7세그 숫자 분류(디스큐 포함)

import os, glob, cv2, numpy as np
from collections import namedtuple
from tqdm import tqdm

# ===== 입출력 =====
IN_DIR   = r"../frames30_pts" # PowerShell 에서 30 fs 단위로 크롭해 추출한 이미지
OUT_CSV  = r"cls_result.csv"
VIS_DIR  = r"_cls_overlay"
os.makedirs(VIS_DIR, exist_ok=True)

# ===== 전처리 =====
INVERT           = False
CLAHE_CLIP       = 2.0

# --- 변경: 적응형 이진화 대신 배경 감산 + Otsu 사용 ---
BG_MED_K         = 31     # 배경 추정용 미디안 블러 커널(홀수)
USE_CLAHE        = True   # 배경 감산 전 CLAHE 적용 여부
USE_OTSU         = True   # 임계화는 Otsu 사용(권장)

# 형태학: 끊김 보정 + 미세 잡음 제거
K_CLOSE          = (3,3)
CLOSE_ITERS      = 1
OPEN_ITERS       = 1

# ===== 핵심 박스(고정 박스 사용) =====
USE_FIXED_BOX    = True     # True면 고정 박스, False면 동적(기존 방식)
# --- 비율 기반 고정 박스(권장): 이미지 기준 패딩 비율 ---
BOX_PAD_W_FR     = 0.04     # 좌우 패딩 비율(이미지 가로 대비)
# BOX_PAD_LEFT_FR   = 0.5         # ← 왼쪽을 많이 잘라내 일의 자리만 보이게
# BOX_PAD_RIGHT_FR  = 0.04        # → 오른쪽은 약간만 여유

BOX_PAD_H_FR     = 0.08     # 상하 패딩 비율(이미지 세로 대비)
# --- 픽셀 기반 고정 박스(둘 중 하나만 사용) ---
USE_PIXEL_BOX    = False    # True면 아래 픽셀 크기 사용
BOX_W_PX         = 0        # 고정 박스 폭(px), 0이면 비활성
BOX_H_PX         = 0        # 고정 박스 높이(px), 0이면 비활성

# ===== 세그먼트 임계 =====
SEG_T_BASE       = 0.50
SEG_T_VERT_DELTA = -0.06
SEG_T_G_DELTA    = +0.18
SEG_T_E_EXTRA    = -0.04
SEG_T_C_EXTRA    = -0.04

# ===== 디스큐 설정 (기울기 보정) =====
DESKEW_USE_FIXED_ROTATION = True   # 모든 이미지에 고정 각도 사용 여부
DESKEW_FIXED_ROT_DEG      = 4.0    # 고정 회전 각도(+: 반시계, -: 시계)

DESKEW_MIN_AREA           = 50     # 동적 디스큐 폴백에서 최소 컨투어 면적
DESKEW_MAX_DEG            = 6.0    # 동적 디스큐 회전량 제한(절대값)
DESKEW_DAMPING            = 0.35   # 동적 디스큐 회전량 댐핑 비율
DESKEW_NEAR_VERT_DEG      = 12     # 수직(90도) 근처로 간주할 허용 편차

DESKEW_HOUGH_CANNY_1      = 50     # Canny 1
DESKEW_HOUGH_CANNY_2      = 150    # Canny 2
DESKEW_HOUGH_THRESH       = 20     # HoughLinesP threshold
DESKEW_MIN_LINE_LEN_FR    = 0.35   # 이미지 높이 대비 최소 선 길이(비율)
DESKEW_MAX_LINE_GAP       = 3      # 선 세그먼트 연결 허용 간격(px)

DESKEW_ROT_BIAS_DEG       = 0      # 추가 바이어스(고정 회전 모드에서는 0 권장)

# ===== 멀티-디지트 분할 파라미터 =====
MAX_DIGITS         = 3       # 최대 자릿수
MIN_DIGIT_W_FR     = 0.18    # 한 자리 최소 폭(핵심박스 폭 대비 비율)
MAX_DIGIT_W_FR     = 0.50    # 한 자리 최대 폭(핵심박스 폭 대비 비율)
MIN_GAP_FR         = 0.02    # 자리 사이 최소 간격(핵심박스 폭 대비)
PROJ_SMOOTH_K      = 7       # 수평 투영 스무딩(홀수)
VALLEY_T_FR        = 0.12    # valley 판단 임계(투영 합의 최대치 대비 비율)
ACTIVE_COL_T_FR    = 0.05    # ‘유효 열’ 판단 임계(흰 픽셀 비율)
RIGHT_ALIGN        = True    # 오른쪽 정렬 디스플레이 가정(권장)

# ===== 디지트 박스(자릿수 박스) 고정 크기 옵션 =====
DIGIT_BOX_USE_FIXED  = True   # True면 분할 결과 폭을 무시하고 고정 크기로 덮어씀
DIGIT_BOX_W_FR       = 0.22    # 핵심박스 너비 대비 자릿수 박스 '고정 폭' 비율 (픽셀 지정이 0이면 사용)
DIGIT_BOX_W_PX       = 0       # 자릿수 박스 '고정 폭' 픽셀 (0이면 비활성 → W_FR 사용)
DIGIT_BOX_H_FR       = 1.00    # 핵심박스 높이 대비 고정 높이 비율 (보통 전체=1.0)
DIGIT_BOX_H_PX       = 0       # 자릿수 박스 '고정 높이' 픽셀 (0이면 비활성 → H_FR 사용)

# ===== 고정 3분할용 존재 판정 임계 =====
DIGIT_PRESENCE_ON_FR = 0.02   # 자릿수 박스 내부 흰 픽셀 비율이 이 이상이면 '숫자 존재'로 간주

# ===== 빈 프레임(블랭크) =====
BG_NOISE_P99_THR     = 12   # sub의 P99가 이보다 낮으면 빈 프레임(8-bit 기준)
POST_WHITE_FRAC_MIN  = 0.003  # 이진화 후 흰 픽셀 비율이 이보다 작으면 빈 프레임
OTSU_MIN_THR         = 16   # Otsu가 너무 낮게 나오면 이 값으로 바닥 고정


# ===== 세그먼트 위치(핵심 박스 내부 상대좌표) =====
Seg = namedtuple("Seg", "x0 y0 x1 y1")
SEGS = {
    # A: 폭 절반, 왼쪽 이동, 위치 약간 아래로
    "A": Seg(0.31, 0.19, 0.48, 0.23),

    # 세로 세그먼트 (B, C)
    "B": Seg(0.74, 0.27, 0.90, 0.44),  # ↓ y0 +0.02, y1 +0.02 → 아래로 이동
    "C": Seg(0.74, 0.63, 0.90, 0.77),

    # 세로 세그먼트 (E, F)
    "E": Seg(0.12, 0.63, 0.28, 0.77),
    "F": Seg(0.12, 0.27, 0.28, 0.44),  # ↓ y0 +0.02, y1 +0.02 → 아래로 이동

    # D: 기준
    "D": Seg(0.35, 0.86, 0.65, 0.90),

    # G: D와 동일한 폭·높이
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

def _weighted_median(angles, weights):
    idx = np.argsort(angles)
    a = np.array(angles)[idx]; w = np.array(weights)[idx]
    cw = np.cumsum(w) / np.sum(w)
    k = np.searchsorted(cw, 0.5)
    return float(a[min(k, len(a)-1)])

# ---------- 전처리 ----------
def preprocess(bgr):
    g  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if USE_CLAHE:
        g = cv2.createCLAHE(CLAHE_CLIP, (8,8)).apply(g)

    # 배경 추정 & 감산
    bg  = cv2.medianBlur(g, BG_MED_K)
    sub = cv2.absdiff(g, bg)          # (= |g - bg|, 노이즈 양수화)

    # 1) 빈 프레임 사전 차단 (노이즈 바닥)
    if np.percentile(sub, 99) < BG_NOISE_P99_THR:
        return np.zeros_like(g)       # 전부 꺼짐 → match_digit에서 0으로 처리됨

    # 2) 임계화(최소 임계 보장)
    if USE_OTSU:
        thr, _ = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thr = max(thr, OTSU_MIN_THR)                  # 너무 낮은 Otsu 보정
        _, bw = cv2.threshold(sub, thr, 255, cv2.THRESH_BINARY)
    else:
        bw = cv2.adaptiveThreshold(sub, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 2)

    if INVERT: bw = 255 - bw

    # 3) 형태학으로 점노이즈 정리(필요시 open 강하게)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, K_CLOSE)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITERS)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  k, iterations=max(OPEN_ITERS, 1))

    # 4) 이진화 후 흰 픽셀 총량이 너무 작으면 빈 프레임 처리
    if (bw > 0).mean() < POST_WHITE_FRAC_MIN:
        return np.zeros_like(bw)

    return bw

# ---------- 디스큐 ----------
def deskew(bgr, bw):
    H, W = bw.shape

    # --- 고정 회전 모드 ---
    if DESKEW_USE_FIXED_ROTATION:
        rot = float(DESKEW_FIXED_ROT_DEG)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, 1.0)  # +: 반시계
        bgr_rot = cv2.warpAffine(bgr, M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        bw_rot = cv2.warpAffine(bw, M, (W, H), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return bgr_rot, bw_rot

    # --- 동적 디스큐 모드 ---
    edges = cv2.Canny(bw, DESKEW_HOUGH_CANNY_1, DESKEW_HOUGH_CANNY_2)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=DESKEW_HOUGH_THRESH,
        minLineLength=int(H * DESKEW_MIN_LINE_LEN_FR),
        maxLineGap=DESKEW_MAX_LINE_GAP
    )

    angle_rel_list, weight_list = [], []
    if lines is not None and len(lines) >= 2:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            dx, dy = (x2 - x1), (y2 - y1)
            if dx == 0 and dy == 0:
                continue
            ang = np.degrees(np.arctan2(dy, dx))  # 0=수평, 90=수직
            rel = ang - 90.0
            if rel > 90:  rel -= 180
            if rel < -90: rel += 180
            if abs(rel) <= DESKEW_NEAR_VERT_DEG:
                angle_rel_list.append(rel)
                weight_list.append(np.hypot(dx, dy))

    if len(angle_rel_list) >= 2:
        rel = _weighted_median(angle_rel_list, weight_list)
        rot = rel * DESKEW_DAMPING
    else:
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return bgr, bw
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < DESKEW_MIN_AREA:
            return bgr, bw
        rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
        w, h = rect[1]
        angle = rect[2]
        if w < h:
            rot = (-angle) * DESKEW_DAMPING
        else:
            rot = (-(angle + 90.0)) * DESKEW_DAMPING

    rot = float(np.clip(rot, -DESKEW_MAX_DEG, DESKEW_MAX_DEG))
    rot += DESKEW_ROT_BIAS_DEG

    M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, 1.0)  # +rot = 반시계
    bgr_rot = cv2.warpAffine(bgr, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    bw_rot = cv2.warpAffine(bw, M, (W, H), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return bgr_rot, bw_rot


# ---------- 핵심 박스(틀) ----------
def _central_bbox_dynamic(bw):
    """(구) 흰 픽셀 기반 동적 바운딩 + 확장 — 호환용."""
    H, W = bw.shape
    ys, xs = np.where(bw > 0)
    if len(xs) == 0:
        return 0,0,W-1,H-1
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad_w  = int((x1 - x0 + 1) * 0.08)
    pad_h  = int((y1 - y0 + 1) * 0.08)
    x0 = max(0, x0 - pad_w);  x1 = min(W-1, x1 + pad_w)
    y0 = max(0, y0 - pad_h);  y1 = min(H-1, y1 + pad_h)
    return x0, y0, x1, y1

def _fixed_bbox_fraction(bw):
    """이미지 크기 기준 비율 패딩으로 고정 박스."""
    H, W = bw.shape
    x0 = int(W * BOX_PAD_W_FR)
    x1 = W - 1 - int(W * BOX_PAD_W_FR)
    y0 = int(H * BOX_PAD_H_FR)
    y1 = H - 1 - int(H * BOX_PAD_H_FR)
    # 경계 안전
    x0 = max(0, min(x0, W-2)); x1 = max(x0+1, min(x1, W-1))
    y0 = max(0, min(y0, H-2)); y1 = max(y0+1, min(y1, H-1))
    return x0, y0, x1, y1

def _fixed_bbox_pixel(bw):
    """절대 픽셀 크기의 고정 박스(중앙 정렬)."""
    H, W = bw.shape
    w = BOX_W_PX if BOX_W_PX > 0 else W
    h = BOX_H_PX if BOX_H_PX > 0 else H
    w = min(w, W); h = min(h, H)
    x0 = (W - w)//2; x1 = x0 + w - 1
    y0 = (H - h)//2; y1 = y0 + h - 1
    return x0, y0, x1, y1

def core_box(bw):
    """현재 설정에 맞는 핵심 박스(파란 네모)를 반환."""
    if USE_FIXED_BOX:
        if USE_PIXEL_BOX:
            return _fixed_bbox_pixel(bw)
        else:
            return _fixed_bbox_fraction(bw)
    else:
        return _central_bbox_dynamic(bw)

# ---------- 세그먼트 on/off ----------
def segment_states(bw, box):
    cx0, cy0, cx1, cy1 = box
    Wc = max(1, cx1 - cx0 + 1)
    Hc = max(1, cy1 - cy0 + 1)

    t_base = SEG_T_BASE
    t_vert = np.clip(t_base + SEG_T_VERT_DELTA, 0, 1)
    t_g    = np.clip(t_base + SEG_T_G_DELTA,   0, 1)
    t_e    = np.clip(t_vert + SEG_T_E_EXTRA,   0, 1)
    t_c    = np.clip(t_vert + SEG_T_C_EXTRA,   0, 1)
    T = {"A":t_base, "B":t_vert, "C":t_c, "D":t_base, "E":t_e, "F":t_vert, "G":t_g}

    states = []
    for key in SEG_ORDER:
        s = SEGS[key]
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
            on_frac = max(vals) if vals else 0.0
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

# ---------- 매칭 & 시각화 ----------
def match_digit(states):
    # 전부 꺼짐(= 모두 0) → 0으로 간주
    if sum(states) == 0:
        return 0, 1.0, 0  # pred=0, conf=1.0(확신), dist=0
    
    best_d, best_dist = None, 99
    for d, pat in DIGIT_PATTERNS.items():
        dist = sum(ps ^ qs for ps, qs in zip(states, pat))
        if dist < best_dist:
            best_dist = dist; best_d = d
    conf = 1.0 - (best_dist / 7.0)
    return best_d, conf, best_dist

def draw_overlay(bgr, bw, box, states, pred, conf):
    vis = bgr.copy()
    cx0, cy0, cx1, cy1 = box
    cv2.rectangle(vis, (cx0,cy0), (cx1,cy1), (255,0,0), 1)  # 파란 네모(고정 박스)
    Wc = cx1 - cx0 + 1; Hc = cy1 - cy0 + 1
    for i, key in enumerate(SEG_ORDER):
        s = SEGS[key]
        x0 = cx0 + int(s.x0 * Wc); x1 = cx0 + int(s.x1 * Wc)
        y0 = cy0 + int(s.y0 * Hc); y1 = cy0 + int(s.y1 * Hc)
        color = (0,255,0) if states[i] == 1 else (0,0,255)
        cv2.rectangle(vis, (x0,y0), (x1,y1), color, 1)
    label = f"{pred} ({conf:.2f})"
    cv2.putText(vis, label, (cx0, max(10, cy0-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(vis, label, (cx0, max(10, cy0-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return vis

# ---------- 멀티-디지트 분할/판독 유틸 추가 ----------
def _smooth1d(arr, k):
    k = max(1, int(k))
    if k % 2 == 0: k += 1
    ker = np.ones(k, dtype=float) / k
    return np.convolve(arr, ker, mode="same")

def _active_band(bw, box):
    """핵심 박스 내부에서 '실제 표시영역'의 좌우 활성 열 구간을 찾음."""
    x0, y0, x1, y1 = box
    sub = bw[y0:y1+1, x0:x1+1]
    H, W = sub.shape
    col_frac = (sub > 0).sum(axis=0) / max(1, H)  # 각 열의 흰픽셀 비율
    act = (col_frac >= ACTIVE_COL_T_FR).astype(np.uint8)

    # 오른쪽 정렬 가정: 오른쪽 끝에서부터 유효열을 찾고, 왼쪽으로 확장
    if RIGHT_ALIGN:
        if act[-1] == 0:
            # 오른쪽 끝이 0이면 오른쪽에서 처음 1이 나오는 지점 찾기
            idxs = np.where(act[::-1] == 1)[0]
            if len(idxs) == 0:
                return x0, x1  # 활성 없음: 원래 박스 반환
            r_end = W - 1 - idxs[0]
        else:
            r_end = W - 1
        # r_end에서 왼쪽으로 0이 충분히 이어지는 지점 이전까지 활성로 본다
        # 간단히: 활성 열이 시작되는 최좌측
        lefts = np.where(act[:r_end+1] == 1)[0]
        if len(lefts) == 0:
            return x0, x1
        l_beg = lefts[0]
        return x0 + l_beg, x0 + r_end
    else:
        # 비정렬: 전체 범위에서 첫 1과 마지막 1
        ones = np.where(act == 1)[0]
        if len(ones) == 0:
            return x0, x1
        return x0 + int(ones[0]), x0 + int(ones[-1])

def _split_digits_by_valleys(bw, box):
    """
    수직 투영의 valley로 자릿수 분할.
    반환: [(dx0,dy0,dx1,dy1), ...]  (이미지 전체 좌표)
    """
    cx0, cy0, cx1, cy1 = box
    # 1) 활성 밴드 좁히기
    ax0, ax1 = _active_band(bw, box)
    Wc = max(1, ax1 - ax0 + 1)
    Hc = max(1, cy1 - cy0 + 1)

    # 2) 수직 투영 (활성 밴드)
    sub = bw[cy0:cy1+1, ax0:ax1+1]
    proj = (sub > 0).sum(axis=0).astype(float) / Hc  # [0..1] 비율
    proj_s = _smooth1d(proj, PROJ_SMOOTH_K)

    # 3) valley 찾기: 최대값 대비 VALLEY_T_FR 이하인 구간
    vmax = proj_s.max() if proj_s.size else 1.0
    if vmax <= 0:  # 활성 없음
        return [(cx0, cy0, cx1, cy1)]

    valley_mask = (proj_s <= vmax * VALLEY_T_FR).astype(np.uint8)

    # 4) valley 경계를 이용해 '활성 덩어리(자릿수 후보)' 추출
    boxes = []
    in_run = False
    run_s = 0
    for i in range(Wc):
        active_col = 1 if proj_s[i] > vmax * VALLEY_T_FR else 0
        if active_col and not in_run:
            in_run = True
            run_s = i
        if (not active_col or i == Wc-1) and in_run:
            in_run = False
            run_e = i if not active_col else i  # inclusive
            # 폭 제한 체크
            w = run_e - run_s + 1
            if w / float(Wc) >= MIN_DIGIT_W_FR and w / float(Wc) <= MAX_DIGIT_W_FR:
                dx0 = ax0 + run_s
                dx1 = ax0 + run_e
                boxes.append((dx0, cy0, dx1, cy1))

    if not boxes:
        # 분할 실패 시 전체를 한 자리로 간주
        return [(ax0, cy0, ax1, cy1)]

    # 5) 자릿수 간 최소 간격 및 오른쪽 정렬 보정
    boxes = sorted(boxes, key=lambda b: b[0])  # left→right
    # gap 필터
    filtered = []
    last = None
    for b in boxes:
        if last is None:
            filtered.append(b)
            last = b
            continue
        gap = (b[0] - last[1]) / float(Wc)
        if gap < MIN_GAP_FR:
            # 너무 붙었으면 병합
            last = (last[0], last[1], max(last[2], b[2]), last[3])
            filtered[-1] = last
        else:
            filtered.append(b)
            last = b
    boxes = filtered

    # 최대 자릿수 제한 및 정렬(오른쪽 정렬이면 오른쪽에서 최대 MAX_DIGITS개)
    if RIGHT_ALIGN:
        boxes = boxes[-MAX_DIGITS:]
    else:
        boxes = boxes[:MAX_DIGITS]

    return boxes

def _classify_digit_box(bgr, bw, dbox):
    """단일 digit box에서 기존 segment_states→match_digit 재사용."""
    states = segment_states(bw, dbox)
    pred, conf, dist = match_digit(states)
    return pred, conf, dist, states

# ---------- 오버레이 함수(멀티 박스 표시) ----------
def draw_overlay_multi(bgr, bw, core, dboxes, per_digit):
    vis = bgr.copy()
    cx0, cy0, cx1, cy1 = core
    # 핵심 박스(파란색)
    cv2.rectangle(vis, (cx0,cy0), (cx1,cy1), (255,0,0), 1)

    # 각 digit 박스(연두), 세그먼트 사각형(초록/빨강)
    for i, (dbox, (pred, conf, dist, states)) in enumerate(zip(dboxes, per_digit)):
        dx0, dy0, dx1, dy1 = dbox
        cv2.rectangle(vis, (dx0,dy0), (dx1,dy1), (255,0,255), 1)

        # 세그먼트 박스도 그리기
        Wc = dx1 - dx0 + 1; Hc = dy1 - dy0 + 1
        for j, key in enumerate(SEG_ORDER):
            s = SEGS[key]
            x0 = dx0 + int(s.x0 * Wc); x1 = dx0 + int(s.x1 * Wc)
            y0 = dy0 + int(s.y0 * Hc); y1 = dy0 + int(s.y1 * Hc)
            color = (0,255,0) if states[j] == 1 else (0,0,255)
            cv2.rectangle(vis, (x0,y0), (x1,y1), color, 1)

        # 자리별 라벨
        label = f"{pred}({conf:.2f})"
        cv2.putText(vis, label, (dx0, max(10, dy0-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(vis, label, (dx0, max(10, dy0-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    return vis

# ---------- 고정 크기 적용 유틸 함수 ----------
def _apply_fixed_digit_size(dboxes, core):
    """분할된 자릿수 박스의 중심을 유지하되, 고정 폭/높이로 덮어쓴다."""
    cx0, cy0, cx1, cy1 = core
    Wc = cx1 - cx0 + 1
    Hc = cy1 - cy0 + 1

    # 폭은 핵심박스 3분의 1 (세자리 이미지일 경우)
    fw = Wc // 3

    # 높이는 전체
    fh = Hc

    # 핵심박스 내부 중앙 정렬 기반으로 클램프
    fixed = []
    for (x0, y0, x1, y1) in dboxes:
        cx = (x0 + x1) // 2
        cy = (cy0 + cy1) // 2  # 세로는 핵심박스 중앙 기준

        nx0 = max(cx0, min(cx - fw // 2, cx1 - fw + 1))
        nx1 = min(cx1, nx0 + fw - 1)

        ny0 = max(cy0, min(cy - fh // 2, cy1 - fh + 1))
        ny1 = min(cy1, ny0 + fh - 1)

        fixed.append((nx0, ny0, nx1, ny1))
    return fixed

# ---------- 고정 위치 자릿수 박스 생성 함수 ----------
def _fixed_triplet_boxes(bw, core):
    """
    파란 핵심 박스를 3등분한 '고정 위치' 보라색 박스들을 만든다.
    - ones: 오른쪽 끝 밀착
    - hundreds: 왼쪽 끝 밀착
    - tens: 정중앙
    - tens/hundreds는 on-비율로 '존재' 판정 후 포함, ones는 항상 포함
    반환: [ (x0,y0,x1,y1), ... ]  (좌->우 정렬)
    """
    cx0, cy0, cx1, cy1 = core
    Wc = cx1 - cx0 + 1
    Hc = cy1 - cy0 + 1
    if Wc <= 0 or Hc <= 0:
        return []

    fw = max(1, Wc // 3)      # 폭 = 핵심박스의 1/3(정수)
    fh = Hc                   # 높이 = 전체

    # 1) 일의 자리: 오른쪽 벽에 밀착
    ones_x1 = cx1
    ones_x0 = max(cx0, ones_x1 - fw + 1)

    # 2) 백의 자리: 왼쪽 벽에 밀착
    hund_x0 = cx0
    hund_x1 = min(cx1, hund_x0 + fw - 1)

    # 3) 십의 자리: '정중앙' 대신 '일의 자리 왼쪽에 바로 밀착' (겹침 방지만 적용)
    tens_x1 = ones_x0 - 1          # ← ones와 바로 붙도록 우측 경계 고정
    tens_x0 = tens_x1 - fw + 1     # 고정 폭 유지
    tens_x0 = max(hund_x1 + 1, tens_x0)  # hundreds와 겹치지 않게 좌측 클램프

    # y 범위(전체 높이)
    y0, y1 = cy0, cy1

    # 존재 판정(일의 자리는 무조건 포함)
    out = []
    def on_frac(x0, y0, x1, y1):
        roi = bw[y0:y1+1, x0:x1+1]
        return 0.0 if roi.size == 0 else (roi > 0).mean()

    # hundreds
    if hund_x1 >= hund_x0:
        if on_frac(hund_x0, y0, hund_x1, y1) >= DIGIT_PRESENCE_ON_FR:
            out.append((hund_x0, y0, hund_x1, y1))

    # tens
    if tens_x1 >= tens_x0:
        if on_frac(tens_x0, y0, tens_x1, y1) >= DIGIT_PRESENCE_ON_FR:
            out.append((tens_x0, y0, tens_x1, y1))

    # ones (항상 포함)
    if ones_x1 >= ones_x0:
        out.append((ones_x0, y0, ones_x1, y1))

    # 좌->우 정렬 보장
    out.sort(key=lambda b: b[0])
    return out

# ---------- 메인 ----------
def main():
    exts = ("*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(IN_DIR, e)))
    paths.sort()

    total = len(paths)

    # CSV 헤더 준비
    rows = ["filename,num_digits,pred_number,preds,confs,dists,states_per_digit\n"]
    ok = 0

    # 진행바 시작
    with tqdm(paths, total=total, desc="분류", unit="img", dynamic_ncols=True) as pbar:
        for i, p in enumerate(pbar):
            fname = os.path.basename(p)
            try:
                # 이미지 읽기
                bgr0 = cv2.imread(p)
                if bgr0 is None:
                    pbar.set_postfix_str(f"skip:{fname}")
                    continue

                # 0) 전처리
                bw0 = preprocess(bgr0)

                # 1) 디스큐
                bgr, bw = deskew(bgr0, bw0)

                # 2) 핵심 박스
                core = core_box(bw)

                # 3) 자릿수 박스 결정
                if DIGIT_BOX_USE_FIXED:
                    dboxes = _fixed_triplet_boxes(bw, core)
                else:
                    dboxes = _split_digits_by_valleys(bw, core)

                # 4) 자리별 분류
                per_digit = []
                preds, confs, dists, states_dump = [], [], [], []
                for dbox in dboxes:
                    pred, conf, dist, states = _classify_digit_box(bgr, bw, dbox)
                    per_digit.append((pred, conf, dist, states))
                    preds.append(str(pred))
                    confs.append(f"{conf:.3f}")
                    dists.append(str(dist))
                    states_dump.append("".join(map(str, states)))

                # 5) 최종 숫자
                try:
                    pred_number = int("".join(preds)) if preds else -1
                except Exception:
                    pred_number = -1

                # 6) 시각화 저장
                vis = draw_overlay_multi(bgr, bw, core, dboxes, per_digit)
                cv2.imwrite(os.path.join(VIS_DIR, f"{i:04d}_{pred_number}.png"), vis)

                # 7) CSV 행 추가
                rows.append(
                    f"{fname},{len(preds)},{pred_number},"
                    f"{' '.join(preds)},{' '.join(confs)},{' '.join(dists)},"
                    f"{'|'.join(states_dump)}\n"
                )
                ok += 1

                # 진행바 상태 표시(예측 결과)
                pbar.set_postfix(pred=pred_number, boxes=len(dboxes), file=fname)

            except Exception:
                # 프레임 단위 오류 시 전체 중단 없이 건너뛰기
                pbar.set_postfix_str(f"err:{fname}")
                continue

    # CSV 저장 및 요약 출력
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.writelines(rows)
    print(f"완료: {ok}장 분류 → {OUT_CSV} / 오버레이: {VIS_DIR}")


if __name__ == "__main__":
    main()


