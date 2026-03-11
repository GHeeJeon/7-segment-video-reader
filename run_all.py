"""
One-click pipeline: extract frames → classify 7-seg → export Excel

시나리오
1) 이 스크립트와 `classify_sevenseg.py`, `export_speed_to_excel.py`를 같은 폴더에 둔다.
2) 압축을 풀어 나온 상위 폴더(그 안에 1, 2, 3, 4/ 각 폴더에 동영상 2개)를 
   이 스크립트가 있는 폴더로 옮긴다.
3) 이 스크립트를 실행하면 1,2,3,4 폴더를 순회하며:
   - ffmpeg로 지정한 크롭+FPS=30으로 프레임 추출 →  `1/동영상이름/frames30_pts/img_0000000001.png` ...
   - 추출 폴더를 IN_DIR로 하여 `classify_sevenseg.py` 실행 → `_cls_result.csv` / `_cls_overlay/`
   - 그 CSV를 입력으로 `export_speed_to_excel.py` 실행 → `1/동영상이름/_speed_time.xlsx`

요구 패키지: ffmpeg (PATH 등록), opencv-python, numpy, pandas, xlsxwriter, openpyxl, tqdm

사용 예: python .\run_all.py --root ".\video" --fps 30 --video-ext mp4,mov --skip-existing
"""
from __future__ import annotations
import os
import sys
import glob
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

# tqdm (optional)
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# --------------------------- 설정 기본값 ---------------------------
DEFAULT_FPS = 30
DEFAULT_CROP = "crop=iw*0.013:ih*0.05:iw*0.0235:ih*0.08"
DEFAULT_VID_EXTS = (".mp4",)  # 쉼표로 추가 가능
ROI_STEER = (1017, 945, 1075, 964)  # (x1, y1, x2, y2) — 58*19

# --------------------------- 유틸 ---------------------------

def which_or_raise(exe: str) -> str:
    path = shutil.which(exe)
    if path is None:
        raise RuntimeError(f"'{exe}' 실행파일을 찾을 수 없습니다. PATH에 추가했는지 확인하세요.")
    return path


def find_videos(source_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    """
    source 하위 모든 폴더를 재귀적으로 순회하며 동영상 파일 검색
    예시:
      source/
        사람A/
          1/
            1-1/
              video.mp4
          2/
            video.mov
        사람B/
          video.mp4
    """
    if not source_dir.is_dir():
        raise RuntimeError(f"source 폴더를 찾을 수 없습니다: {source_dir}")

    videos: List[Path] = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            path = Path(root) / f
            if path.suffix.lower() in exts:
                videos.append(path)

    return sorted(videos)


def run_ffmpeg_split(
    video: Path,
    speed_frames_dir: Path,
    steer_frames_dir: Path,
    speed_crop: str,
    steer_crop: str,
    fps: int,
) -> None:
    """
    ffmpeg를 1회만 실행하여 속력 UI와 핸들 UI 프레임을 동시에 추출합니다.
    split 필터를 사용하므로 영상 디코딩이 1회만 발생합니다.
    """
    speed_frames_dir.mkdir(parents=True, exist_ok=True)
    steer_frames_dir.mkdir(parents=True, exist_ok=True)

    speed_out = str(speed_frames_dir / "img_%010d.png")
    steer_out = str(steer_frames_dir / "img_%010d.png")

    # split 필터: 동일 디코딩 스트림을 두 브랜치로 분기
    filter_complex = (
        f"split=2[s_speed][s_steer];"
        f"[s_speed]{speed_crop},fps={fps}[out_speed];"
        f"[s_steer]{steer_crop},fps={fps}[out_steer]"
    )

    cmd = [
        which_or_raise("ffmpeg"),
        "-y",
        "-i", str(video),
        "-filter_complex", filter_complex,
        # 속력 UI 출력
        "-map", "[out_speed]",
        "-frame_pts", "1",
        speed_out,
        # 핸들 UI 출력
        "-map", "[out_steer]",
        "-frame_pts", "1",
        steer_out,
    ]

    # Windows 등에서 한글 인코딩(cp949) 충돌 방지를 위해 바이트로 받고 수동 디코딩
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr_text = result.stderr.decode("utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 실패: {video.name}\n{stderr_text[-800:]}")

    created_speed = list(speed_frames_dir.glob("img_*.png"))
    created_steer = list(steer_frames_dir.glob("img_*.png"))
    if not created_speed:
        raise RuntimeError(f"ffmpeg가 속력 프레임을 생성하지 못했습니다: {video}")
    if not created_steer:
        raise RuntimeError(f"ffmpeg가 핸들 프레임을 생성하지 못했습니다: {video}")



def classify_frames(frames_dir: Path, work_dir: Path, overlay: bool = False) -> Path:
    """classify_sevenseg.py 모듈 실행"""
    import importlib
    cls = importlib.import_module("classify_sevenseg")

    out_csv = work_dir / "_cls_result.csv"
    vis_dir = work_dir / "_cls_overlay"
    vis_dir.mkdir(parents=True, exist_ok=True)

    cls.IN_DIR = str(frames_dir)
    cls.OUT_CSV = str(out_csv)
    cls.VIS_DIR = str(vis_dir)
    
    cls.main(overlay=overlay)

    if not out_csv.exists():
        raise RuntimeError(f"분류 CSV가 생성되지 않았습니다: {out_csv}")
    return out_csv


def classify_steer_frames(frames_dir: Path, work_dir: Path, fps: int, overlay: bool = False) -> Path:
    """classify_steering.py 모듈 실행 (frames30_pts_steer 입력)"""
    import importlib
    mod = importlib.import_module("classify_steering")

    out_csv = work_dir / "_steer_result.csv"

    # classify_steering exposes analyze_steering_frames(frames_dir, work_dir, fps, ...)
    mod.analyze_steering_frames(
        frames_dir=Path(frames_dir),
        work_dir=Path(work_dir),
        fps=float(fps),
        overlay=overlay,
    )

    if not out_csv.exists():
        raise RuntimeError(f"steer CSV가 생성되지 않았습니다: {out_csv}")
    return out_csv

def export_excel(cls_csv: Path, steer_csv: Path, frames_dir: Path, fps: int, debug: bool = False, all_cols: bool = False) -> Path:
    """export_speed_to_excel.py 모듈 실행"""
    import importlib
    ex = importlib.import_module("export_speed_to_excel")
    out_xlsx = cls_csv.parent / "_speed_time.xlsx"

    ex.export_speed_xlsx(
        cls_csv_path=str(cls_csv),
        out_xlsx_path=str(out_xlsx),
        fps=fps,
        debug=debug,
        show_tech_cols=all_cols,
        steer_csv_path=str(steer_csv) if steer_csv and steer_csv.exists() else None,
    )
    if not out_xlsx.exists():
        raise RuntimeError(f"엑셀 파일이 생성되지 않았습니다: {out_xlsx}")
    return out_xlsx

def roi_to_crop(roi: Tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = roi
    w = x2 - x1
    h = y2 - y1
    return f"crop={w}:{h}:{x1}:{y1}"

# --------------------------- 메인 파이프라인 ---------------------------

def process_video(video: Path, crop: str, fps: int, overlay: bool, debug: bool, all_cols: bool) -> Path:
    parent = video.parent
    stem = video.stem
    work_dir = parent / stem

    # 1) ffmpeg 1회 실행으로 핸들 UI + 속력 UI 프레임 동시 추출 (split 필터)
    steer_crop = roi_to_crop(ROI_STEER)
    steer_frames_dir = work_dir / "frames30_pts_steer"
    frames_dir = work_dir / "frames30_pts"

    run_ffmpeg_split(
        video=video,
        speed_frames_dir=frames_dir,
        steer_frames_dir=steer_frames_dir,
        speed_crop=crop,
        steer_crop=steer_crop,
        fps=fps,
    )

    # 2) 속력 인식 분석 (classify_sevenseg.py)
    cls_csv = classify_frames(frames_dir, work_dir, overlay=overlay)

    # 3) 핸들 각도 분석 (classify_steering.py)
    steer_csv = work_dir / "_steer_result.csv"
    try:
        classify_steer_frames(steer_frames_dir, work_dir, fps=fps, overlay=overlay)
    except Exception as e:
        print(f"[경고] steering 분석 중 오류(무시하고 진행): {e}")

    # 4) 엑셀 파일 내보내기 (export_speed_to_excel.py)
    xlsx = export_excel(cls_csv, steer_csv, frames_dir, fps=fps,
                        debug=debug, all_cols=all_cols)

    return xlsx


def _process_one(
    video_path: str,
    crop: str,
    fps: int,
    overlay: bool,
    debug: bool,
    all_cols: bool,
    skip_existing: bool,
) -> Tuple[str, str, str]:
    """영상 1개 전체 파이프라인 실행 함수 (ProcessPoolExecutor에서 호출됩니다).

    반환값: (video_path, xlsx_path_or_empty, status)
        status: 'ok' | 'ok-export-only' | 'skipped' | 'error:...' 중 하나

    주의: ProcessPoolExecutor에서 호출되므로 반드시 최상위(top-level) 함수여야 합니다.
    클래스 메서드나 람다는 pickle 직렬화가 안 되어 오류가 발생합니다.
    각 프로세스는 독립된 메모리와 출력 경로를 가지므로 데이터가 섞이지 않습니다.
    """
    video = Path(video_path)
    work_dir  = video.parent / video.stem
    cls_csv   = work_dir / "_cls_result.csv"
    out_xlsx  = work_dir / "_speed_time.xlsx"
    steer_csv = work_dir / "_steer_result.csv"

    # 스킵 조건: 이미 결과 파일이 있는 경우
    if skip_existing and out_xlsx.exists():
        return video_path, str(out_xlsx), "skipped"

    # export-only 조건: 분류 CSV는 있지만 엑셀이 없는 경우
    if cls_csv.exists() and not out_xlsx.exists():
        try:
            xlsx = export_excel(cls_csv, steer_csv, cls_csv.parent, fps)
            return video_path, str(xlsx), "ok-export-only"
        except Exception as e:
            return video_path, "", f"error:{e}"

    # 기본 전체 파이프라인 실행
    try:
        xlsx = process_video(
            video, crop=crop, fps=fps,
            overlay=overlay, debug=debug, all_cols=all_cols,
        )
        return video_path, str(xlsx), "ok"
    except Exception as e:
        return video_path, "", f"error:{e}"


def main():
    import argparse
    ap = argparse.ArgumentParser(description="ffmpeg 추출 → 분류 → 엑셀 내보내기")
    ap.add_argument("-r", "--root", default=".", help="스크립트 기준 루트 경로")
    ap.add_argument("-s", "--source", default="source", help="사람 폴더들을 담은 상위 폴더(경로/이름)")
    ap.add_argument("-f", "--fps", type=int, default=DEFAULT_FPS, help="프레임 추출 FPS (기본 30)")
    ap.add_argument("-c", "--crop", default=DEFAULT_CROP, help="ffmpeg crop 필터 문자열")
    ap.add_argument("-v", "--video-ext", default=",".join(DEFAULT_VID_EXTS), help="처리할 동영상 확장자 콤마구분(.mp4,.mov)")
    ap.add_argument("-x", "--skip-existing", action="store_true", help="이미 _speed_time.xlsx가 있으면 해당 영상은 건너뜀")

    # 병렬 처리 옵션
    ap.add_argument("-w", "--workers", type=int, default=1,
                    help="동시 처리 영상 수 (기본 1=순차실행). "
                         "게이밍 PC 권장: 4~6 / 일반 PC 권장: 2~3. "
                         "높을수록 빠르지만 CPU/디스크 부하 증가.")

    # classify_sevenseg 관련 옵션
    ap.add_argument("-o", "--overlay", action="store_true", help="인식 결과 오버레이 이미지 저장")

    # export_speed_to_excel 관련 옵션
    ap.add_argument("-d", "--debug", action="store_true", help="디버그 출력 (export_speed_to_excel.py)")
    ap.add_argument("-a", "--all-cols", action="store_true", help="모든 컬럼 표시 (num_digits, preds 등 포함)")

    args = ap.parse_args()

    root = Path(args.root).resolve()
    source_dir = (root / args.source).resolve()
    exts = tuple(
        s.strip().lower() if s.strip().startswith(".") else "." + s.strip().lower()
        for s in args.video_ext.split(",") if s.strip()
        )

    # ffmpeg 확인
    which_or_raise("ffmpeg")

    videos = find_videos(source_dir, exts)
    if not videos:
        print(f"동영상을 찾지 못했습니다. source_dir: {source_dir}, 확장자: {exts}")
        sys.exit(1)

    print(f"총 {len(videos)}개 동영상 처리 시작… source_dir={source_dir}")
    if args.workers > 1:
        print(f"  병렬 모드: --workers {args.workers} (동시 처리)")
    else:
        print(f"  순차 모드: --workers 1 (기본)")

    # 공통 인자 딕셔너리 (모든 영상에 동일하게 전달)
    common = dict(
        crop=args.crop,
        fps=args.fps,
        overlay=args.overlay,
        debug=args.debug,
        all_cols=args.all_cols,
        skip_existing=args.skip_existing,
    )

    results: List[Tuple[Path, str, str]] = []

    if args.workers <= 1:
        # ─── 순차 실행 (workers=1, 기본값) ───────────────────────────────
        bar = tqdm(videos, desc="전체 파이프라인", unit="vid", dynamic_ncols=True) if tqdm else videos
        for v in bar:
            vp, xlsx, status = _process_one(str(v), **common)
            results.append((Path(vp), xlsx, status))
            if tqdm:
                bar.set_postfix_str(f"{status}:{v.name}")

    else:
        # ─── 병렬 실행 (workers > 1) ────────────────────────────────────
        # ProcessPoolExecutor: 각 프로세스가 독립 메모리 → 데이터 혼용 없음
        # as_completed: 먼저 끝난 프로세스부터 결과를 받아 진행 바를 즉시 업데이트
        bar = tqdm(total=len(videos), desc="전체 파이프라인", unit="vid", dynamic_ncols=True) if tqdm else None
        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for v in videos:
                future = pool.submit(_process_one, str(v), **common)
                futures[future] = v

            try:
                for future in as_completed(futures):
                    v = futures[future]
                    try:
                        vp, xlsx, status = future.result()
                    except Exception as e:
                        vp, xlsx, status = str(v), "", f"error:{e}"
                    results.append((Path(vp), xlsx, status))
                    if bar:
                        bar.set_postfix_str(f"{status}:{v.name}")
                        bar.update(1)
            except KeyboardInterrupt:
                print("\n[알림] 사용자에 의해 병렬 작업이 중단되었습니다. 작업 취소 중...")
                for future in futures:
                    future.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)

        if bar:
            bar.close()

    # ─── 결과 요약 ────────────────────────────────────────────────────
    ok_cnt   = sum(1 for _, _, s in results if s.startswith("ok"))
    skip_cnt = sum(1 for _, _, s in results if s == "skipped")
    err_cnt  = sum(1 for _, _, s in results if s.startswith("error"))

    print(f"완료. 성공 {ok_cnt}, 스킵 {skip_cnt}, 오류 {err_cnt}")
    for v, x, s in results:
        if s.startswith("ok"):
            suffix = " (엑셀만)" if s == "ok-export-only" else ""
            print(f"  ✔ {v} → {x}{suffix}")
        elif s == "skipped":
            print(f"  ↷ {v} (기존 결과 유지)")
        else:
            print(f"  ✖ {v} — {s}")


# ──────────────────────────────────────────────────────────────────────────────
# Windows / macOS spawn 안전 가드
# ProcessPoolExecutor는 새 프로세스 시작 시 이 파일 전체를 다시 import합니다.
# 이 가드가 없으면 프로세스가 무한히 자기 자신을 복제합니다.
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
