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

# --------------------------- 유틸 ---------------------------

def which_or_raise(exe: str) -> str:
    path = shutil.which(exe)
    if path is None:
        raise RuntimeError(f"'{exe}' 실행파일을 찾을 수 없습니다. PATH에 추가했는지 확인하세요.")
    return path


def find_videos(source_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    """
    구조:
      source/
        사람A/
          1/ 2/ (3/ 4/ 있을 수도)
            *.mp4, *.mov ...
        사람B/
          1/ 2/
            *.mp4 ...
    """
    videos: List[Path] = []
    if not source_dir.is_dir():
        raise RuntimeError(f"source 폴더를 찾을 수 없습니다: {source_dir}")

    for person_dir in sorted([p for p in source_dir.iterdir() if p.is_dir()]):
        # 우선 1,2,3,4만 대상으로
        numbered = [person_dir / d for d in ["1", "2", "3", "4"] if (person_dir / d).is_dir()]
        # 없으면 사람 폴더 바로 하위의 모든 디렉터리를 대상으로(유연성)
        if not numbered:
            numbered = [p for p in person_dir.iterdir() if p.is_dir()]

        for stage_dir in sorted(numbered):
            for p in sorted(stage_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in exts:
                    videos.append(p)

    return videos



def run_ffmpeg(video: Path, frames_dir: Path, crop: str, fps: int) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(frames_dir / "img_%010d.png")
    cmd = [
        which_or_raise("ffmpeg"),
        "-y",
        "-i", str(video),
        "-vf", f"{crop},fps={fps}",
        "-frame_pts", "1",
        out_pattern,
    ]
    # ffmpeg 출력은 꽤 큼: 그대로 stderr로 흘려보냄
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 실패: {video.name}\n{result.stderr[-800:]}" )

    # 최소 한 장 생성 확인
    created = list(frames_dir.glob("img_*.png"))
    if not created:
        raise RuntimeError(f"ffmpeg가 프레임을 생성하지 못했습니다: {video}")


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
    cls.OVERLAY = overlay  # 인식 이미지 생성하는 argument 전달

    cls.main()

    if not out_csv.exists():
        raise RuntimeError(f"분류 CSV가 생성되지 않았습니다: {out_csv}")
    return out_csv

def export_excel(cls_csv: Path, frames_dir: Path, fps: int, debug: bool = False, all_cols: bool = False) -> Path:
    """export_speed_to_excel.py 모듈 실행"""
    import importlib
    ex = importlib.import_module("export_speed_to_excel")
    out_xlsx = cls_csv.parent / "_speed_time.xlsx"

    ex.export_speed_xlsx(
        cls_csv_path=str(cls_csv),
        out_xlsx_path=str(out_xlsx),
        fps=fps,
        debug=debug,
        all_cols=all_cols,
    )
    if not out_xlsx.exists():
        raise RuntimeError(f"엑셀 파일이 생성되지 않았습니다: {out_xlsx}")
    return out_xlsx

# --------------------------- 메인 파이프라인 ---------------------------

def process_video(video: Path, crop: str, fps: int, overlay: bool, debug: bool, all_cols: bool) -> Path:
    parent = video.parent
    stem = video.stem
    work_dir = parent / stem
    frames_dir = work_dir / "frames30_pts"

    # 1) ffmpeg 추출
    run_ffmpeg(video, frames_dir, crop=crop, fps=fps)

    # 2) 분류 (overlay 플래그 전달)
    cls_csv = classify_frames(frames_dir, work_dir, overlay=overlay)

    # 3) 엑셀 내보내기 (debug, all-cols 플래그 전달)
    xlsx = export_excel(cls_csv, frames_dir, fps=fps, debug=debug, all_cols=all_cols)

    return xlsx


def main():
    import argparse
    ap = argparse.ArgumentParser(description="ffmpeg 추출 → 분류 → 엑셀 내보내기")
    ap.add_argument("--root", "-r", default=".", help="스크립트 기준 루트 경로")
    ap.add_argument("--source", "-s", default="source", help="사람 폴더들을 담은 상위 폴더(경로/이름)")
    ap.add_argument("--fps", "-f", type=int, default=DEFAULT_FPS, help="프레임 추출 FPS (기본 30)")
    ap.add_argument("--crop", "-c", default=DEFAULT_CROP, help="ffmpeg crop 필터 문자열")
    ap.add_argument("--video-ext", "-v", default=",".join(DEFAULT_VID_EXTS), help="처리할 동영상 확장자 콤마구분(.mp4,.mov)")
    ap.add_argument("--skip-existing", "-x", action="store_true", help="이미 _speed_time.xlsx가 있으면 해당 영상은 건너뜀")

    # classify_sevenseg 관련 옵션
    ap.add_argument("--overlay", "-o", action="store_true", help="인식 결과 오버레이 이미지 저장 (classify_sevenseg.py)")

    # export_speed_to_excel 관련 옵션
    ap.add_argument("--debug", "-d", action="store_true", help="디버그 출력 (export_speed_to_excel.py)")
    ap.add_argument("--all-cols", "-a", action="store_true", help="모든 컬럼 표시 (num_digits, preds 등 포함)")

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

    bar = tqdm(videos, desc="전체 파이프라인", unit="vid", dynamic_ncols=True) if tqdm else videos
    results = []

    for v in bar:
        try:
            work_dir = v.parent / v.stem
            cls_csv = work_dir / "_cls_result.csv"
            out_xlsx = work_dir / "_speed_time.xlsx"

            # --- 스킵 조건 ---
            if args.skip_existing and out_xlsx.exists():
                if tqdm: bar.set_postfix_str(f"skip:{v.name}")
                results.append((v, out_xlsx, "skipped"))
                continue

            # --- export-only 조건 ---
            if cls_csv.exists() and not out_xlsx.exists():
                if tqdm: bar.set_postfix_str(f"export_only:{v.name}")
                xlsx = export_excel(cls_csv, cls_csv.parent, args.fps)
                results.append((v, xlsx, "ok-export-only"))
                if tqdm: bar.set_postfix_str(f"ok-export-only:{v.name}")
                continue

            # --- 기본 전체 파이프라인 실행 ---
            if tqdm: bar.set_postfix_str(f"ffmpeg:{v.name}")
            xlsx = process_video(
                v,
                crop=args.crop,
                fps=args.fps,
                overlay=args.overlay,
                debug=args.debug,
                all_cols=args.all_cols,
            )
            results.append((v, xlsx, "ok"))
            if tqdm: bar.set_postfix_str(f"ok:{v.name}")

        except Exception as e:
            results.append((v, None, f"error:{e}"))
            if tqdm:
                bar.set_postfix_str(f"err:{v.name}")
            else:
                print(f"[오류] {v}: {e}")

    # 요약
    ok_cnt = sum(1 for _, _, s in results if s == "ok")
    err_cnt = sum(1 for _, _, s in results if isinstance(s, str) and s.startswith("error"))
    skip_cnt = sum(1 for _, _, s in results if s == "skipped")

    print(f"완료. 성공 {ok_cnt}, 스킵 {skip_cnt}, 오류 {err_cnt}")
    for v, x, s in results:
        if s == "ok":
            print(f"  ✔ {v} → {x}")
        elif s == "skipped":
            print(f"  ↷ {v} (기존 결과 유지)")
        else:
            print(f"  ✖ {v} — {s}")


if __name__ == "__main__":
    main()
