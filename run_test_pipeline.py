
import os
import sys
import glob
import shutil
import subprocess
import argparse
from pathlib import Path

# ----------------------
# Defaults (editable)
# ----------------------
DEFAULT_FPS = 30
DEFAULT_CROP = "crop=iw*0.013:ih*0.05:iw*0.0235:ih*0.08"
FRAMES_DIRNAME = "frames30_pts"      # must match classify_sevenseg's expected input dir
CLS_CSV_NAME = "_cls_result.csv"     # default output name from classify_sevenseg.py

SUPPORTED_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV", ".MKV", ".AVI")

def run(cmd, cwd=None):
    print(">>", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, cwd=cwd)

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        print("[ERROR] ffmpeg가 설치되어 있지 않거나 PATH에 없습니다.")
        print(" - Windows: https://ffmpeg.org/download.html 에서 설치 후 PATH 추가")
        print(" - macOS:   brew install ffmpeg")
        sys.exit(1)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def clear_dir(path: Path):
    if path.exists():
        for p in path.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    else:
        path.mkdir(parents=True, exist_ok=True)

def list_videos(test_dir: Path):
    vids = []
    for ext in SUPPORTED_VIDEO_EXTS:
        vids.extend(test_dir.glob(f"*{ext}"))
    return sorted(vids)

def main():
    parser = argparse.ArgumentParser(description="Run 7-seg test pipeline (ffmpeg -> classify -> export)")
    parser.add_argument("--test-dir", default="test", help="테스트 디렉토리 (기본: test)")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="FFmpeg FPS (기본: 30)")
    parser.add_argument("--crop", default=DEFAULT_CROP, help="FFmpeg crop 필터 (기본: %(default)s)")
    parser.add_argument("--keep-frames", action="store_true", help="프레임 이미지 보존 (기본: 삭제)")
    parser.add_argument("--video", default=None, help="특정 비디오만 처리 (파일명 또는 경로)")
    parser.add_argument("--python", default=sys.executable, help="파이썬 실행 경로 (기본: 현재 인터프리터)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    test_dir = (repo_root / args.test_dir).resolve()

    classify_py = (repo_root / "classify_sevenseg.py").resolve()
    export_py   = (repo_root / "export_speed_to_excel.py").resolve()

    if not classify_py.exists():
        print(f"[ERROR] classify_sevenseg.py 를 찾을 수 없습니다: {classify_py}")
        sys.exit(1)
    if not export_py.exists():
        print(f"[ERROR] export_speed_to_excel.py 를 찾을 수 없습니다: {export_py}")
        sys.exit(1)

    check_ffmpeg()

    frames_dir = test_dir / FRAMES_DIRNAME
    ensure_dir(test_dir)
    ensure_dir(frames_dir)

    # 비디오 목록 결정
    if args.video:
        vpath = Path(args.video)
        if not vpath.is_absolute():
            vpath = test_dir / vpath
        if not vpath.exists():
            print(f"[ERROR] 지정한 비디오 파일을 찾을 수 없습니다: {vpath}")
            sys.exit(1)
        videos = [vpath]
    else:
        videos = list_videos(test_dir)
        if not videos:
            print(f"[ERROR] {test_dir} 에서 비디오 파일을 찾지 못했습니다. (확장자: {', '.join(SUPPORTED_VIDEO_EXTS)})")
            sys.exit(1)

    results = []
    for v in videos:
        stem = v.stem
        print(f"\n=== Processing: {v.name} ===")

        # 1) 프레임 추출 (frames30_pts를 비움)
        if not args.keep_frames:
            print(f"[INFO] {frames_dir} 비우는 중...")
            clear_dir(frames_dir)

        # ffmpeg 실행
        out_pattern = str(frames_dir / "img_%010d.png")
        vf = f"{args.crop},fps={args.fps}"
        run([
            "ffmpeg", "-y",
            "-i", str(v),
            "-vf", vf,
            "-frame_pts", "1",
            out_pattern
        ])

        # 프레임 생성 확인
        frames = sorted(frames_dir.glob("img_*.png"))
        if not frames:
            print(f"[ERROR] 프레임이 생성되지 않았습니다: {frames_dir}")
            sys.exit(1)

        # 2) classify 실행 (cwd=test_dir, 기본 출력: _cls_result.csv)
        print("[INFO] classify_sevenseg.py 실행...")
        run([args.python, str(classify_py)], cwd=str(test_dir))

        cls_csv = test_dir / CLS_CSV_NAME
        if not cls_csv.exists():
            print(f"[ERROR] 분류 결과 CSV({CLS_CSV_NAME})가 생성되지 않았습니다.")
            sys.exit(1)

        # 비디오별로 결과 보존 (이름에 stem 추가)
        cls_csv_renamed = test_dir / f"_cls_result__{stem}.csv"
        if cls_csv_renamed.exists():
            cls_csv_renamed.unlink()
        cls_csv.rename(cls_csv_renamed)

        # 3) export 실행 (입력/출력 파일명 지정)
        out_xlsx = test_dir / f"_speed_time__{stem}.xlsx"
        print("[INFO] export_speed_to_excel.py 실행...")
        run([
            args.python, str(export_py),
            "--cls_csv", str(cls_csv_renamed),
            "--out_xlsx", str(out_xlsx),
            "--fps", str(args.fps)
        ])

        if not out_xlsx.exists():
            print(f"[ERROR] 엑셀 결과가 생성되지 않았습니다: {out_xlsx}")
            sys.exit(1)

        results.append((v.name, cls_csv_renamed.name, out_xlsx.name))

        # 다음 비디오 처리를 위해 프레임 정리
        if not args.keep_frames:
            clear_dir(frames_dir)

    # 요약 출력
    print("\n=== Summary ===")
    for vname, csvname, xlsxname in results:
        print(f"- Video: {vname}")
        print(f"  CSV:   {csvname}")
        print(f"  XLSX:  {xlsxname}")
    print("\nDone.")

if __name__ == "__main__":
    main()
