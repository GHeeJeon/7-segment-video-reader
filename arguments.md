# 요약
| 항목 | `run_all.py` | `classify_sevenseg.py` | `export_speed_to_excel.py` |
|------|---------------|------------------------|-----------------------------|
| 주요 기능 | 전체 자동 파이프라인 | 7세그 인식 (CSV 생성) | CSV → Excel 변환 |
| 결과물 | `_cls_result.csv`, `_speed_time.xlsx` | `_cls_result.csv`, `_cls_overlay/` | `_speed_time.xlsx` |
| 실행 방식 | `python run_all.py` | `python classify_sevenseg.py` | `python export_speed_to_excel.py` |
| 주요 옵션 | `--overlay`, `--debug`, `--all-cols` | `--overlay` | `--debug`, `--all-cols` |

# run_all.py
> 전체 파이프라인 실행 스크립트  
> (ffmpeg 추출 → classify → export)

| 단축어 | 전체 이름 | 타입 | 기본값 | 설명 |
|:--:|:--|:--:|:--:|:--|
| `-r` | `--root` | `str` | `.` | 스크립트 기준 루트 경로. 예: `./video` |
| `-s` | `--source` | `str` | `source` | 사람 폴더들이 들어 있는 상위 폴더 이름. 예: `source/사람A/1/*.mp4` |
| `-f` | `--fps` | `int` | `30` | 프레임 추출 시 초당 프레임 수 (FPS) |
| `-c` | `--crop` | `str` | `"crop=iw*0.013:ih*0.05:iw*0.0235:ih*0.08"` | ffmpeg crop 필터 문자열 (영상 일부 영역만 추출) |
| `-v` | `--video-ext` | `str` | `.mp4` | 처리할 동영상 확장자 목록 (콤마 구분). 예: `mp4,mov,avi` |
| `-x` | `--skip-existing` | `flag` | 없음 | 이미 `_speed_time.xlsx` 파일이 있으면 해당 영상은 스킵 |
| `-w` | `--workers` | `int` | `1` | 동시 처리할 영상 수 (병렬 처리). 기본 1은 순차 실행 |
| `-o` | `--overlay` | `flag` | 없음 | 인식 결과 오버레이 이미지를 저장 (`classify_sevenseg`에 전달) |
| `-d` | `--debug` | `flag` | 없음 | 디버그 로그 출력 (`export_speed_to_excel`에 전달) |
| `-a` | `--all-cols` | `flag` | 없음 | 모든 컬럼 표시 (`export_speed_to_excel`에 전달) |

**실행 예시**
```bash
python run_all.py -r ./video -f 30 -o -d -a -x
```

---

# classify_sevenseg.py
> 영상에서 7-세그먼트 숫자를 인식하여 `_cls_result.csv` 생성

| 단축어 | 전체 이름 | 타입 | 기본값 | 설명 |
|:--:|:--|:--:|:--:|:--|
| `-o` | `--overlay` | `flag` | 없음 | 인식 결과를 이미지로 오버레이하여 `_cls_overlay/` 폴더에 저장 |

**단독 실행 예시**
```bash
python classify_sevenseg.py -o
```

> 💡 `run_all.py`에서는 `classify_sevenseg`의 `IN_DIR`, `OUT_CSV`, `VIS_DIR` 값을 코드에서 직접 지정합니다.  
> 따라서 `run_all.py` 실행 시에는 CLI로 전달하지 않아도 됩니다.

---

# export_speed_to_excel.py
> `_cls_result.csv`를 읽어 `_speed_time.xlsx`로 변환  
> 속도·시간 계산 및 필터링 수행

| 단축어 | 전체 이름 | 타입 | 기본값 | 설명 |
|:--:|:--|:--:|:--:|:--|
| `-c` | `--cls_csv` | `str` | `_cls_result.csv` | 입력 CSV 경로 |
| `-o` | `--out_xlsx` | `str` | `_speed_time.xlsx` | 출력 XLSX 경로 |
| `-f` | `--fps` | `int` | `30` | FPS (초당 프레임 수) |
| `-d` | `--debug` | `flag` | 없음 | 디버그 출력 활성화 |
| `-a` | `--all-cols` | `flag` | 없음 | 모든 컬럼 표시 (num_digits, preds, confs 등 포함) |

**단독 실행 예시**
```bash
python export_speed_to_excel.py -c ./video1/_cls_result.csv -o ./video1/_speed_time.xlsx -f 30 -d -a
```

---

# classify_steering.py
> 영상에서 핸들 UI 각도와 움직임을 분석하여 `_steer_result.csv` 생성

| 단축어 | 전체 이름 | 타입 | 기본값 | 설명 |
|:--:|:--|:--:|:--:|:--|
| (없음) | `--in-dir` | `str` | `.` | 입력 핸들 프레임 풀더 경로 |
| (없음) | `--work-dir` | `str` | `.` | 결과 CSV 저장 경로 |
| (없음) | `--fps` | `float` | `30.0` | FPS (초당 프레임 수, 시간 계산용) |
| (없음) | `--overlay` | `flag` | 없음 | 샘플 디버그 이미지 저장 활성화 |
| (없음) | `--overlay-every` | `int` | `30` | 오버레이 저장 주기 |
| (없음) | `--thresh` | `int` | `200` | 이미지 이진화 threshold 값 |
| (없음) | `--zero-center-x` | `int` | `26` | 0 라벨 기준 픽셀 x 좌표 |
| (없음) | `--x-bias` | `int` | `0` | 감지점 x 좌표 보정량 |
| (없음) | `--tol` | `int` | `6` | 0 라벨(중앙) 판정 허용 오차 범위(px) |

**단독 실행 예시**
```bash
# 기본값으로 현재 폴더에서 바로 실행
python classify_steering.py

# 경로 및 옵션 지정 실행
python classify_steering.py --in-dir ./frames30_pts_steer --work-dir . --fps 30.0 --overlay
```
