# run_all.py 파이프라인 성능 분석 및 최적화 제안

## 현재 파이프라인 구조

영상 1개당 순차적으로 5단계가 실행됩니다.

```
[영상 1개]
   │
   ├─ (1) ffmpeg — 핸들 UI 프레임 추출   → frames30_pts_steer/
   ├─ (2) ffmpeg — 속력 프레임 추출      → frames30_pts/
   ├─ (3) classify_sevenseg.py           → _cls_result.csv
   ├─ (4) classify_steering.py           → _steer_result.csv
   └─ (5) export_speed_to_excel.py       → _speed_time.xlsx
```

---

## 병목 지점 분석

### 🔴 [High] ffmpeg가 동일 영상을 2회 디코딩 (단계 1, 2)

**현재 구조:** 같은 `.mp4` 파일을 서로 다른 영역으로 크롭하기 위해 ffmpeg 프로세스를 두 번 실행합니다.

**개선 방안:** ffmpeg의 `split` 필터를 사용하여 **1회 디코딩 → 2개 출력 스트림**으로 분기합니다.

```bash
ffmpeg -i video.mp4 \
  -filter_complex "split=2[s1][s2]; \
    [s1]crop=58:19:1017:945,fps=30[steer]; \
    [s2]crop=iw*0.013:ih*0.05:iw*0.0235:ih*0.08,fps=30[speed]" \
  -map "[steer]" frames_steer/img_%010d.png \
  -map "[speed]" frames_speed/img_%010d.png
```

> **예상 효과:** 프레임 추출 시간 약 **30~50% 단축**

---

### 🟡 [Medium] 오버레이 이미지 저장 오버헤드 (단계 4, `--overlay`)

**현재 구조:** `--overlay` 활성화 시 모든 프레임(수천 장)을 PNG로 개별 저장합니다. PNG 기본 압축 레벨(6)은 CPU를 상당히 사용합니다.

**개선 방안 1 — PNG 압축 레벨 낮추기 (즉시 적용 가능):**

```python
cv2.imwrite(str(out_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
```

> 파일 크기가 다소 늘어나지만, 저장 속도 **20~30% 단축** 가능

**개선 방안 2 — 저장 작업 스레드 병렬화:**

```python
from concurrent.futures import ThreadPoolExecutor

save_queue = []  # (path, img) 수집 후

with ThreadPoolExecutor(max_workers=4) as pool:
    pool.map(lambda args: cv2.imwrite(args[0], args[1]), save_queue)
```

---

### 🟡 [Medium] classify_steering.py 순차 이미지 로딩

**현재 구조:** 프레임을 한 장씩 순차 로딩 후 분석합니다.

```python
for i, p in enumerate(paths):
    img = cv2.imread(str(p))   # I/O 블로킹
    result = _analyze_one(img)  # CPU 연산
```

**개선 방안:** 이미지 로딩을 비동기(스레드)로 미리 prefetch하면 I/O와 CPU 연산을 겹쳐서 실행할 수 있습니다.

```python
from concurrent.futures import ThreadPoolExecutor

def load_img(p):
    return cv2.imread(str(p))

with ThreadPoolExecutor(max_workers=4) as pool:
    imgs = list(pool.map(load_img, paths))
```

---

### 🟢 [Low] 다중 영상 처리 시 순차 실행

**현재 구조:** 영상 파일이 여러 개인 경우, 하나가 끝나야 다음 영상이 시작됩니다.

**개선 방안:** 영상 단위로 `ProcessPoolExecutor` 병렬 처리

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=2) as pool:
    pool.map(process_video_wrapper, videos)
```

> ⚠️ CPU/디스크 사용량이 급증할 수 있어 `max_workers=2~3`이 안정적입니다.

---

## 우선순위 요약

| 우선순위 | 개선 방안 | 예상 효과 | 구현 난이도 |
|:---:|---|---|:---:|
| 🔴 1 | ffmpeg `split` 필터로 1회 실행 통합 | 추출 시간 ~50% 단축 | 중간 |
| 🟡 2 | PNG 압축 레벨 1로 낮추기 | 저장 시간 ~20~30% 단축 | **낮음** |
| 🟡 3 | 오버레이 저장 스레드 병렬화 | I/O 부하 감소 | 중간 |
| 🟡 4 | 이미지 로딩 prefetch 병렬화 | 分析 처리 시간 감소 | 중간 |
| 🟢 5 | 영상 단위 멀티프로세스 | 다수 영상 처리에 효과적 | 높음 |

---

## 즉시 적용 가능한 제안

코드를 변경하지 않고도 성능을 올릴 수 있는 방법:

1. **`--overlay` 옵션을 필요할 때만 사용** — 분석 자체와 오버레이 저장은 독립적이므로, 기본 분석 시에는 overlay 없이 실행하고 필요한 구간만 별도 재실행합니다.
2. **상태 변화 프레임만 저장하는 모드 추가** — L→N, N→R 등 전환이 일어나는 프레임만 저장하면 저장 파일 수를 대폭 줄일 수 있습니다.
