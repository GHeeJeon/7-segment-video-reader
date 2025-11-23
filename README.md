# 7-segment-video-reader
### 저도 영상 인식 처리는 처음 해보는데요...  

[City Car Driving - 시뮬레이션 게임](https://store.steampowered.com/app/493490/City_Car_Driving/?l=koreana) 
<details>
<summary> (클릭) 위 시뮬레이션 플레이 영상을 기반으로 </summary>
<div markdown="1">  

</br>
  
<img width="3360" height="2100" alt="490101310-a9d8fe30-939d-44aa-9070-4f70b6636482" src="https://github.com/user-attachments/assets/842c1c83-d408-44dd-803d-63e7a46efb42" />

</div>
</details>


<details>
<summary> (클릭) 위치가 고정된 속력 UI(좌측 상단) 를 30fps 단위로 크롭해서 </summary>
<div markdown="1">

</br>
  
<img width="2516" height="1532" alt="490106400-cd0db321-2237-49b6-88d7-8d8fcbce5ed3" src="https://github.com/user-attachments/assets/96df607a-2f82-49af-9cb4-4e7e067f7e94" />

</div>
</details>

<details>
<summary> (클릭) 크롭한 이미지 속 속력, 7-segment 형태의 숫자를 인식하고 </summary>
<div markdown="1">

</br>
  
<img width="2512" height="1516" alt="490106936-96327d78-621b-427b-81b5-ee9a113dcef9" src="https://github.com/user-attachments/assets/cc15ba2f-4b8a-4837-8010-de4a094806b8" />

</div>
</details>

<details>
<summary> (클릭) 자동차의 속력과 통계 데이터를 .xlsx 파일로 저장하는 프로그램입니다. </summary>
<div markdown="1">

</br>
  
<img width="3136" height="1474" alt="490119725-d0335327-1fe9-49a1-9ec1-89ec7a05c588" src="https://github.com/user-attachments/assets/4c1d2808-e8ee-48df-8fbb-f36ffbf33d8d" />

</div>
</details>


## 준비물
1. **ffmpeg** (윈도우는 `choco`로, 맥에서는 `brew` 로 설치)

</br>

**For Mac : `brew install ffmpeg`**  
macOS 용 패키지 관리자 `Homebrew` 가 없다면?  
- 터미널에서 다음의 명령어 실행 후 `ffmpeg` 설치
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
- `ffmpeg` 를 입력했을 때 버전 등의 정보가 출력되면 설치 성공

</br>

**For Windows : `choco install ffmpeg`**  
Windows 용 커맨드 라인 패키지 매니저 `Chocolatey` 가 없다면?  
다음의 절차 진행 후  `ffmpeg` 설치  
- **관리자 권한** 으로 PowerShell 실행하기  
- 다음의 명령어 실행하기  
```bash
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```
- `choco` 를 입력하여 버전이 나오면 설치 성공

</br>

3. 시뮬레이션 플레이 영상 (파일명은 자유, 형식은 `.mp4` 권장)
4. 파이썬 가상환경 세팅  
For Mac
`python3 -m venv .venv`  
`source .venv/bin/activate`  
`pip install -r requirements.txt`

</br>

For Windows
`python -m venv .venv`  
`.venv\Scripts\Activate.ps1`  
`pip install -r requirements.txt`  

## 어떻게 사용하냐면요... 
1. `source/[플레이어 이름]/[번호]/` 위치에 시뮬레이션 플레이 영상을 **하나씩** 추가해요

</br>

![Oct-10-2025 00-43-08](https://github.com/user-attachments/assets/facc73e8-5d31-44cc-b5b5-694bf2e8356e)


2. 한 플레이어가 여러 번 주행했을 경우 **다음 번호에 동영상을 추가**해요

</br>

![Oct-10-2025 00-52-19](https://github.com/user-attachments/assets/3fc3be50-d27c-40f1-9efa-fe73eff4a4ec)

3. **여러 플레이어**를 추가할 수 있어요

</br>

![Oct-10-2025 00-57-23](https://github.com/user-attachments/assets/8b3b93e8-64a7-441e-bc4c-e32876963ac3)

4. 프로젝트 최상단 디렉토리에서 다음의 명령어 중 하나를 실행해요 [arguments](https://github.com/GHeeJeon/7-segment-video-reader/blob/main/arguments.md)
```shell
python run_all.py
python3 run_all.py
```

5. 기다려요. 완료 메시지가 뜰 때까지요!  
5분 57초 영상 하나를 30fps 기준으로 완료하는데 3분 41초가 걸렸어요. (1만 장 이상의 이미지 처리 필요)

</br>

![Oct-10-2025 01-19-15](https://github.com/user-attachments/assets/1c2eec9c-6abe-4bbc-8226-b00228ddb86a)


## `run_all.py` 를 실행하면요...
1. 동영상의 **좌측 상단 속력 UI** 부분을 크롭해, 지정된 **프레임 단위로 캡쳐**해요 (기본 `30fps`)
2. 캡쳐한 이미지를 `source/[플레이어 이름]/[번호]/[동영상 이름]/frames30_pts` 위치에 저장해요
3. `frames30_pts` 폴더 속 이미지에서 **7-segment 형태의 숫자를 추출**해 `_cls_result.csv` 로 저장해요
4. `_cls_result.csv` 데이터를 바탕으로 통계를 계산해 `_speed_time.xlsx` 로 저장해요

</br>

## 추가기능: `run_all.py` 실행 완료 후 전체 요약 통계 추출하기
`run_all.py`로 `_speed_time.xlsx` 를 생성한 후에...
1. 프로젝트 최상단 디렉토리에서 다음의 명령어 중 하나를 실행해요
```shell
python total_statistics.py
python3 total_statistics.py
```
2. `total_speed_statistics.xlsx` 파일이 생성되었어요
