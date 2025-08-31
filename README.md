# 7-segment-video-reader
저도 영상 인식 처리는 처음 해보는데요...  
[City Car Driving - 시뮬레이션 게임](https://store.steampowered.com/app/493490/City_Car_Driving/?l=koreana) 플레이 영상을 기반으로  
주행 시간 동안의 자동차의 속력을 `.xlsx` 파일로 저장하는 프로그램입니다.

## 어떤 게 필요하냐면요...
1. ffmpeg (윈도우는 `choco`로, 맥에서는 `brew` 로 설치)
2. 파이썬 가상환경 세팅(`python3 -m venv .venv` `source .venv/bin/activate` `pip install -r requirements.txt`)
3. frames30_pts 라는 이름의 폴더 (크롭한 계기판을 이 폴더에 저장)

## 어떻게 작동하냐면요...
1. 동영상을 30 fps 단위로 캡쳐해요. (`ffmpeg` 이용)
2. 7-segment 형태의 숫자를 인식할 부분만 크롭해요. (City Car Driving 플레이 화면 기준이라 크롭 위치는 고정값)
#### 1 ~ 2 번 Windows PowreShell
```shell
ffmpeg -y -i ".\video.mp4" -vf "crop=iw*0.013:ih*0.05:iw*0.0235:ih*0.08,fps=30" -frame_pts 1 ".\frames30_pts\img_%010d.png"
```

#### 1 ~ 2 번 MacOS Terminal
```shell
ffmpeg -y -i "./video.mp4" -vf "crop=iw*0.013:ih*0.05:iw*0.0235:ih*0.08,fps=30" -frame_pts 1 "./frames30_pts/img_%010d.png"
```
3. `classify_sevenseg.py` 에서 숫자 부분을 인식해요.
4. 인식한 숫자를 저장해 `.csv` 파일로 내보내요.
#### 3 ~ 4번 다음의 명령어 중 하나를 사용해요
```shell
python classify_sevenseg.py
python3 classify_sevenseg.py
```
5. `export_speed_to_excel.py`에서 내보낸 `.csv` 파일로 `.xlsx` 파일을 만들어요.
#### 5번 다음의 명령어 중 하나를 사용해요
```shell
python export_speed_to_excel.py
python3 export_speed_to_excel.py
```

## 지금도 잘 돌아가지만, 추가로 뭘 더 해야하냐면요...
1. 코드 리팩토링
2. 범용성 고민해보기 - 현재 특정 사이즈의 검정 배경 흰 글자만 인식 가능
