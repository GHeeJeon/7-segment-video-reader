# 7-segment-video-reader
### 저도 영상 인식 처리는 처음 해보는데요...  

[City Car Driving - 시뮬레이션 게임](https://store.steampowered.com/app/493490/City_Car_Driving/?l=koreana) 
<details>
<summary> 위 시뮬레이션 플레이 영상을 기반으로  </summary>
<div markdown="1">  

</br>
  
<img width="3360" height="2100" alt="490101310-a9d8fe30-939d-44aa-9070-4f70b6636482" src="https://github.com/user-attachments/assets/842c1c83-d408-44dd-803d-63e7a46efb42" />

</div>
</details>


<details>
<summary> 위치가 고정된 속력 UI(좌측 상단) 를 30fps 단위로 크롭해서 </summary>
<div markdown="1">

</br>
  
<img width="2516" height="1532" alt="490106400-cd0db321-2237-49b6-88d7-8d8fcbce5ed3" src="https://github.com/user-attachments/assets/96df607a-2f82-49af-9cb4-4e7e067f7e94" />

</div>
</details>

<details>
<summary> 크롭한 이미지 속 속력, 7-segment 형태의 숫자를 인식하고 </summary>
<div markdown="1">

</br>
  
<img width="2512" height="1516" alt="490106936-96327d78-621b-427b-81b5-ee9a113dcef9" src="https://github.com/user-attachments/assets/cc15ba2f-4b8a-4837-8010-de4a094806b8" />

</div>
</details>

<details>
<summary> 자동차의 속력과 통계 데이터를 .xlsx 파일로 저장하는 프로그램입니다. </summary>
<div markdown="1">

</br>
  
<img width="3136" height="1474" alt="490119725-d0335327-1fe9-49a1-9ec1-89ec7a05c588" src="https://github.com/user-attachments/assets/4c1d2808-e8ee-48df-8fbb-f36ffbf33d8d" />

</div>
</details>


## 어떤 게 필요하냐면요...
1. ffmpeg (윈도우는 `choco`로, 맥에서는 `brew` 로 설치)
2. `frames30_pts` 라는 이름의 빈 폴더 (크롭한 이미지를 이 폴더에 저장)
3. 파이썬 가상환경 세팅  
For Mac : `python3 -m venv .venv` `source .venv/bin/activate` `pip install -r requirements.txt`  
For Windows : `python -m venv .venv` `.venv\Scripts\Activate.ps1` `pip install -r requirements.txt`  

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
#### 5번 다음의 명령어 중 하나를 사용해요!
```shell
python export_speed_to_excel.py
python3 export_speed_to_excel.py
```

## 지금도 잘 돌아가지만, 추가로 뭘 더 해야하냐면요...
1. 코드 리팩토링
2. 범용성 고민해보기 - 현재 특정 사이즈의 검정 배경 흰 글자만 인식 가능
