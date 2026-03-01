# Data Inspector (데이터 인스펙터)

공정 데이터(로봇 경로, 오디오, 열화상, RGB 영상 등)를 시각화하고 이상 징후를 라벨링하여 종합 리포트를 생성하는 대화형 대시보드 도구입니다.

---

## 1. 실행 방법

본 프로젝트는 Python 3.11 환경에서 최적화되어 있습니다.

### 설치 및 환경 설정
```powershell
# 가상환경 생성 및 활성화
python -m venv .venv
.\.venv\Scripts\activate

# 의존성 패키지 설치
pip install -r requirements.txt
```

### 앱 실행
기본적으로 `DB` 폴더 내의 세션들을 자동으로 탐색합니다.
```powershell
# 기본 실행
python main.py

# 디버그 모드 및 특정 세션 지정 실행
python main.py --db-root DB --session 260227_155409 --debug
```
실행 후 브라우저에서 `http://127.0.0.1:8050`에 접속하여 대시보드를 확인하세요.

---

## 2. 데이터베이스(DB) 구조 및 사양

이 도구는 특정 폴더 구조를 기반으로 데이터를 로드합니다. 각 세션은 `YYMMDD_HHMMSS` 형식의 폴더명을 가져야 합니다.

### 권장 폴더 구조
```text
DB/
└── 210101_123456/ (세션 폴더)
    ├── Robot/
    │   └── 0.jsonl      # 로봇 경로(XYZ), 관절 각도, ArcOn 상태 등
    ├── Model/
    │   └── 0.jsonl      # 모델 경로(XYZ), LaserOn 상태, 레이어 정보
    ├── Audio/
    │   └── 0.wav        # 공정 중 녹음된 오디오 파일
    ├── IR_High/
    │   ├── 0.jsonl      # 열화상 메타데이터 (MaxTemp, PeakX, PeakY, Contour 등)
    │   └── 0.raw        # 열화상 RAW 프레임 데이터 (1100x2000 권장)
    ├── IR_Low/
    │   └── 0.raw        # 저해상도 열화상 RAW 데이터
    └── Image/
        └── *.jpg        # 공정 모니터링 RGB 프레임 이미지들
```

### 주요 데이터 소스 설명
- **Robot & Model (`.jsonl`)**: 시각화의 타임라인 기준이 되며, 3D 경로 플롯의 소스로 사용됩니다.
- **IR_High (`.raw` & `.jsonl`)**: 용융지(Melt Pool)의 온도 분포와 형상을 분석하기 위한 고해상도 데이터입니다.
- **Audio (`.wav`)**: 공정 중 발생하는 소리의 에너지(dB) 및 스펙트로그램을 생성합니다.

---

## 3. 주요 기능 및 결과물

### 대화형 시각화
- 3D XYZ 경로의 특정 지점을 클릭하여 해당 시점의 모든 센서 데이터로 즉시 이동할 수 있습니다.
- 모든 그래프(오디오, 관절, 온도, 영상)가 타임라인 슬라이더와 실시간으로 동기화됩니다.

### 라벨링 및 리포트 생성
1. **영역 지정**: `Set Time 1/2` 버튼을 사용하여 이상 징후 구간을 선택합니다.
2. **라벨링**: `Anomaly type` 선택 및 `Notes` 입력 후 `Add Label` 버튼을 눌러 저장합니다. (메모는 자동으로 캐시되어 재사용 가능합니다.)
3. **리포트 내보내기**: `Export Labels`를 누르면 해당 세션 폴더 안에 `session_report.json`이 생성됩니다.

### 최종 결과물: `session_report.json`
추출된 리포트는 다음과 같은 세분화된 구조를 가집니다:
- **overall_process**: 전체 공정의 시작(First Arc On)과 끝(Last Arc Off) 시간.
- **components**: 레이저 작동 신호를 기반으로 자동 감지된 개별 컴포넌트 구간 (Normal로 자동 표기).
- **anomaly_annotations**: 사용자가 직접 태깅한 이상 징후 구간 정보.

---
문의 사항이나 데이터 규격 변경이 필요한 경우 개발팀에 연락주세요.
