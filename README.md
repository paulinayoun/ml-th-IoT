# 🌡️ IoT 온습도 예측 및 이상 탐지 프로젝트

Azure ML을 활용한 데이터센터 컨테인먼트 온습도 시계열 예측 및 이상 탐지 시스템

## 📋 프로젝트 구조

```
ml-th-IoT/
├── .env                          # Azure 설정
├── requirements.txt              # Python 패키지
│
├── data/                         # 원본 데이터
│   ├── cont_processed.csv
│   └── rack_processed.csv
│
├── cont_forecast_clean/          # Azure 업로드용
│   ├── MLTable
│   └── data.csv                  # 23,804행, 15분 간격
│
├── models/                       # 학습된 모델
├── visualizations/               # 시각화 결과
│
├── pages/                        # Streamlit 페이지
│   ├── 1_🌡️_Forecast_Dashboard.py
│   └── 2_🚨_Anomaly_Dashboard.py
│
├── azure_config.py               # Azure ML 연결
├── clean_data.py                 # 데이터 정제
├── 02_train_forecast_model.py    # AutoML 예측
├── 03_train_anomaly_detector.py  # 이상 탐지
├── 04_run_local_prediction.py    # 로컬 예측
└── main_dashboard.py             # 대시보드
```

## 🚀 시작하기

### 1. 환경 설정
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 데이터 준비
```bash
python clean_data.py
```

### 3. AutoML 학습 (Azure ML Studio)
- Data: cont_forecast_clean_15min (MLTable)
- Target: target_tempHot_30min
- Time column: colDate
- Time series ID: contID
- Frequency: 15min
- Forecast horizon: 2

### 4. 이상 탐지 모델 학습
```bash
python 03_train_anomaly_detector.py
```

### 5. 대시보드 실행
```bash
streamlit run main_dashboard.py
```

## 📊 주요 기능

- ✅ 15분 간격 시계열 데이터 정제
- ✅ 중복 제거 및 연속성 확보
- ✅ Azure AutoML 예측 (30분 후 온도)
- ✅ Isolation Forest 이상 탐지
- ✅ Streamlit 대시보드

## 🎯 데이터 특징

- 총 23,804행
- 4개 zone (zone_1 ~ zone_4)
- 각 zone당 5,951개 시점
- 15분 간격
- 중복 0개, 빠진 시간대 0개
