# Azure ML 온습도 예측 및 이상 탐지 프로젝트

## 프로젝트 개요
- **목표 1**: 30분 후 Hot Aisle 온도 예측
- **목표 2**: 비정상 온습도 패턴 이상 탐지
- **데이터**: 컨테인먼트 4개 존 + 랙 12개 (2개월)

---

## VS Code에서 실행 가이드

### 1. 사전 준비

#### 1-1. 환경 변수 설정
```bash
# .env.example을 .env로 복사
cp .env.example .env

# .env 파일을 열어서 Azure 구독 ID 입력
# AZURE_SUBSCRIPTION_ID=your-subscription-id-here
```

#### 1-2. Azure CLI 로그인
```bash
az login
az account set --subscription "your-subscription-id"
```

#### 1-3. Python 패키지 설치
```bash
pip install azure-ai-ml azure-identity pandas scikit-learn matplotlib seaborn joblib python-dotenv
```

#### 1-4. 데이터 파일 준비
다음 파일들을 프로젝트 폴더에 복사:
- `cont_processed.csv` (컨테인먼트 전처리 데이터)
- `rack_processed.csv` (랙 전처리 데이터)

---

### 2. 실행 순서

#### Step 1: Azure 연결 테스트
```bash
python azure_config.py
```
**예상 출력**: ✅ Azure ML Workspace 연결 성공

#### Step 2: 이상 탐지 모델 학습 (로컬)
```bash
python 03_train_anomaly_detector.py
```
**소요 시간**: 2-3분  
**결과물**:
- `models/anomaly_detector_cont.pkl`
- `models/anomaly_detector_rack.pkl`
- `cont_with_anomalies.csv`
- `visualizations/` 폴더

#### Step 3: 예측 모델 준비
```bash
python 02_train_forecast_model.py
```
**결과물**: `cont_forecast_data.csv` (예측용 데이터)

---

### 3. Azure ML Studio에서 AutoML 실행 (권장)

로컬에서 AutoML 실행은 시간이 오래 걸리므로, **Azure ML Studio GUI 사용을 권장**합니다.

#### 3-1. Azure ML Studio 접속
https://ml.azure.com

#### 3-2. 데이터 업로드
1. **Data** → **+ Create**
2. Name: `cont_forecast_data`
3. Upload: `cont_forecast_data.csv`

#### 3-3. AutoML 실험 생성
1. **Automated ML** → **+ New Automated ML job**
2. **데이터셋**: `cont_forecast_data` 선택
3. **Task type**: Forecasting
4. **Target column**: `target_tempHot_30min`
5. **Time column**: `colDate`
6. **Time series ID**: `contID`
7. **Forecast horizon**: 3

#### 3-4. Compute 설정
- **Compute type**: Compute cluster
- **Virtual machine size**: Standard_DS3_v2 (4 cores)
- **Min nodes**: 0, Max nodes**: 4

#### 3-5. 실험 실행
- **Timeout**: 1 hour
- **Max trials**: 10
- **Primary metric**: Normalized RMSE

**소요 시간**: 30-60분

---

### 4. 대시보드 생성 (Power BI 또는 Streamlit)

#### Option A: Streamlit 대시보드 (간단)
```bash
pip install streamlit plotly
streamlit run dashboard.py
```

#### Option B: Power BI 연결
1. Power BI Desktop 설치
2. Azure ML 데이터 연결
3. 시각화 생성

---

## 프로젝트 구조

```
project/
├── azure_config.py              # Azure 연결 설정
├── 01_upload_data.py            # 데이터 업로드 (선택)
├── 02_train_forecast_model.py   # 예측 데이터 준비
├── 03_train_anomaly_detector.py # 이상 탐지 모델 학습
├── cont_processed.csv           # 전처리된 컨테인먼트 데이터
├── rack_processed.csv           # 전처리된 랙 데이터
├── models/                      # 학습된 모델 저장
│   ├── anomaly_detector_cont.pkl
│   ├── anomaly_detector_rack.pkl
│   ├── scaler_cont.pkl
│   └── scaler_rack.pkl
└── visualizations/              # 시각화 결과
    ├── containment_anomalies_timeseries.png
    └── anomaly_score_analysis.png
```

---

## 주요 결과

### 이상 탐지 결과 (예상)
- **존4**: 이상치 비율 가장 높음 (냉각 효율 문제)
- **존2, 3**: 정상 범위
- **존1**: 중간 수준

### 예측 모델 성능 목표
- **RMSE**: < 0.5°C
- **MAE**: < 0.3°C

---

## 다음 단계

1. **API 배포** (선택): Azure ML 엔드포인트로 실시간 예측
2. **재학습 파이프라인**: Azure DevOps로 자동화
3. **알림 시스템**: Logic Apps로 이상 탐지 시 알림

---

## 트러블슈팅

### 문제: Azure 인증 실패
```bash
az login --use-device-code
```

### 문제: Compute 할당량 부족
Azure Portal → Quotas → ML Compute 요청

### 문제: 패키지 설치 오류
```bash
pip install --upgrade azure-ai-ml
```

---

## 참고 자료
- Azure ML Documentation: https://learn.microsoft.com/azure/machine-learning/
- Isolation Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html


더 간단한 방법: GUI로 직접 진행
SDK 버전 문제가 계속되면, Azure ML Studio GUI로 바로 진행하는 게 더 빠릅니다:
1. Azure ML Studio 접속
https://ml.azure.com
2. 데이터 업로드

좌측 메뉴: Data → + Create
Name: cont_forecast_data
Type: File (URI_FILE)
Upload from: Local files
파일 선택: cont_forecast_data.csv (방금 생성됨)

3. AutoML Job 생성

좌측 메뉴: Automated ML → + New Automated ML job
Select data asset: cont_forecast_data
Task type: Forecasting
Target column: target_tempHot_30min
View additional configuration settings:

Time column: colDate
Time series identifier columns: contID
Forecast horizon: 3



4. Compute 생성

+ New compute cluster
VM size: Standard_DS3_v2
Min/Max nodes: 0 / 4