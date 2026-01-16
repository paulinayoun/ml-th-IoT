# Azure AutoML 학습 전략 계획 (상세)

## 📊 전체 데이터 현황

### 1. Spring 데이터 (4-5월)
- **기간**: 2025-04-10 ~ 2025-05-19 (39일, 약 5,624개 측정/존)
- **컨테인먼트**: 4개 (contID 1,2,3,4)
  - contID 1: 5,625행 (완전)
  - contID 2: 5,622행 (완전)
  - contID 3: 5,624행 (완전)
  - contID 4: 5,624행 (⚠️ 공사 중 - 불완전 데이터, 약 30% 결측)
- **랙**: 52개 (contID별 12, 12, 12, 16개)
- **특징**:
  - 봄철 (평균 온도: 24-28°C)
  - 서버 미거치 (rack_count 변동)
  - contID 4 센서 불안정
  - 10분 간격 측정

### 2. Summer 데이터 (7-9월)
- **기간**: 2025-07-25 ~ 2025-09-25 (62일, 약 8,902개 측정/존)
- **컨테인먼트**: 4개 (contID 1,2,3,4)
  - 각 contID: 약 8,900행 (완전 데이터, 결측 < 0.1%)
- **랙**: 전체 거치 완료 (각 contID: 12개)
- **특징**:
  - 혹서기 (평균 온도: 30-32°C, 피크: 34°C)
  - 완전한 운영 데이터
  - 안정적인 rack_count (고정값)
  - 10분 간격 측정

### 3. 데이터 품질 비교

| 항목 | Spring (4-5월) | Summer (7-9월) |
|------|----------------|----------------|
| 데이터 완전성 | 70% (contID 4 문제) | 99.9% |
| rack_count | 변동 (0-12) | 고정 (12) |
| 센서 안정성 | 불안정 | 안정 |
| 평균 온도 | 26°C | 31°C |
| 온도 범위 | 22-30°C | 28-34°C |
| 사용 권장 | ⚠️ 주의 | ✅ 우수 |

---

## 🎯 학습 전략 옵션 (상세)

### 옵션 1: 7-8월로 학습 → 9월 예측 ⭐ **추천**

#### 📋 목적
혹서기 패턴 학습 후 진정한 out-of-sample 검증

#### ✅ 장점
1. **데이터 품질**: 99.9% 완전한 데이터
2. **진정한 검증**: 9월 데이터는 학습에 미사용
3. **혹서기 패턴**: 여름 더위 패턴 집중 학습
4. **실용성**: 복잡한 전처리 불필요
5. **비교 가능**: 기존 모델(7-9월)과 성능 비교

#### ⚠️ 단점
1. 4-5월 봄철 패턴 미학습
2. 계절 전환 패턴 학습 불가
3. 상대적으로 적은 데이터량 (37일)

#### 📊 학습 데이터 분할
```
전체 데이터: 2025-07-25 ~ 2025-09-25 (62일)
├─ Train: 2025-07-25 ~ 2025-08-31 (38일)
│  └─ 약 14,800 rows (4 zones × 3,700 rows)
└─ Test:  2025-09-01 ~ 2025-09-25 (25일)
   └─ 약 9,600 rows (4 zones × 2,400 rows)

비율: Train 60% / Test 40%
```

#### 🔧 상세 실행 단계

**Step 1: 데이터 추출 및 전처리**
```python
import pandas as pd

# 데이터 로드
df = pd.read_csv('cont_forecast_clean/data.csv', parse_dates=['colDate'])

# 7-8월 데이터만 추출
train_df = df[
    (df['colDate'] >= '2025-07-25') &
    (df['colDate'] < '2025-09-01')
].copy()

# 검증: 데이터 품질 확인
print(f"Train 데이터: {len(train_df)} 행")
print(f"날짜 범위: {train_df['colDate'].min()} ~ {train_df['colDate'].max()}")
print(f"결측치: {train_df.isnull().sum().sum()}")
print(f"contID별 행수:\n{train_df.groupby('contID').size()}")

# MLTable 폴더 생성
train_df.to_csv('cont_forecast_train_jul_aug/data.csv', index=False)
```

**Step 2: MLTable 파일 생성**
```yaml
# cont_forecast_train_jul_aug/MLTable
$schema: https://azuremlschemas.azureedge.net/latest/MLTable.schema.json

type: mltable
paths:
  - file: ./data.csv

transformations:
  - read_delimited:
      delimiter: ','
      encoding: utf8
      header: all_files_same_headers
```

**Step 3: Azure AutoML 설정**
```python
from azure.ai.ml import automl, Input
from azure.ai.ml.constants import AssetTypes
from azure_config import get_ml_client

ml_client = get_ml_client()

# 학습 데이터 입력
training_data = Input(
    type=AssetTypes.MLTABLE,
    path="./cont_forecast_train_jul_aug"
)

# Forecasting Job 설정
forecasting_job = automl.forecasting(
    compute="cpu-cluster",
    experiment_name="forecast-jul-aug-sep-validation",
    training_data=training_data,
    target_column_name="target_tempHot_30min",
    primary_metric="normalized_root_mean_squared_error",
    n_cross_validations=5,
    enable_model_explainability=True,

    # Time Series 설정
    time_column_name="colDate",
    time_series_id_column_names=["contID"],
    forecast_horizon=2,  # 30분 (10분 × 3 step)
    frequency="10min",

    # Feature 설정
    blocked_transformers=["LabelEncoder"],  # 불필요한 변환 차단
)

# 학습 제한 설정
forecasting_job.set_limits(
    timeout_minutes=120,           # 전체 타임아웃: 2시간
    trial_timeout_minutes=30,      # 단일 모델 타임아웃: 30분
    max_trials=15,                 # 최대 15개 모델 시도
    max_concurrent_trials=4,       # 동시 실행: 4개
    enable_early_termination=True, # 조기 종료 활성화
)

# Job 제출
returned_job = ml_client.jobs.create_or_update(forecasting_job)
print(f"Job 생성됨: {returned_job.name}")
print(f"Studio URL: {returned_job.studio_url}")
```

**Step 4: 학습 완료 후 모델 다운로드**
```python
# Best model 다운로드
ml_client.jobs.download(
    name=returned_job.name,
    download_path="./models_jul_aug",
    output_name="mlflow-model"
)
```

**Step 5: 9월 데이터로 예측**
```python
import joblib

# 모델 로드
model = joblib.load('models_jul_aug/model.pkl')

# 9월 데이터 로드
test_df = df[df['colDate'] >= '2025-09-01'].copy()

# 학습 데이터 마지막 시점부터 예측
last_train_date = pd.to_datetime('2025-08-31 23:50:00')
forecast_dest = last_train_date + pd.Timedelta(minutes=10)

# 예측 실행
predictions, pred_df = model.forecast(
    X_pred=None,
    y_pred=None,
    forecast_destination=test_df['colDate'].max()
)
```

**Step 6: 성능 평가**
```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 실제값 vs 예측값
actual = test_df['target_tempHot_30min'].values
predicted = predictions[:len(actual)]

# 평가 지표 계산
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

print(f"MAE:  {mae:.4f}°C")
print(f"RMSE: {rmse:.4f}°C")
print(f"MAPE: {mape:.2f}%")
```

#### 💰 예상 비용 및 시간
- **학습 시간**: 1.5 ~ 2.5시간
- **Azure 비용**: $5 ~ $10 (cpu-cluster 기준)
- **총 소요 시간**: 약 3시간 (데이터 준비 + 학습 + 평가)

#### 📈 예상 성능
```
목표 지표:
  MAE:  < 0.8°C  (우수)
  RMSE: < 1.2°C  (우수)
  MAPE: < 2.5%   (우수)

실제 예상:
  MAE:  0.6 ~ 1.0°C
  RMSE: 0.8 ~ 1.5°C
  MAPE: 2.0 ~ 3.5%
```

---

### 옵션 2: 4-5월 + 7-8월로 학습 → 9월 예측

#### 📋 목적
계절 변화 패턴 포함, 데이터량 증대

#### ✅ 장점
1. **더 많은 데이터**: 76일 (옵션1의 2배)
2. **계절 패턴**: 봄→여름 전환 학습
3. **모델 일반화**: 다양한 온도 범위 경험
4. **장기 트렌드**: 계절적 변화 패턴 포착

#### ⚠️ 단점 및 도전 과제
1. **contID 4 불완전**: 약 30% 데이터 결측
2. **rack_count 변동**: 4-5월 (0-12), 7-8월 (12 고정)
3. **데이터 공백**: 5월 19일 ~ 7월 25일 (67일 간격)
4. **온도 분포 차이**: 봄(24-28°C) vs 여름(30-32°C)
5. **복잡한 전처리**: 정규화, 보간 필요

#### 📊 학습 데이터 분할
```
Spring:  2025-04-10 ~ 2025-05-19 (39일)
  └─ 약 22,500 rows (4 zones × 5,625 rows)

Summer:  2025-07-25 ~ 2025-08-31 (38일)
  └─ 약 14,800 rows (4 zones × 3,700 rows)

Train 총계: 77일, 약 37,300 rows
Test:       2025-09-01 ~ 2025-09-25 (25일)
  └─ 약 9,600 rows
```

#### 🔧 상세 실행 단계

**Step 1: Spring 데이터 전처리**
```python
# Spring 데이터 로드
spring_cont = pd.read_csv('data/spring/tbContTHRaw_202601151344.csv',
                          parse_dates=['colDate'])
spring_rack = pd.read_csv('data/spring/tbContRack_202601151338.csv')

# contID 4 처리 방안 A: 제외
spring_cont = spring_cont[spring_cont['contID'].isin([1, 2, 3])].copy()

# 또는 방안 B: 결측치 보간
# spring_cont = spring_cont.groupby('contID').apply(
#     lambda x: x.interpolate(method='time', limit=10)
# )

# rack_count 추가
rack_counts = spring_rack.groupby('contID').size().to_dict()
spring_cont['rack_count'] = spring_cont['contID'].map(rack_counts)

# Feature 생성 (Summer와 동일하게)
spring_cont['hour'] = spring_cont['colDate'].dt.hour
spring_cont['day_of_week'] = spring_cont['colDate'].dt.dayofweek
spring_cont['temp_diff'] = spring_cont['tempHot'] - spring_cont['tempCold']
spring_cont['humi_diff'] = spring_cont['humiHot'] - spring_cont['humiCold']

# target 생성 (30분 = 3 step)
spring_cont = spring_cont.sort_values(['contID', 'colDate'])
spring_cont['target_tempHot_30min'] = spring_cont.groupby('contID')['tempHot'].shift(-3)

# 결측치 제거
spring_cont = spring_cont.dropna(subset=['target_tempHot_30min'])

print(f"Spring 처리 완료: {len(spring_cont)} 행")
```

**Step 2: 데이터 결합 및 정규화**
```python
# Summer 데이터 로드
summer = pd.read_csv('cont_forecast_clean/data.csv', parse_dates=['colDate'])
summer_train = summer[summer['colDate'] < '2025-09-01'].copy()

# 계절 feature 추가
spring_cont['season'] = 'spring'
summer_train['season'] = 'summer'

# 데이터 결합
combined = pd.concat([spring_cont, summer_train], ignore_index=True)
combined = combined.sort_values(['contID', 'colDate']).reset_index(drop=True)

# rack_count 정규화 (0-1 범위로)
combined['rack_count_normalized'] = combined['rack_count'] / 12.0

# 온도 정규화 (선택사항)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# combined[['tempHot', 'tempCold']] = scaler.fit_transform(
#     combined[['tempHot', 'tempCold']]
# )

print(f"Combined 데이터: {len(combined)} 행")
print(f"날짜 범위: {combined['colDate'].min()} ~ {combined['colDate'].max()}")
```

**Step 3: 데이터 공백 처리**
```python
# 옵션 A: 그대로 사용 (AutoML이 자동 처리)
# 옵션 B: 명시적 표시
combined['days_since_start'] = (
    combined['colDate'] - combined['colDate'].min()
).dt.total_seconds() / 86400

# 옵션 C: 계절 인덱스 추가
combined['is_spring'] = (combined['season'] == 'spring').astype(int)
combined['is_summer'] = (combined['season'] == 'summer').astype(int)
```

**Step 4: MLTable 생성 및 Azure AutoML 실행**
```python
# 최종 컬럼 정리
final_columns = [
    'contID', 'colDate', 'tempHot', 'tempCold',
    'humiHot', 'humiCold', 'temp_diff', 'humi_diff',
    'hour', 'day_of_week', 'rack_count',
    'target_tempHot_30min'
]

combined_final = combined[final_columns].copy()
combined_final.to_csv('cont_forecast_train_spring_summer/data.csv', index=False)

# Azure AutoML Job 설정 (옵션 1과 동일, experiment_name만 변경)
forecasting_job = automl.forecasting(
    compute="cpu-cluster",
    experiment_name="forecast-spring-summer-sep-validation",  # 변경
    training_data=Input(
        type=AssetTypes.MLTABLE,
        path="./cont_forecast_train_spring_summer"
    ),
    # ... 나머지 설정 동일
)
```

#### 💰 예상 비용 및 시간
- **전처리 시간**: 30분
- **학습 시간**: 2 ~ 3시간 (데이터량 2배)
- **Azure 비용**: $8 ~ $15
- **총 소요 시간**: 약 4시간

#### 📈 예상 성능
```
기대 시나리오:
  MAE:  < 0.7°C  (더 우수)
  RMSE: < 1.0°C  (더 우수)
  MAPE: < 2.0%   (더 우수)

현실 시나리오:
  MAE:  0.7 ~ 1.2°C
  RMSE: 1.0 ~ 1.8°C
  MAPE: 2.5 ~ 4.0%

⚠️ 주의: 데이터 불일치로 인해 옵션1보다
        성능이 낮을 가능성 있음
```

---

## 💡 최종 추천 및 근거

### ⭐ **추천: 옵션 1 (7-8월 학습 → 9월 예측)**

#### 추천 근거
1. **데이터 품질 우선**: 완전한 데이터 > 많은 데이터
2. **실용성**: 복잡한 전처리 없이 빠른 실행
3. **검증 신뢰성**: 명확한 out-of-sample 검증
4. **비용 효율**: 시간/비용 최소화
5. **리스크 최소**: 데이터 불일치 문제 회피

#### 의사결정 프레임워크
```
데이터 품질  [■■■■■ 5/5]  옵션 1 우세
학습 시간    [■■■■■ 5/5]  옵션 1 우세
비용         [■■■■■ 5/5]  옵션 1 우세
성능 잠재력  [■■■□□ 3/5]  옵션 2 우세
일반화 능력  [■■■□□ 3/5]  옵션 2 우세
복잡도       [■■■■■ 5/5]  옵션 1 우세 (낮음)

종합 점수:
  옵션 1: 26/30 ⭐⭐⭐⭐⭐
  옵션 2: 21/30 ⭐⭐⭐⭐
```

---

## 📋 상세 실행 체크리스트

### Phase 1: 준비 (30분)
- [ ] Azure ML Studio 접속 확인
- [ ] Compute Cluster 상태 확인 (cpu-cluster)
- [ ] cont_forecast_clean/data.csv 존재 확인
- [ ] Python 환경 확인 (azure-ai-ml 설치)
- [ ] azure_config.py 설정 확인

### Phase 2: 데이터 준비 (1시간)
- [ ] 7-8월 데이터 추출 스크립트 실행
- [ ] 데이터 품질 검증
  - [ ] 결측치 확인 (< 0.1%)
  - [ ] contID 4개 모두 존재
  - [ ] 날짜 범위 정확 (7/25 ~ 8/31)
  - [ ] 행 수 확인 (~14,800)
- [ ] MLTable 폴더 생성
- [ ] MLTable 파일 작성
- [ ] 로컬에서 MLTable 로드 테스트

### Phase 3: Azure AutoML 실행 (2-3시간)
- [ ] Forecasting Job 설정 코드 작성
- [ ] Job 제출
- [ ] Studio URL에서 진행 상황 모니터링
- [ ] 각 모델 시도 결과 확인
- [ ] Best model 선택 확인

### Phase 4: 모델 다운로드 (15분)
- [ ] Best model 다운로드
- [ ] models_jul_aug/ 폴더 확인
- [ ] model.pkl 파일 존재 확인
- [ ] 로컬에서 모델 로드 테스트

### Phase 5: 9월 데이터 예측 (30분)
- [ ] 9월 데이터 추출
- [ ] 예측 스크립트 작성
- [ ] 예측 실행
- [ ] 결과 저장 (CSV)

### Phase 6: 성능 평가 (1시간)
- [ ] MAE, RMSE, MAPE 계산
- [ ] 시각화 (실제 vs 예측)
- [ ] 존별 성능 분석
- [ ] 시간대별 오차 분석
- [ ] 리포트 작성

---

## 📊 평가 지표 상세 설명

### 1. MAE (Mean Absolute Error)
```
MAE = (1/n) × Σ|실제값 - 예측값|

해석:
  < 0.5°C : 매우 우수
  < 1.0°C : 우수
  < 1.5°C : 양호
  > 1.5°C : 개선 필요

예: MAE = 0.6°C
  → 평균적으로 예측이 실제값과 0.6°C 차이
```

### 2. RMSE (Root Mean Squared Error)
```
RMSE = √[(1/n) × Σ(실제값 - 예측값)²]

해석:
  < 0.8°C : 매우 우수
  < 1.2°C : 우수
  < 2.0°C : 양호
  > 2.0°C : 개선 필요

특징: 큰 오차에 더 민감
```

### 3. MAPE (Mean Absolute Percentage Error)
```
MAPE = (100/n) × Σ|(실제값 - 예측값) / 실제값|

해석:
  < 2% : 매우 우수
  < 5% : 우수
  < 10% : 양호
  > 10% : 개선 필요

예: MAPE = 2.5%
  → 평균적으로 2.5%의 오차율
```

### 4. 존별 성능 비교
```python
# 존별 MAE 계산
for zone in [1, 2, 3, 4]:
    zone_data = results[results['contID'] == zone]
    mae = np.abs(zone_data['actual'] - zone_data['predicted']).mean()
    print(f"Zone {zone} MAE: {mae:.4f}°C")
```

---

## ⚠️ 리스크 및 대응 방안

### 리스크 1: Azure AutoML 학습 실패
**원인**:
- Compute cluster 문제
- 데이터 형식 오류
- 타임아웃

**대응**:
1. Compute 재시작
2. MLTable 형식 재확인
3. timeout 시간 연장 (120 → 180분)

### 리스크 2: 예측 정확도 낮음 (MAE > 2.0°C)
**원인**:
- 7-8월과 9월 패턴 차이
- Feature 부족
- 모델 underfitting

**대응**:
1. Feature 추가 (외부 온도, 전력)
2. 학습 데이터 기간 조정
3. Hyperparameter 튜닝
4. 옵션 2 시도 (Spring 데이터 추가)

### 리스크 3: 특정 존 성능 저하
**원인**:
- 존별 센서 특성 차이
- 물리적 위치 영향

**대응**:
1. 존별 개별 모델 학습
2. Ensemble 모델
3. 물리적 요인 조사

### 리스크 4: 9월 중순 이후 정확도 급락
**원인**:
- 계절 전환 (가을 시작)
- 학습 데이터 범위 벗어남

**대응**:
1. 단기 예측으로 제한 (7일)
2. 주기적 재학습 (Weekly)
3. Adaptive 모델 도입

---

## 🗓️ 타임라인

### Day 1 (오늘)
- [x] 데이터 현황 파악
- [x] 학습 전략 수립
- [ ] 옵션 선택 및 의사결정
- [ ] 데이터 추출 스크립트 작성

### Day 2
- [ ] MLTable 생성
- [ ] Azure AutoML Job 설정 및 제출
- [ ] 학습 모니터링

### Day 3
- [ ] 모델 다운로드
- [ ] 예측 스크립트 작성 및 실행
- [ ] 성능 평가

### Day 4
- [ ] 결과 시각화
- [ ] 리포트 작성
- [ ] 대시보드 통합

---

## 📞 의사결정 필요 사항

### 🔴 필수 결정
1. **학습 전략 선택**:
   - [ ] 옵션 1: 7-8월 → 9월 (추천)
   - [ ] 옵션 2: 4-5월+7-8월 → 9월

2. **contID 4 처리** (옵션 2 선택 시):
   - [ ] A) 제외 (contID 1,2,3만)
   - [ ] B) 포함 (보간 처리)

### 🟡 선택 사항
3. **추가 Feature**:
   - [ ] 외부 온도 데이터 (날씨 API)
   - [ ] 전력 데이터 (tbPowerRaw)
   - [ ] 랙별 상세 데이터 (tbRackTHRaw)

4. **평가 방법**:
   - [ ] 단순 비교 (MAE, RMSE)
   - [ ] 시계열 분해 분석
   - [ ] 존별 개별 평가

---

## 🚀 즉시 실행 가능 명령어

### 옵션 1 선택 시
```bash
# 1. 데이터 추출
python scripts/prepare_jul_aug_data.py

# 2. Azure AutoML 실행
python scripts/train_jul_aug_model.py

# 3. 예측 및 평가
python scripts/evaluate_sep_predictions.py
```

### 옵션 2 선택 시
```bash
# 1. Spring 데이터 전처리
python scripts/preprocess_spring_data.py

# 2. 데이터 결합
python scripts/combine_spring_summer.py

# 3. Azure AutoML 실행
python scripts/train_combined_model.py

# 4. 예측 및 평가
python scripts/evaluate_sep_predictions.py
```

---

## 💬 다음 단계

**지금 결정해주세요:**

1. ✅ **옵션 1** (7-8월 → 9월) 또는 **옵션 2** (4-5월+7-8월 → 9월)?

2. 📅 **시작 시점**: 지금 바로 vs 나중에?

3. 📊 **대시보드**: 모델 학습 먼저 vs 대시보드 개선 먼저?

결정해주시면 즉시 작업 시작하겠습니다! 🎯
