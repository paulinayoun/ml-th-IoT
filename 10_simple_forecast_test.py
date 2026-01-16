# -*- coding: utf-8 -*-
"""
Azure AutoML 모델 간단한 예측 테스트
04_run_local_prediction.py 방식 참고
"""
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = "models/model.pkl"
DATA_PATH = "./cont_forecast_clean/data.csv"  # 모델이 학습한 원본 데이터로 테스트

print("="*60)
print("간단한 예측 테스트")
print("="*60)

# 1. 모델 로드
print("\n[1] 모델 로드")
model = joblib.load(MODEL_PATH)
print(f"  [OK] {type(model)}")

# 2. 데이터 로드
print("\n[2] 데이터 로드")
df = pd.read_csv(DATA_PATH, parse_dates=['colDate'])
print(f"  전체: {len(df)} 행")
print(f"  날짜: {df['colDate'].min()} ~ {df['colDate'].max()}")

# 3. 단일 contID만 필터링 (04_run_local_prediction.py 방식)
zone_id = 1
print(f"\n[3] contID={zone_id} 필터링")
zone_df = df[df['contID'] == zone_id].copy()
print(f"  {len(zone_df)} 행")

# 4. target 컬럼 제거
print("\n[4] 예측용 데이터 준비")
X_test = zone_df.drop(columns=['target_tempHot_30min'])
print(f"  Shape: {X_test.shape}")
print(f"  Columns: {list(X_test.columns)}")

# 5. 예측 실행 (04_run_local_prediction.py 방식)
print("\n[5] 예측 실행 (forecast 메서드)")
try:
    # y_pred 없이 X_test만 전달
    y_predictions, X_trans = model.forecast(X_test)

    print(f"  [OK] 예측 성공!")
    print(f"  y_predictions shape: {y_predictions.shape if hasattr(y_predictions, 'shape') else len(y_predictions)}")
    print(f"  X_trans shape: {X_trans.shape}")
    print(f"  X_trans columns: {list(X_trans.columns)}")

    # 결과 확인
    print("\n[6] 예측 결과 (처음 10개)")
    print(y_predictions[:10])

except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
