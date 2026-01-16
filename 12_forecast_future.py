# -*- coding: utf-8 -*-
"""
forecast_destination을 사용하여 학습 데이터 끝 이후 예측
"""
import pandas as pd
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = "models/model.pkl"
DATA_PATH = "./cont_forecast_clean/data.csv"

print("="*60)
print("forecast_destination 사용 테스트")
print("="*60)

# 1. 모델 로드
print("\n[1] 모델 로드")
model = joblib.load(MODEL_PATH)
print(f"  [OK] {type(model)}")
print(f"  forecast_horizon: {model.max_horizon if hasattr(model, 'max_horizon') else 'N/A'}")

# 2. 데이터 로드
print("\n[2] 데이터 로드")
df = pd.read_csv(DATA_PATH, parse_dates=['colDate'])
print(f"  전체: {len(df)} 행")
print(f"  날짜: {df['colDate'].min()} ~ {df['colDate'].max()}")

# 3. 모델 학습 데이터의 마지막 시점 확인
print("\n[3] 모델 학습 데이터")
print(f"  마지막 시점: {df['colDate'].max()}")

# 4. forecast_destination 지정 (학습 마지막 시점 이후)
# 2025-09-25 09:30:00 + 30분 = 2025-09-25 10:00:00
forecast_dest = df['colDate'].max() + pd.Timedelta(minutes=30)
print(f"\n[4] 예측 목표 시점")
print(f"  forecast_destination: {forecast_dest}")
print(f"  (학습 데이터 마지막 +30분)")

print(f"\n[5] 예측 실행 (forecast_destination만 사용)")

try:
    # forecast_destination만 사용 (X_pred 없이)
    y_predictions, X_trans = model.forecast(
        X_pred=None,
        y_pred=None,
        forecast_destination=forecast_dest
    )

    print(f"  [OK] 예측 성공!")
    print(f"  예측 결과 수: {len(y_predictions)}")
    print(f"  X_trans shape: {X_trans.shape}")
    print(f"  X_trans columns: {list(X_trans.columns)}")

    # 6. 결과 확인
    print("\n[6] 예측 결과")
    print(X_trans.head(10))
    print(f"\n예측값 샘플:")
    print(y_predictions[:10])

    # 7. 실제 8월 1일 00:00:00 ~ 00:30:00 데이터와 비교
    print("\n[7] 실제값과 비교")
    actual_future = df[df['colDate'] >= train_end].head(10)
    print(actual_future[['colDate', 'contID', 'tempHot']])

except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
