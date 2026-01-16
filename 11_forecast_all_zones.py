# -*- coding: utf-8 -*-
"""
전체 contID로 예측 후 특정 zone 추출
"""
import pandas as pd
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = "models/model.pkl"
DATA_PATH = "./cont_forecast_clean/data.csv"

print("="*60)
print("전체 contID 예측 테스트")
print("="*60)

# 1. 모델 로드
print("\n[1] 모델 로드")
model = joblib.load(MODEL_PATH)
print(f"  [OK] {type(model)}")

# 2. 전체 데이터 로드 (모든 contID 포함)
print("\n[2] 전체 데이터 로드")
df = pd.read_csv(DATA_PATH, parse_dates=['colDate'])
print(f"  전체: {len(df)} 행")
print(f"  날짜: {df['colDate'].min()} ~ {df['colDate'].max()}")
print(f"  contID: {df['contID'].unique()}")

# 3. Train/Test 분할 (7월 vs 8월 초)
print("\n[3] Train/Test 분할")
train_end = pd.to_datetime('2025-08-01')
train_df = df[df['colDate'] < train_end].copy()
test_df = df[df['colDate'] >= train_end].head(400).copy()  # 8월 초 일부만

print(f"  Train: {len(train_df)} 행 ({train_df['colDate'].min()} ~ {train_df['colDate'].max()})")
print(f"  Test:  {len(test_df)} 행 ({test_df['colDate'].min()} ~ {test_df['colDate'].max()})")

# 4. target 제거 및 예측
print("\n[4] 예측 실행 (전체 contID)")
X_test = test_df.drop(columns=['target_tempHot_30min'])

try:
    # 전체 contID로 예측
    y_predictions, X_trans = model.forecast(X_test)

    print(f"  [OK] 예측 성공!")
    print(f"  예측 결과 수: {len(y_predictions)}")
    print(f"  X_trans shape: {X_trans.shape}")

    # 5. 결과 확인
    print("\n[5] 예측 결과 샘플")
    if 'contID' in X_trans.columns:
        print("  contID별 예측 수:")
        print(X_trans.groupby('contID').size())

    # 6. contID=1만 추출
    print("\n[6] contID=1 결과 추출")
    if 'contID' in X_trans.columns:
        zone1_results = X_trans[X_trans['contID'] == 1].copy()
        zone1_predictions = y_predictions[X_trans['contID'] == 1]

        print(f"  contID=1 예측 수: {len(zone1_predictions)}")
        print(f"  실제값과 비교:")

        # 실제값
        actual = test_df[test_df['contID'] == 1]['target_tempHot_30min'].values[:len(zone1_predictions)]

        # 비교 테이블
        comparison = pd.DataFrame({
            'colDate': test_df[test_df['contID'] == 1]['colDate'].values[:len(zone1_predictions)],
            'actual': actual,
            'predicted': zone1_predictions,
            'error': actual - zone1_predictions
        })

        print(comparison.head(10))

        # 평가 지표
        mae = np.abs(comparison['error']).mean()
        rmse = np.sqrt((comparison['error'] ** 2).mean())
        print(f"\n  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")

except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
