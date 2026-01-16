# -*- coding: utf-8 -*-
"""
대시보드 데모용 가짜 데이터 생성
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 60)
print("대시보드 데모용 데이터 생성")
print("=" * 60)

# 설정
START_DATE = datetime(2025, 9, 23, 0, 0, 0)
END_DATE = datetime(2025, 9, 25, 12, 0, 0)
INTERVAL_MINUTES = 10
ZONES = [1, 2, 3, 4]

# 시간 범위 생성
time_range = pd.date_range(start=START_DATE, end=END_DATE, freq=f'{INTERVAL_MINUTES}min')

data = []

for zone_id in ZONES:
    # Zone별로 약간씩 다른 베이스 온도
    base_temp = 30.5 + (zone_id * 0.2)

    for i, timestamp in enumerate(time_range):
        # 시간대별 패턴 (낮에 더 더움)
        hour = timestamp.hour
        daily_pattern = 1.5 * np.sin((hour - 6) * np.pi / 12)  # 오후 2시경 최고

        # 랜덤 노이즈
        noise = np.random.normal(0, 0.3)

        # 실제 온도
        actual_temp = base_temp + daily_pattern + noise

        # 30분 후 예측 (실제 + 약간의 변화 + 예측 오차)
        future_trend = 0.1 * np.random.randn()  # 미래 추세
        prediction_error = np.random.normal(0, 0.4)  # 예측 오차
        predicted_temp = actual_temp + future_trend + prediction_error

        # Zone 3은 오후 시간대에 과열되도록 (임계값 32도 초과)
        if zone_id == 3 and 13 <= hour <= 16:
            actual_temp += 1.5
            predicted_temp += 1.8

        # Zone 2는 특정 시점에 급격한 변화
        if zone_id == 2 and timestamp.day == 25 and 8 <= hour <= 9:
            actual_temp += 1.2
            predicted_temp += 1.5

        data.append({
            'colDate': timestamp,
            'contID': zone_id,
            'tempHot': round(actual_temp, 2),
            'target_tempHot_30min': round(predicted_temp, 2)
        })

# DataFrame 생성
df = pd.DataFrame(data)

# 정렬
df = df.sort_values(['colDate', 'contID']).reset_index(drop=True)

# 저장
output_file = "cont_forecast_data.csv"
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n[생성 완료]")
print(f"파일: {output_file}")
print(f"행 수: {len(df)}")
print(f"기간: {df['colDate'].min()} ~ {df['colDate'].max()}")
print(f"Zone: {sorted(df['contID'].unique())}")
print(f"\n[온도 범위]")
print(df.groupby('contID')['tempHot'].agg(['min', 'max', 'mean']))
print(f"\n[샘플 데이터]")
print(df.head(10))

# 임계값 초과 확인
threshold = 32.0
over_threshold = df[df['target_tempHot_30min'] >= threshold]
print(f"\n[임계값 초과 예측]")
print(f"임계값 {threshold}도 이상 예측 건수: {len(over_threshold)}")
if len(over_threshold) > 0:
    print(over_threshold[['colDate', 'contID', 'tempHot', 'target_tempHot_30min']].head())

print("\n" + "=" * 60)
print("데모 데이터 생성 완료!")
print("이제 대시보드에서 확인하세요: http://localhost:8501")
print("=" * 60)
