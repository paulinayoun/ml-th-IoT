# -*- coding: utf-8 -*-
"""
Anomaly Dashboard 데모용 가짜 데이터 생성
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 60)
print("Anomaly Dashboard 데모용 데이터 생성")
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
    base_humi = 45.0 + (zone_id * 0.5)

    for i, timestamp in enumerate(time_range):
        # 시간대별 패턴 (낮에 더 더움)
        hour = timestamp.hour
        daily_pattern = 1.5 * np.sin((hour - 6) * np.pi / 12)

        # 랜덤 노이즈
        temp_noise = np.random.normal(0, 0.3)
        humi_noise = np.random.normal(0, 2.0)

        # 정상 온도 및 습도
        temp = base_temp + daily_pattern + temp_noise
        humi = base_humi + humi_noise

        # 이상치 주입 (약 5% 확률)
        is_anomaly = 0
        anomaly_score = np.random.uniform(-0.1, 0.1)  # 정상 범위

        # Zone 1: 9/24 새벽 급격한 온도 상승 (장비 고장 시뮬레이션)
        if zone_id == 1 and timestamp.day == 24 and 2 <= hour <= 4:
            temp += 4.5
            is_anomaly = 1
            anomaly_score = np.random.uniform(-0.8, -0.5)

        # Zone 2: 9/24 오후 습도 급감 (냉각 시스템 문제)
        if zone_id == 2 and timestamp.day == 24 and 14 <= hour <= 16:
            humi -= 15
            is_anomaly = 1
            anomaly_score = np.random.uniform(-0.7, -0.4)

        # Zone 3: 9/25 오전 온도 스파이크 (순간적 이상)
        if zone_id == 3 and timestamp.day == 25 and hour == 9 and timestamp.minute in [0, 10, 20]:
            temp += 5.0
            is_anomaly = 1
            anomaly_score = np.random.uniform(-0.9, -0.6)

        # Zone 4: 9/23 저녁 불규칙한 변동
        if zone_id == 4 and timestamp.day == 23 and 18 <= hour <= 20:
            if np.random.random() < 0.3:
                temp += np.random.uniform(-2, 3)
                is_anomaly = 1
                anomaly_score = np.random.uniform(-0.6, -0.3)

        # 랜덤 이상치 추가 (전체 데이터의 2%)
        if np.random.random() < 0.02 and is_anomaly == 0:
            temp += np.random.choice([-3, 3])
            is_anomaly = 1
            anomaly_score = np.random.uniform(-0.5, -0.2)

        data.append({
            'colDate': timestamp,
            'contID': f'zone_{zone_id}',
            'tempHot': round(temp, 2),
            'humiHot': round(humi, 2),
            'is_anomaly': is_anomaly,
            'anomaly_score': round(anomaly_score, 4)
        })

# DataFrame 생성
df = pd.DataFrame(data)

# 정렬
df = df.sort_values(['colDate', 'contID']).reset_index(drop=True)

# 저장
output_file = "cont_with_anomalies.csv"
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n[생성 완료]")
print(f"파일: {output_file}")
print(f"행 수: {len(df)}")
print(f"기간: {df['colDate'].min()} ~ {df['colDate'].max()}")
print(f"Zone: {sorted(df['contID'].unique())}")

print(f"\n[이상치 통계]")
anomaly_stats = df.groupby('contID')['is_anomaly'].agg(['sum', 'count'])
anomaly_stats['rate'] = (anomaly_stats['sum'] / anomaly_stats['count'] * 100).round(2)
anomaly_stats.columns = ['이상치_개수', '전체_개수', '이상치_비율(%)']
print(anomaly_stats)

print(f"\n[이상치 샘플]")
anomalies = df[df['is_anomaly'] == 1].sort_values('anomaly_score')
print(anomalies[['colDate', 'contID', 'tempHot', 'humiHot', 'anomaly_score']].head(10))

print("\n" + "=" * 60)
print("Anomaly 데모 데이터 생성 완료!")
print("이제 Anomaly Dashboard에서 확인하세요: http://localhost:8501")
print("=" * 60)
