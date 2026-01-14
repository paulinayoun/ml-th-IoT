import pandas as pd
import numpy as np
import os
import yaml

print("="*60)
print("강력한 중복 제거 및 재집계")
print("="*60)

# 1. 데이터 로드
cont_df = pd.read_csv('./data/cont_processed.csv')
cont_df['colDate'] = pd.to_datetime(cont_df['colDate'])

print(f"\n원본 데이터: {len(cont_df):,} 행")

# 2. 중복 확인
before_dup = cont_df.groupby(['contID', 'colDate']).size()
before_dup_count = (before_dup > 1).sum()
print(f"중복된 시간-존 조합: {before_dup_count}개")

# 3. 완전 재집계 (같은 contID + colDate는 평균으로 합침)
print("\n재집계 중 (중복 완전 제거)...")

agg_dict = {
    'tempHot': 'mean',
    'tempCold': 'mean',
    'humiHot': 'mean',
    'humiCold': 'mean',
    'temp_diff': 'mean',
    'humi_diff': 'mean',
    'hour': 'first',
    'day_of_week': 'first',
    'rack_count': 'first'
}

df_agg = cont_df.groupby(['contID', 'colDate'], as_index=False).agg(agg_dict)
print(f"✅ 재집계 후: {len(df_agg):,} 행")

# 4. 정렬
df_agg = df_agg.sort_values(['contID', 'colDate']).reset_index(drop=True)

# 5. 15분 간격으로 리샘플링 (Azure AutoML 최소 간격)
print("\n15분 간격으로 리샘플링 중...")
print(f"  리샘플링 전: {len(df_agg):,} 행 (10분 간격)")

# 각 존별로 15분 간격 리샘플링
resampled_dfs = []
for cid in sorted(df_agg['contID'].unique()):
    zone_data = df_agg[df_agg['contID'] == cid].copy()
    zone_data = zone_data.set_index('colDate')

    # 15분 간격으로 리샘플링 (평균)
    zone_resampled = zone_data.resample('15min').mean()
    zone_resampled['contID'] = cid
    zone_resampled.reset_index(inplace=True)

    # hour, day_of_week 재계산 (정수형이므로 평균이 이상해짐)
    zone_resampled['hour'] = zone_resampled['colDate'].dt.hour
    zone_resampled['day_of_week'] = zone_resampled['colDate'].dt.dayofweek
    # rack_count는 NaN을 제거하고 정수로 변환 (forward fill -> backward fill)
    zone_resampled['rack_count'] = zone_resampled['rack_count'].ffill().bfill().round().astype(int)

    resampled_dfs.append(zone_resampled)

df_agg = pd.concat(resampled_dfs, ignore_index=True)
df_agg = df_agg.sort_values(['contID', 'colDate']).reset_index(drop=True)
print(f"✅ 리샘플링 후: {len(df_agg):,} 행 (15분 간격)")

# 5.5. 빠진 시간대 채우기 (Azure AutoML 요구사항)
print("\n빠진 시간대 채우기 중...")
print(f"  채우기 전: {len(df_agg):,} 행")

# 전체 시간 범위 확인
min_time = df_agg['colDate'].min()
max_time = df_agg['colDate'].max()

# 각 존별로 완전한 시계열 생성
filled_dfs = []
for cid in sorted(df_agg['contID'].unique()):
    zone_data = df_agg[df_agg['contID'] == cid].copy()

    # 15분 간격의 완전한 시간 범위 생성
    complete_times = pd.DataFrame({
        'colDate': pd.date_range(start=min_time, end=max_time, freq='15min'),
        'contID': cid
    })

    # 기존 데이터와 병합 (빠진 시간대는 NaN으로 채워짐)
    zone_filled = complete_times.merge(zone_data, on=['contID', 'colDate'], how='left')

    # 정수형 컬럼은 forward fill
    zone_filled['hour'] = zone_filled['colDate'].dt.hour
    zone_filled['day_of_week'] = zone_filled['colDate'].dt.dayofweek
    zone_filled['rack_count'] = zone_filled['rack_count'].ffill().bfill().astype(int)

    # 센서 값은 forward fill (최근 값 사용)
    sensor_cols = ['tempHot', 'tempCold', 'humiHot', 'humiCold', 'temp_diff', 'humi_diff']
    for col in sensor_cols:
        zone_filled[col] = zone_filled[col].ffill()

    filled_dfs.append(zone_filled)

df_agg = pd.concat(filled_dfs, ignore_index=True)
df_agg = df_agg.sort_values(['contID', 'colDate']).reset_index(drop=True)
print(f"✅ 채우기 후: {len(df_agg):,} 행 (연속적인 15분 간격)")

# 6. 30분 후 타겟 생성
print("\n타겟 생성 중...")
# 15분 간격이므로 2칸 이동 = 30분 후
df_agg['target_tempHot_30min'] = df_agg.groupby('contID')['tempHot'].shift(-2)
df_agg = df_agg.dropna(subset=['target_tempHot_30min'])
print(f"✅ 타겟 생성 후: {len(df_agg):,} 행")

# 7. 최종 중복 확인
final_dup = df_agg.groupby(['contID', 'colDate']).size()
final_dup_count = (final_dup > 1).sum()

if final_dup_count > 0:
    print(f"\n⚠️ 여전히 {final_dup_count}개 중복!")
    # 강제 중복 제거
    df_agg = df_agg.drop_duplicates(subset=['contID', 'colDate'], keep='first')
    print(f"강제 제거 후: {len(df_agg):,} 행")
else:
    print("\n✅ 중복 0개 확인!")

# 8. 컬럼 순서 정리
final_cols = [
    'contID', 'colDate', 'tempHot', 'tempCold', 'humiHot', 'humiCold',
    'temp_diff', 'humi_diff', 'hour', 'day_of_week', 'rack_count',
    'target_tempHot_30min'
]

df_final = df_agg[final_cols].copy()

# 9. 인덱스 리셋 (깔끔하게)
df_final = df_final.reset_index(drop=True)

print(f"\n최종 데이터: {len(df_final):,} 행 × {len(final_cols)} 컬럼")

# 10. MLTable 폴더 생성
print("\nMLTable 폴더 생성...")
os.makedirs('cont_forecast_clean', exist_ok=True)

# CSV 저장 (인덱스 제외)
df_final.to_csv('cont_forecast_clean/data.csv', index=False)

# MLTable 파일
mltable = {
    'type': 'mltable',
    'paths': [{'file': 'data.csv'}],
    'transformations': [
        {
            'read_delimited': {
                'delimiter': ',',
                'encoding': 'utf8',
                'header': 'all_files_same_headers',
                'empty_as_string': False
            }
        }
    ]
}

with open('cont_forecast_clean/MLTable', 'w') as f:
    yaml.dump(mltable, f)

print("✅ MLTable 폴더: cont_forecast_clean/")

# 11. 철저한 검증
print("\n" + "="*60)
print("데이터 검증")
print("="*60)

# 중복 재확인
verify_dup = df_final.groupby(['contID', 'colDate']).size()
verify_dup_count = (verify_dup > 1).sum()
print(f"\n중복 검사: {verify_dup_count}개")

if verify_dup_count > 0:
    print("❌ 중복 발견!")
    print(df_final[df_final.duplicated(subset=['contID', 'colDate'], keep=False)].head(20))
else:
    print("✅ 중복 없음!")

# 시간 연속성 확인
print("\n시간 연속성 확인:")
for cid in sorted(df_final['contID'].unique()):
    zone_data = df_final[df_final['contID'] == cid].sort_values('colDate')
    time_diff = zone_data['colDate'].diff().dt.total_seconds() / 60
    
    # 가장 많은 간격
    mode_interval = time_diff.mode()[0] if len(time_diff.mode()) > 0 else time_diff.median()
    
    print(f"  존{cid}: 주 간격 {mode_interval:.0f}분, 총 {len(zone_data):,}개 시점")

print("\n존별 데이터 개수:")
print(df_final.groupby('contID').size())

print("\n" + "="*60)
print("✅ 완료!")
print("="*60)
print("\ncont_forecast_clean 폴더를 업로드하세요")