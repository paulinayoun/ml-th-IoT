import pandas as pd
import os
import yaml

print("="*60)
print("예측 데이터 재생성 (중복 제거)")
print("="*60)

# 1. 데이터 로드
cont_df = pd.read_csv('./cont_processed.csv')
cont_df['colDate'] = pd.to_datetime(cont_df['colDate'])

print(f"\n원본 데이터: {len(cont_df):,} 행")

# 2. 중복 확인
duplicates = cont_df.groupby(['contID', 'colDate']).size()
dup_count = (duplicates > 1).sum()
print(f"중복된 시간-존 조합: {dup_count}개")

if dup_count > 0:
    print("\n중복 제거 중...")
    # 중복 제거: 같은 contID, colDate의 경우 첫 번째 행만 유지
    cont_df = cont_df.drop_duplicates(subset=['contID', 'colDate'], keep='first')
    print(f"✅ 중복 제거 후: {len(cont_df):,} 행")

# 3. 정렬
cont_df = cont_df.sort_values(['contID', 'colDate']).reset_index(drop=True)

# 4. 30분 후 온도 타겟 생성
print("\n30분 후 온도 타겟 생성 중...")
cont_df['target_tempHot_30min'] = cont_df.groupby('contID')['tempHot'].shift(-3)

# 결측치 제거
cont_df = cont_df.dropna(subset=['target_tempHot_30min'])
print(f"✅ 타겟 생성 후: {len(cont_df):,} 행")

# 5. 최종 중복 확인
final_dup = cont_df.groupby(['contID', 'colDate']).size()
final_dup_count = (final_dup > 1).sum()

if final_dup_count > 0:
    print(f"\n⚠️ 경고: 여전히 {final_dup_count}개 중복 존재!")
    print("추가 중복 제거...")
    cont_df = cont_df.drop_duplicates(subset=['contID', 'colDate'], keep='first')
else:
    print("\n✅ 중복 없음 확인!")

# 6. Feature 선택
feature_cols = [
    'contID', 'colDate', 'tempHot', 'tempCold', 'humiHot', 'humiCold',
    'temp_diff', 'humi_diff', 'hour', 'day_of_week', 'rack_count',
    'target_tempHot_30min'
]

df_forecast = cont_df[feature_cols].copy()

print(f"\n최종 데이터: {len(df_forecast):,} 행 × {len(feature_cols)} 컬럼")

# 7. MLTable 폴더 생성
print("\nMLTable 폴더 생성 중...")
os.makedirs('cont_forecast_mltable_v2', exist_ok=True)

# CSV 저장
df_forecast.to_csv('cont_forecast_mltable_v2/cont_forecast_data.csv', index=False)

# MLTable 파일 생성
mltable_content = {
    'type': 'mltable',
    'paths': [{'file': './cont_forecast_data.csv'}],
    'transformations': [
        {
            'read_delimited': {
                'delimiter': ',',
                'encoding': 'utf8',
                'header': 'all_files_same_headers'
            }
        }
    ]
}

with open('cont_forecast_mltable_v2/MLTable', 'w') as f:
    yaml.dump(mltable_content, f, default_flow_style=False)

print("✅ MLTable 폴더 생성 완료: cont_forecast_mltable_v2/")

# 8. 데이터 검증
print("\n" + "="*60)
print("데이터 검증")
print("="*60)

print("\n존별 데이터 개수:")
print(df_forecast.groupby('contID').size())

print("\n각 존의 시간 범위:")
for cont_id in sorted(df_forecast['contID'].unique()):
    zone_data = df_forecast[df_forecast['contID'] == cont_id]
    print(f"  존{cont_id}: {zone_data['colDate'].min()} ~ {zone_data['colDate'].max()}")

print("\n시간 간격 확인 (10분이어야 함):")
for cont_id in sorted(df_forecast['contID'].unique())[:2]:  # 처음 2개 존만 체크
    zone_data = df_forecast[df_forecast['contID'] == cont_id].copy()
    zone_data = zone_data.sort_values('colDate')
    time_diff = zone_data['colDate'].diff().dt.total_seconds() / 60
    unique_intervals = time_diff.dropna().unique()
    print(f"  존{cont_id}: {unique_intervals[:5]}분 간격")

print("\n" + "="*60)
print("✅ 완료! cont_forecast_mltable_v2 폴더를 업로드하세요")
print("="*60)