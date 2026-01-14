import pandas as pd
import numpy as np
import os
import yaml

print("="*60)
print("강력한 중복 제거 및 재집계")
print("="*60)

# 1. 데이터 로드
cont_df = pd.read_csv('./cont_processed.csv')
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

# 5. 30분 후 타겟 생성
print("\n타겟 생성 중...")
df_agg['target_tempHot_30min'] = df_agg.groupby('contID')['tempHot'].shift(-3)
df_agg = df_agg.dropna(subset=['target_tempHot_30min'])
print(f"✅ 타겟 생성 후: {len(df_agg):,} 행")

# 6. 최종 중복 확인
final_dup = df_agg.groupby(['contID', 'colDate']).size()
final_dup_count = (final_dup > 1).sum()

if final_dup_count > 0:
    print(f"\n⚠️ 여전히 {final_dup_count}개 중복!")
    # 강제 중복 제거
    df_agg = df_agg.drop_duplicates(subset=['contID', 'colDate'], keep='first')
    print(f"강제 제거 후: {len(df_agg):,} 행")
else:
    print("\n✅ 중복 0개 확인!")

# 7. 컬럼 순서 정리
final_cols = [
    'contID', 'colDate', 'tempHot', 'tempCold', 'humiHot', 'humiCold',
    'temp_diff', 'humi_diff', 'hour', 'day_of_week', 'rack_count',
    'target_tempHot_30min'
]

df_final = df_agg[final_cols].copy()

# 8. 인덱스 리셋 (깔끔하게)
df_final = df_final.reset_index(drop=True)

print(f"\n최종 데이터: {len(df_final):,} 행 × {len(final_cols)} 컬럼")

# 9. MLTable 폴더 생성
print("\nMLTable 폴더 생성...")
os.makedirs('cont_forecast_clean', exist_ok=True)

# CSV 저장 (인덱스 제외)
df_final.to_csv('cont_forecast_clean/data.csv', index=False)

# MLTable 파일
mltable = {
    'type': 'mltable',
    'paths': [{'file': './data.csv'}],
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

# 10. 철저한 검증
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