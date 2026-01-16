# -*- coding: utf-8 -*-
"""
cont_processed.csv를 cont_forecast_clean 형식으로 변환
"""
import pandas as pd

print("="*60)
print("검증 데이터 준비")
print("="*60)

# 1. cont_processed.csv 로드
print("\n[1] cont_processed.csv 로드")
df = pd.read_csv('./data/cont_processed.csv', parse_dates=['colDate'])
print(f"  Shape: {df.shape}")
print(f"  Date range: {df['colDate'].min()} ~ {df['colDate'].max()}")
print(f"  Columns: {list(df.columns)}")

# 2. cont_forecast_clean 형식 확인
print("\n[2] cont_forecast_clean 형식 확인")
df_clean = pd.read_csv('./cont_forecast_clean/data.csv', parse_dates=['colDate'])
print(f"  Shape: {df_clean.shape}")
print(f"  Columns: {list(df_clean.columns)}")

# 3. 변환 작업
print("\n[3] 데이터 변환")

# 3-1. month, day 컬럼 제거
if 'month' in df.columns:
    df = df.drop(columns=['month'])
    print("  - month 컬럼 제거")
if 'day' in df.columns:
    df = df.drop(columns=['day'])
    print("  - day 컬럼 제거")

# 3-2. target_tempHot_30min 생성
# 30분 후 = 15분 x 2 step
df = df.sort_values(['contID', 'colDate']).reset_index(drop=True)
df['target_tempHot_30min'] = df.groupby('contID')['tempHot'].shift(-2)
print("  - target_tempHot_30min 생성 (30분 후 온도)")

# 3-3. NaN 제거 (마지막 2개 행)
df_before = len(df)
df = df.dropna(subset=['target_tempHot_30min'])
print(f"  - NaN 제거: {df_before} -> {len(df)} 행")

# 3-4. 컬럼 순서 맞추기
column_order = list(df_clean.columns)
df = df[column_order]
print(f"  - 컬럼 순서 정렬: {list(df.columns)}")

# 4. 결과 저장
output_path = './data/cont_validation.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n[4] 저장 완료: {output_path}")
print(f"  Shape: {df.shape}")
print(f"  Date range: {df['colDate'].min()} ~ {df['colDate'].max()}")

# 5. 비교
print("\n[5] cont_forecast_clean vs cont_validation 비교")
print(f"  cont_forecast_clean: {len(df_clean):,} 행")
print(f"  cont_validation:     {len(df):,} 행")
print(f"  차이:                {len(df) - len(df_clean):,} 행")

if len(df) > len(df_clean):
    print(f"\n  [OK] cont_validation에 {len(df) - len(df_clean)}행의 추가 데이터가 있습니다!")
    print(f"       이 데이터는 모델이 학습하지 않은 새로운 데이터입니다.")
elif len(df) == len(df_clean):
    print(f"\n  [주의] 두 데이터셋의 행 수가 동일합니다.")
    print(f"        날짜 범위를 확인하여 차이가 있는지 검토하세요.")
else:
    print(f"\n  [주의] cont_validation이 더 적습니다. 데이터를 확인하세요.")

print("\n" + "="*60)
print("완료!")
print("="*60)
