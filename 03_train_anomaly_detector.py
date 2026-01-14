"""
Step 3: 이상 탐지 모델 학습 (Isolation Forest)
목표: 비정상 온습도 패턴 감지
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def train_anomaly_detector():
    """Isolation Forest로 이상 탐지 모델 학습"""
    
    print("\n" + "="*60)
    print("이상 탐지 모델 학습")
    print("="*60)
    
    # 1. 데이터 로드
    cont_df = pd.read_csv('./cont_processed.csv')
    rack_df = pd.read_csv('./rack_processed.csv')
    
    print(f"\n데이터 로드:")
    print(f"  컨테인먼트: {len(cont_df):,} 행")
    print(f"  랙: {len(rack_df):,} 행")
    
    # 2. Feature 선택 (온도/습도 + 차이값)
    feature_cols = ['tempHot', 'tempCold', 'humiHot', 'humiCold', 'temp_diff', 'humi_diff']
    
    # 2-1. 컨테인먼트 이상 탐지 모델
    print("\n[1] 컨테인먼트 이상 탐지 모델 학습...")
    X_cont = cont_df[feature_cols].values
    
    # 스케일링
    scaler_cont = StandardScaler()
    X_cont_scaled = scaler_cont.fit_transform(X_cont)
    
    # Isolation Forest 학습
    iso_forest_cont = IsolationForest(
        contamination=0.05,  # 5% 이상치 가정
        random_state=42,
        n_estimators=100
    )
    iso_forest_cont.fit(X_cont_scaled)
    
    # 예측
    cont_df['anomaly_score'] = iso_forest_cont.score_samples(X_cont_scaled)
    cont_df['is_anomaly'] = iso_forest_cont.predict(X_cont_scaled)
    cont_df['is_anomaly'] = (cont_df['is_anomaly'] == -1).astype(int)
    
    anomaly_count_cont = cont_df['is_anomaly'].sum()
    print(f"✅ 컨테인먼트 이상 탐지 완료")
    print(f"   이상치 개수: {anomaly_count_cont:,} ({anomaly_count_cont/len(cont_df)*100:.2f}%)")
    
    # 2-2. 랙 이상 탐지 모델
    print("\n[2] 랙 이상 탐지 모델 학습...")
    X_rack = rack_df[feature_cols].values
    
    # 스케일링
    scaler_rack = StandardScaler()
    X_rack_scaled = scaler_rack.fit_transform(X_rack)
    
    # Isolation Forest 학습
    iso_forest_rack = IsolationForest(
        contamination=0.05,
        random_state=42,
        n_estimators=100
    )
    iso_forest_rack.fit(X_rack_scaled)
    
    # 예측
    rack_df['anomaly_score'] = iso_forest_rack.score_samples(X_rack_scaled)
    rack_df['is_anomaly'] = iso_forest_rack.predict(X_rack_scaled)
    rack_df['is_anomaly'] = (rack_df['is_anomaly'] == -1).astype(int)
    
    anomaly_count_rack = rack_df['is_anomaly'].sum()
    print(f"✅ 랙 이상 탐지 완료")
    print(f"   이상치 개수: {anomaly_count_rack:,} ({anomaly_count_rack/len(rack_df)*100:.2f}%)")
    
    # 3. 모델 저장
    print("\n[3] 모델 저장 중...")
    joblib.dump(iso_forest_cont, './models/anomaly_detector_cont.pkl')
    joblib.dump(scaler_cont, './models/scaler_cont.pkl')
    joblib.dump(iso_forest_rack, './models/anomaly_detector_rack.pkl')
    joblib.dump(scaler_rack, './models/scaler_rack.pkl')
    print("✅ 모델 저장 완료: ./models/")
    
    # 4. 이상치 분석
    print("\n" + "="*60)
    print("이상치 분석")
    print("="*60)
    
    # 컨테인먼트 존별 이상치
    print("\n[컨테인먼트] 존별 이상치 비율:")
    anomaly_by_zone = cont_df.groupby('contID')['is_anomaly'].agg(['sum', 'count', 'mean'])
    anomaly_by_zone.columns = ['이상치_개수', '전체_개수', '이상치_비율']
    anomaly_by_zone['이상치_비율'] = (anomaly_by_zone['이상치_비율'] * 100).round(2)
    print(anomaly_by_zone)
    
    # 랙 존별 이상치
    print("\n[랙] 존별 이상치 비율:")
    rack_anomaly_by_zone = rack_df.groupby('contID')['is_anomaly'].agg(['sum', 'count', 'mean'])
    rack_anomaly_by_zone.columns = ['이상치_개수', '전체_개수', '이상치_비율']
    rack_anomaly_by_zone['이상치_비율'] = (rack_anomaly_by_zone['이상치_비율'] * 100).round(2)
    print(rack_anomaly_by_zone)
    
    # 5. 시각화
    print("\n[4] 이상치 시각화 생성 중...")
    create_anomaly_visualizations(cont_df, rack_df)
    
    # 6. 결과 저장
    cont_df.to_csv('./cont_with_anomalies.csv', index=False)
    rack_df.to_csv('./rack_with_anomalies.csv', index=False)
    print("\n✅ 이상치 포함 데이터 저장 완료")
    
    return iso_forest_cont, iso_forest_rack

def create_anomaly_visualizations(cont_df, rack_df):
    """이상치 시각화"""
    
    import os
    os.makedirs('./visualizations', exist_ok=True)
    
    # 1. 컨테인먼트 이상치 시계열
    plt.figure(figsize=(16, 6))
    for cont_id in sorted(cont_df['contID'].unique()):
        zone_data = cont_df[cont_df['contID'] == cont_id]
        anomalies = zone_data[zone_data['is_anomaly'] == 1]
        
        plt.subplot(1, 4, cont_id)
        plt.scatter(zone_data['colDate'], zone_data['tempHot'], 
                   c='blue', alpha=0.3, s=1, label='Normal')
        plt.scatter(anomalies['colDate'], anomalies['tempHot'], 
                   c='red', alpha=0.8, s=10, label='Anomaly')
        plt.xlabel('Date')
        plt.ylabel('Hot Aisle Temp (°C)')
        plt.title(f'Zone {cont_id}')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./visualizations/containment_anomalies_timeseries.png', dpi=150, bbox_inches='tight')
    print("✅ 저장: containment_anomalies_timeseries.png")
    
    # 2. 온도 차이 vs 이상치 스코어
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for cont_id in sorted(cont_df['contID'].unique()):
        zone_data = cont_df[cont_df['contID'] == cont_id]
        plt.scatter(zone_data['temp_diff'], zone_data['anomaly_score'], 
                   alpha=0.3, s=5, label=f'Zone {cont_id}')
    plt.xlabel('Temperature Difference (Hot - Cold)')
    plt.ylabel('Anomaly Score')
    plt.title('Containment: Temp Diff vs Anomaly Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for cont_id in sorted(rack_df['contID'].unique()):
        zone_data = rack_df[rack_df['contID'] == cont_id]
        plt.scatter(zone_data['temp_diff'], zone_data['anomaly_score'], 
                   alpha=0.3, s=5, label=f'Zone {cont_id}')
    plt.xlabel('Temperature Difference (Hot - Cold)')
    plt.ylabel('Anomaly Score')
    plt.title('Rack: Temp Diff vs Anomaly Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./visualizations/anomaly_score_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ 저장: anomaly_score_analysis.png")

if __name__ == "__main__":
    import os
    os.makedirs('./models', exist_ok=True)
    
    # 이상 탐지 모델 학습
    iso_forest_cont, iso_forest_rack = train_anomaly_detector()
    
    print("\n" + "="*60)
    print("✅ 이상 탐지 모델 학습 완료!")
    print("="*60)
    print("\n생성된 파일:")
    print("  - models/anomaly_detector_cont.pkl")
    print("  - models/anomaly_detector_rack.pkl")
    print("  - cont_with_anomalies.csv")
    print("  - rack_with_anomalies.csv")
    print("  - visualizations/containment_anomalies_timeseries.png")
    print("  - visualizations/anomaly_score_analysis.png")
