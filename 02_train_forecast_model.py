"""
Step 2: AutoML을 이용한 온도 예측 모델 학습
목표: 30분 후 Hot Aisle 온도 예측
"""
from azure_config import get_ml_client
from azure.ai.ml import automl, Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.automl import ForecastingJob
import pandas as pd

def prepare_forecast_data():
    """예측용 데이터 준비 - 30분 후 온도를 타겟으로"""
    
    print("\n" + "="*60)
    print("예측 데이터 준비")
    print("="*60)
    
    # 컨테인먼트 데이터 로드
    df = pd.read_csv('./cont_processed.csv')
    df['colDate'] = pd.to_datetime(df['colDate'])
    df = df.sort_values(['contID', 'colDate'])
    
    # 30분 후 온도를 타겟으로 설정 (3개 행 = 30분)
    df['target_tempHot_30min'] = df.groupby('contID')['tempHot'].shift(-3)
    
    # 결측치 제거 (마지막 3개 행)
    df = df.dropna(subset=['target_tempHot_30min'])
    
    # 학습에 필요한 컬럼만 선택
    feature_cols = [
        'contID', 'colDate', 'tempHot', 'tempCold', 'humiHot', 'humiCold',
        'temp_diff', 'humi_diff', 'hour', 'day_of_week', 'rack_count',
        'target_tempHot_30min'
    ]
    
    df_forecast = df[feature_cols]
    
    # 저장
    df_forecast.to_csv('./cont_forecast_data.csv', index=False)
    print(f"✅ 예측 데이터 생성: {len(df_forecast):,} 행")
    print(f"   타겟: 30분 후 Hot Aisle 온도")
    print(f"   Feature 개수: {len(feature_cols)-1}개")
    
    return df_forecast

def create_automl_forecast_job():
    """AutoML Forecasting Job 생성"""
    
    ml_client = get_ml_client()
    
    print("\n" + "="*60)
    print("AutoML 예측 모델 학습 시작")
    print("="*60)
    
    # 데이터 입력 설정
    my_training_data_input = Input(
        type=AssetTypes.URI_FILE,
        path="./cont_forecast_data.csv"
    )
    
    # Forecasting Job 설정
    forecasting_job = automl.forecasting(
        compute="cpu-cluster",  # 컴퓨팅 클러스터 (다음 단계에서 생성)
        experiment_name="th-forecasting-experiment",
        training_data=my_training_data_input,
        target_column_name="target_tempHot_30min",
        primary_metric="normalized_root_mean_squared_error",
        n_cross_validations=5,
        enable_model_explainability=True,
        # Time series 설정
        time_column_name="colDate",
        time_series_id_column_names=["contID"],
        forecast_horizon=3,  # 30분 (3개 타임스텝)
    )
    
    # Limits 설정 (학습 시간 제한)
    forecasting_job.set_limits(
        timeout_minutes=60,
        trial_timeout_minutes=20,
        max_trials=10,
        max_concurrent_trials=4,
        enable_early_termination=True,
    )
    
    # Featurization 설정
    forecasting_job.set_featurization(
        mode="auto",
        blocked_transformers=["LabelEncoder"],
    )
    
    print("✅ Job 설정 완료")
    print(f"   실험명: th-forecasting-experiment")
    print(f"   타겟: 30분 후 온도")
    print(f"   Cross Validation: 5-fold")
    print(f"   최대 시도: 10개 모델")
    print(f"   타임아웃: 60분")
    
    return forecasting_job

def main():
    """메인 실행 함수"""
    
    # 1. 예측 데이터 준비
    prepare_forecast_data()
    
    # 2. AutoML Job 생성
    job = create_automl_forecast_job()
    
    print("\n" + "="*60)
    print("다음 단계:")
    print("="*60)
    print("1. 컴퓨팅 클러스터 생성 필요 (02_create_compute.py)")
    print("2. Job 제출 (03_submit_job.py)")
    print("\n또는 Azure ML Studio에서 GUI로 진행 가능합니다.")

if __name__ == "__main__":
    main()
