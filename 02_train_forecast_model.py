"""
Step 2: AutoML을 이용한 온도 예측 모델 학습
목표: 30분 후 Hot Aisle 온도 예측

주의: 먼저 clean_data.py를 실행하여 cont_forecast_clean/ 폴더를 생성해야 합니다.
"""
from azure_config import get_ml_client
from azure.ai.ml import automl, Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.automl import ForecastingJob
import pandas as pd
import os

def verify_clean_data():
    """정제된 데이터 확인"""

    print("\n" + "="*60)
    print("정제된 데이터 확인")
    print("="*60)

    # MLTable 폴더 확인
    if not os.path.exists('./cont_forecast_clean'):
        raise FileNotFoundError(
            "cont_forecast_clean 폴더가 없습니다.\n"
            "먼저 clean_data.py를 실행하여 중복 없는 데이터를 생성하세요."
        )

    # 데이터 로드 및 검증
    df = pd.read_csv('./cont_forecast_clean/data.csv')
    df['colDate'] = pd.to_datetime(df['colDate'])

    print(f"✅ 데이터 로드: {len(df):,} 행")
    print(f"   컬럼: {list(df.columns)}")

    # 중복 확인
    dup_check = df.groupby(['contID', 'colDate']).size()
    dup_count = (dup_check > 1).sum()

    if dup_count > 0:
        raise ValueError(f"⚠️ 여전히 {dup_count}개의 중복이 있습니다. clean_data.py를 다시 실행하세요.")

    print("✅ 중복 검사 통과: 0개")
    print(f"   존 개수: {df['contID'].nunique()}개")
    print(f"   시간 범위: {df['colDate'].min()} ~ {df['colDate'].max()}")

    return df

def create_automl_forecast_job():
    """AutoML Forecasting Job 생성"""

    ml_client = get_ml_client()

    print("\n" + "="*60)
    print("AutoML 예측 모델 학습 시작")
    print("="*60)

    # MLTable 데이터 입력 설정 (중복 제거된 데이터)
    my_training_data_input = Input(
        type=AssetTypes.MLTABLE,
        path="./cont_forecast_clean"
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
        forecast_horizon=2,  # 30분 (15분 × 2 = 30분)
        frequency="15min",  # 15분 간격 (Azure AutoML 최소 간격)
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
    print(f"   데이터: cont_forecast_clean (MLTable 형식)")
    print(f"   실험명: th-forecasting-experiment")
    print(f"   타겟: 30분 후 온도")
    print(f"   Cross Validation: 5-fold")
    print(f"   최대 시도: 10개 모델")
    print(f"   타임아웃: 60분")

    return forecasting_job

def main():
    """메인 실행 함수"""

    # 1. 정제된 데이터 확인
    df = verify_clean_data()

    # 2. cont_forecast_data.csv 파일로 저장
    output_path = "cont_forecast_data.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ 예측용 데이터 저장 완료: {output_path}")

    # AutoML Job 생성 및 제출 부분은 주석 처리하거나 필요에 따라 활성화
    # job = create_automl_forecast_job()
    # print("\n" + "="*60)
    # print("다음 단계:")
    # print("="*60)
    # print("1. Azure ML Studio에서 cont_forecast_clean 폴더를 데이터 자산으로 등록")
    # print("2. 컴퓨팅 클러스터 생성 (cpu-cluster)")
    # print("3. 위 Job 설정으로 AutoML Forecasting 실행")
    # print("\n💡 또는 이 스크립트를 확장하여 ml_client.jobs.create_or_update(job)로 제출 가능합니다.")

if __name__ == "__main__":
    main()
