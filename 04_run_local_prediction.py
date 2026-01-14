

import pandas as pd
import joblib
import os

# --- 설정 ---
# Azure ML Studio에서 다운로드하여 models/ 폴더에 저장한 모델 파일의 경로
MODEL_PATH = "models/automl_forecast_model.pkl"

def load_model(path):
    """
    지정된 경로에서 pkl 모델 파일을 로드합니다.
    """
    print(f"모델 로딩 중: {path}")
    if not os.path.exists(path):
        print("="*60)
        print(f"오류: 모델 파일({path})을 찾을 수 없습니다.")
        print("1. Azure ML Studio에서 AutoML 실험 결과를 확인하세요.")
        print("2. 가장 좋은 모델(Best model)을 선택하고 'Download' 버튼을 눌러 zip 파일을 받으세요.")
        print("3. 압축을 풀고 'model.pkl' 파일을 'models/' 폴더에 복사하세요.")
        print("   (파일 이름을 'automl_forecast_model.pkl'로 변경하는 것을 권장합니다.)")
        print("="*60)
        return None
    
    try:
        model = joblib.load(path)
        print("✅ 모델 로딩 성공")
        return model
    except Exception as e:
        print(f"❌ 모델 로딩 중 오류 발생: {e}")
        print("이 모델은 특정 환경에서 학습되었습니다.")
        print("Azure에서 모델과 함께 다운로드된 'conda_env.yml' 파일을 사용하여")
        print("Conda 환경을 생성한 후 다시 시도해 보세요.")
        print("명령어 예시: conda env create -f path/to/your/conda_env.yml")
        return None


def prepare_input_data(zone_id='zone_1'):
    """
    예측에 사용할 입력 데이터를 준비합니다.
    AutoML은 학습에 사용된 전체 시계열 데이터(X)를 입력으로 받습니다.
    """
    print(f"\n'{zone_id}'의 예측용 입력 데이터 준비 중...")
    try:
        df = pd.read_csv('cont_forecast_data.csv', parse_dates=['colDate'])
    except FileNotFoundError:
        print("오류: 'cont_forecast_data.csv' 파일을 찾을 수 없습니다.")
        return None

    # 특정 존의 데이터만 필터링
    zone_df = df[df['contID'] == zone_id].copy()
    
    if zone_df.empty:
        print(f"오류: '{zone_id}'에 해당하는 데이터가 없습니다.")
        return None

    # AutoML 시계열 모델은 예측을 위해 학습에 사용된 전체 데이터를 입력으로 받습니다.
    # 모델이 내부적으로 최신 데이터를 기반으로 미래를 예측합니다.
    # 예측 대상(y)인 'target_tempHot_30min' 컬럼은 입력에서 제외합니다.
    X_test = zone_df.drop(columns=['target_tempHot_30min'])
    
    print(f"✅ 입력 데이터 준비 완료: {len(X_test)}개 행")
    return X_test


def main():
    """
    메인 실행 함수
    """
    print("\n" + "="*60)
    print("Azure AutoML 모델 로컬 예측 실행")
    print("="*60)

    # 1. 모델 로드
    model = load_model(MODEL_PATH)
    if model is None:
        return

    # 2. 예측할 데이터 준비 (예: zone_1)
    X_test = prepare_input_data(zone_id='zone_1')
    if X_test is None:
        return

    # 3. 예측 실행
    # AutoML 시계열 모델은 .forecast() 메소드를 사용합니다.
    # 이 메소드는 X_test 데이터의 마지막 시점 이후를 예측합니다.
    print("\n예측 실행 중...")
    try:
        # forecast()는 예측 결과와 함께 예측 시점의 데이터프레임을 반환합니다.
        y_predictions, X_trans = model.forecast(X_test)
        
        # 예측 결과에서 시간과 예측값만 추출
        # X_trans 에는 'time_index'라는 예측 시점 컬럼이 포함됩니다.
        forecast_df = X_trans[['time_index']].copy()
        forecast_df['prediction'] = y_predictions
        
        print("✅ 예측 성공!")
        print("\n" + "-"*60)
        print("30분 후 온도 예측 결과 (15분 간격 2개 스텝):")
        print(forecast_df)
        print("-"*60)

    except Exception as e:
        print(f"❌ 예측 중 오류 발생: {e}")
        print("입력 데이터의 형식이 모델이 기대하는 것과 다를 수 있습니다.")
        print("또는 Conda 환경이 올바르게 설정되지 않았을 수 있습니다.")


if __name__ == "__main__":
    main()

