"""
MLflow를 사용한 Azure AutoML 모델 예측
"""
import pandas as pd
import mlflow

# 설정
MODEL_DIR = "models"  # MLflow 모델 디렉토리
DATA_PATH = "cont_forecast_clean/data.csv"

def load_model():
    """MLflow 모델 로드"""
    print(f"MLflow 모델 로딩 중: {MODEL_DIR}")
    try:
        model = mlflow.pyfunc.load_model(MODEL_DIR)
        print("✅ 모델 로딩 성공")
        return model
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def prepare_data(zone_id='zone_1'):
    """예측용 데이터 준비"""
    print(f"\n'{zone_id}' 데이터 준비 중...")

    # 전체 데이터 로드
    df = pd.read_csv(DATA_PATH, parse_dates=['colDate'])

    # 특정 zone만 필터링
    zone_df = df[df['contID'] == zone_id].copy()

    # target 컬럼 제거 (예측에는 불필요)
    if 'target_tempHot_30min' in zone_df.columns:
        zone_df = zone_df.drop(columns=['target_tempHot_30min'])

    print(f"✅ 데이터 준비 완료: {len(zone_df)}개 행")
    print(f"   시작: {zone_df['colDate'].min()}")
    print(f"   종료: {zone_df['colDate'].max()}")
    print(f"   컬럼: {list(zone_df.columns)}")

    return zone_df

def predict(model, data):
    """예측 실행"""
    print("\n예측 실행 중...")

    try:
        # MLflow 모델로 예측
        predictions = model.predict(data)

        print("✅ 예측 성공!")
        print(f"\n예측 결과 타입: {type(predictions)}")

        if isinstance(predictions, pd.DataFrame):
            print(f"예측 결과 shape: {predictions.shape}")
            print("\n예측 결과 (처음 10개):")
            print(predictions.head(10))

            # CSV로 저장
            output_file = "forecast_predictions.csv"
            predictions.to_csv(output_file, index=False)
            print(f"\n✅ 예측 결과 저장: {output_file}")
        else:
            print("\n예측 결과:")
            print(predictions[:10] if len(predictions) > 10 else predictions)

            # 배열인 경우 DataFrame으로 변환 후 저장
            result_df = pd.DataFrame({'prediction': predictions})
            output_file = "forecast_predictions.csv"
            result_df.to_csv(output_file, index=False)
            print(f"\n✅ 예측 결과 저장: {output_file}")

        return predictions

    except Exception as e:
        print(f"❌ 예측 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*60)
    print("MLflow AutoML 모델 예측 실행")
    print("="*60)

    # 1. 모델 로드
    model = load_model()
    if model is None:
        return

    # 2. 데이터 준비
    data = prepare_data(zone_id='zone_1')
    if data is None:
        return

    # 3. 예측 실행
    predictions = predict(model, data)

    print("\n" + "="*60)
    print("예측 완료!")
    print("="*60)

if __name__ == "__main__":
    main()
