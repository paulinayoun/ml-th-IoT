"""
Joblib로 직접 모델을 로드하여 예측
(Azure ML 패키지 없이 시도)
"""
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# 설정
MODEL_PATH = "models/model.pkl"
DATA_PATH = "cont_forecast_clean/data.csv"

def load_model():
    """Joblib로 모델 직접 로드"""
    print(f"모델 로딩 중: {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ 모델 로딩 성공")
        print(f"   모델 타입: {type(model)}")
        print(f"   모델 속성: {dir(model)[:10]}...")
        return model
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def prepare_data(zone_id='zone_1', last_n_rows=100):
    """예측용 데이터 준비 (마지막 N개 행만)"""
    print(f"\n'{zone_id}' 데이터 준비 중 (마지막 {last_n_rows}개 행)...")

    # 전체 데이터 로드
    df = pd.read_csv(DATA_PATH, parse_dates=['colDate'])

    # 특정 zone만 필터링
    zone_df = df[df['contID'] == zone_id].copy()

    # 마지막 N개 행만 사용
    zone_df = zone_df.tail(last_n_rows)

    # target 컬럼 제거
    if 'target_tempHot_30min' in zone_df.columns:
        zone_df = zone_df.drop(columns=['target_tempHot_30min'])

    print(f"✅ 데이터 준비 완료: {len(zone_df)}개 행")
    print(f"   시작: {zone_df['colDate'].min()}")
    print(f"   종료: {zone_df['colDate'].max()}")

    return zone_df

def main():
    print("="*60)
    print("Joblib 직접 로드 방식 테스트")
    print("="*60)

    # 모델 로드
    model = load_model()
    if model is None:
        print("\n⚠️  azureml 패키지가 필요합니다.")
        print("\n해결 방법:")
        print("1. Azure ML Studio에서 직접 예측 실행")
        print("2. models/requirements.txt 패키지 전체 설치")
        print("3. Conda 환경 생성 (models/conda.yaml 사용)")
        return

    # 데이터 준비 (테스트용으로 100개 행만)
    data = prepare_data(zone_id='zone_1', last_n_rows=100)

    print("\n" + "="*60)
    print("모델 메서드 확인")
    print("="*60)

    # 사용 가능한 메서드 확인
    methods = [m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))]
    print(f"사용 가능한 메서드 ({len(methods)}개):")
    for i, method in enumerate(methods[:20]):  # 처음 20개만
        print(f"  {i+1}. {method}")
    if len(methods) > 20:
        print(f"  ... 외 {len(methods) - 20}개")

if __name__ == "__main__":
    main()
