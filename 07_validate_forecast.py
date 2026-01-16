# -*- coding: utf-8 -*-
"""
Azure AutoML Forecasting 모델 검증
7월 데이터로 학습한 모델로 8-9월을 예측하고 실제 값과 비교
"""
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 설정
MODEL_PATH = "models/model.pkl"
DATA_PATH = "./data/cont_validation.csv"  # cont_processed를 cont_forecast_clean 형식으로 변환한 데이터

def load_model():
    """모델 로드"""
    print("="*60)
    print("모델 로딩")
    print("="*60)
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[OK] 모델 로드 성공: {type(model)}")
        return model
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        return None

def prepare_data(train_end_date='2025-08-01', test_end_date='2025-08-15'):
    """
    데이터 준비
    - train_data: 7월 25일 ~ train_end_date (컨텍스트용)
    - test_data: train_end_date ~ test_end_date (검증용)
    """
    print("\n" + "="*60)
    print("데이터 준비")
    print("="*60)

    df = pd.read_csv(DATA_PATH, parse_dates=['colDate'])

    # 날짜 범위 확인
    print(f"전체 데이터 기간: {df['colDate'].min()} ~ {df['colDate'].max()}")
    print(f"전체 데이터 행: {len(df):,}")

    # Train/Test 분리
    train_end = pd.to_datetime(train_end_date)
    test_end = pd.to_datetime(test_end_date)

    train_df = df[df['colDate'] < train_end].copy()
    test_df = df[(df['colDate'] >= train_end) & (df['colDate'] < test_end)].copy()

    print(f"\n[Train 데이터]")
    print(f"  기간: {train_df['colDate'].min()} ~ {train_df['colDate'].max()}")
    print(f"  행수: {len(train_df):,}")

    print(f"\n[Test 데이터]")
    print(f"  기간: {test_df['colDate'].min()} ~ {test_df['colDate'].max()}")
    print(f"  행수: {len(test_df):,}")

    return train_df, test_df, df

def forecast_with_model(model, train_df, test_df, zone_id=1):
    """
    전체 contID로 forecasting 수행 후 특정 zone 필터링
    """
    print(f"\n" + "="*60)
    print(f"전체 contID 예측 (결과에서 contID={zone_id}만 추출)")
    print("="*60)

    print(f"Train 데이터: {len(train_df)} 행")
    print(f"Test 데이터: {len(test_df)} 행")

    try:
        # Azure AutoML forecasting 모델의 rolling_forecast 사용
        # 전체 contID를 포함한 데이터로 예측

        print("\n예측 실행 중 (rolling_forecast)...")

        # 전체 데이터 (train + test) - 모든 contID 포함
        full_data = pd.concat([train_df, test_df], ignore_index=True)

        # target 컬럼 이름 변경 (모델이 기대하는 이름으로)
        if 'target_tempHot_30min' in full_data.columns:
            full_data['_automl_target_col'] = full_data['target_tempHot_30min']

        # rolling_forecast: 학습 데이터 끝부터 테스트 끝까지 예측
        # X_pred: 전체 feature, y_pred: 전체 target
        if '_automl_target_col' in full_data.columns:
            y_full = full_data['_automl_target_col'].values
        else:
            y_full = np.zeros(len(full_data))  # 더미 값

        results_df = model.rolling_forecast(
            X_pred=full_data,
            y_pred=y_full,
            step=1,
            ignore_data_errors=True
        )

        print(f"[OK] 예측 완료")
        print(f"  결과 shape: {results_df.shape}")
        print(f"  컬럼: {list(results_df.columns)}")

        # 예측 결과 추출
        # rolling_forecast 결과 DataFrame에서 예측값 추출
        if 'predicted' in results_df.columns:
            predictions = results_df['predicted'].values
        elif model.forecast_column_name in results_df.columns:
            predictions = results_df[model.forecast_column_name].values
        else:
            print(f"[ERROR] 예측 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(results_df.columns)}")
            return None

        # 예측 결과를 full_data에 추가
        full_data['predicted'] = predictions[:len(full_data)]

        # 실제 값 추출
        if '_automl_target_col' in full_data.columns:
            full_data['actual'] = full_data['_automl_target_col']
        elif 'target_tempHot_30min' in full_data.columns:
            full_data['actual'] = full_data['target_tempHot_30min']
        else:
            full_data['actual'] = full_data['tempHot'].shift(-2)  # 30분 후 온도

        # 테스트 기간만 필터링
        test_results = full_data[full_data['colDate'] >= test_df['colDate'].min()].copy()

        # 특정 zone만 필터링
        zone_results = test_results[test_results['contID'] == zone_id].copy()

        if len(zone_results) == 0:
            print(f"[ERROR] contID={zone_id}의 예측 결과가 없습니다.")
            return None

        # 결과 데이터프레임 생성
        results = zone_results[['colDate', 'actual', 'predicted', 'contID']].copy()

        # NaN 제거
        results = results.dropna()

        # 오차 계산
        results['error'] = results['actual'] - results['predicted']
        results['abs_error'] = np.abs(results['error'])
        results['squared_error'] = results['error'] ** 2

        # 평가 지표
        mae = results['abs_error'].mean()
        rmse = np.sqrt(results['squared_error'].mean())
        mape = (results['abs_error'] / results['actual']).mean() * 100

        print(f"\n[평가 결과]")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")

        return results

    except Exception as e:
        print(f"[ERROR] 예측 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_results(results, zone_id):
    """결과 시각화"""
    if results is None or len(results) == 0:
        print("[WARNING] 표시할 결과가 없습니다.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 1. 실제 vs 예측
    ax1 = axes[0]
    ax1.plot(results['colDate'], results['actual'], label='실제 온도', linewidth=2, alpha=0.7)
    ax1.plot(results['colDate'], results['predicted'], label='예측 온도', linewidth=2, alpha=0.7)
    ax1.set_xlabel('날짜')
    ax1.set_ylabel('온도 (°C)')
    ax1.set_title(f'contID={zone_id} - 실제 vs 예측 온도')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 오차
    ax2 = axes[1]
    ax2.plot(results['colDate'], results['error'], label='오차 (실제 - 예측)', color='red', alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.fill_between(results['colDate'], 0, results['error'], alpha=0.3, color='red')
    ax2.set_xlabel('날짜')
    ax2.set_ylabel('오차 (°C)')
    ax2.set_title('예측 오차')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = f'forecast_validation_cont{zone_id}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[OK] 그래프 저장: {output_file}")
    plt.show()

def main():
    """메인 실행"""
    # 1. 모델 로드
    model = load_model()
    if model is None:
        return

    # 2. 데이터 준비 (7월 데이터를 컨텍스트로, 8월 초를 예측)
    train_df, test_df, full_df = prepare_data(
        train_end_date='2025-08-01',  # 7월 데이터
        test_end_date='2025-08-15'     # 8월 1~15일 예측
    )

    # 3. contID=1에 대해 예측 및 검증
    results = forecast_with_model(model, train_df, test_df, zone_id=1)

    # 4. 결과 저장
    if results is not None:
        output_file = 'forecast_validation_results.csv'
        results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n[OK] 결과 저장: {output_file}")

        # 5. 시각화
        plot_results(results, zone_id=1)

    print("\n" + "="*60)
    print("검증 완료!")
    print("="*60)

if __name__ == "__main__":
    main()
