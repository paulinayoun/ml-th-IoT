
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="온습도 분석 대시보드",
    page_icon="📊",
    layout="wide",
)

# --- Main Page Content ---
st.title("📊 온습도 예측 및 이상 탐지 대시보드")
st.markdown("---")

st.header("환영합니다!")

st.info(
    """
    이 대시보드는 컨테인먼트 및 랙의 온습도 데이터를 분석하고 예측하기 위한 도구입니다.
    
    **👈 왼쪽 사이드바에서 분석하고 싶은 대시보드를 선택하세요.**
    
    ### 사용 가능한 대시보드:
    
    **1. 🌡️ Forecast Dashboard**
    - 30분 후의 Hot Aisle 온도를 예측한 결과를 시계열 차트로 보여줍니다.
    - 컨테인먼트 존(Zone)별로 필터링하여 볼 수 있습니다.
    
    **2. 🚨 Anomaly Dashboard**
    - Isolation Forest 모델을 사용하여 탐지된 온습도 이상 패턴을 시각화합니다.
    - 이상치로 탐지된 지점을 차트 위에서 확인할 수 있습니다.
    
    ---
    
    **데이터 소스:**
    - 예측 대시보드: `cont_forecast_data.csv`
    - 이상 탐지 대시보드: `cont_with_anomalies.csv`
    """
)

st.markdown(
    """
    **실행 가이드:**
    - 각 대시보드에 필요한 데이터 파일이 생성되었는지 확인하세요.
    - 문제가 발생하면 `02_train_forecast_model.py` 또는 `03_train_anomaly_detector.py` 스크립트를 다시 실행해야 할 수 있습니다.
    """
)
