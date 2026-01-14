
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Configuration ---
st.set_page_config(
    page_title="이상 탐지 대시보드",
    page_icon="🚨",
    layout="wide",
)

# --- Data Loading ---
@st.cache_data
def load_data(filepath):
    """
    CSV 파일에서 이상 탐지 데이터를 로드하고, colDate를 datetime으로 변환합니다.
    """
    try:
        df = pd.read_csv(filepath)
        df['colDate'] = pd.to_datetime(df['colDate'])
        # contID가 숫자로 되어 있을 경우 'zone_' 접두사 추가
        if pd.api.types.is_numeric_dtype(df['contID']):
            df['contID'] = 'zone_' + df['contID'].astype(str)
        return df
    except FileNotFoundError:
        st.error(f"오류: '{filepath}' 파일을 찾을 수 없습니다. '03_train_anomaly_detector.py'를 먼저 실행했는지 확인하세요.")
        return None

# --- Main Application ---
def main():
    """
    Streamlit 대시보드의 메인 함수
    """
    st.title("🚨 컨테인먼트 이상 탐지 대시보드")
    st.markdown("---")

    # 데이터 로드
    data = load_data('cont_with_anomalies.csv')

    if data is not None:
        # --- Sidebar Filters ---
        st.sidebar.header("필터 설정")
        
        all_zones = sorted(data['contID'].unique())
        selected_zone = st.sidebar.selectbox(
            "컨테인먼트 존(Zone) 선택:",
            options=all_zones,
            index=0
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info(
            """
            **안내:**
            - **Hot Aisle 온도**: 실제 측정된 온도입니다.
            - **이상 탐지 지점**: Isolation Forest 모델이 비정상 패턴으로 판단한 시점입니다.
            """
        )

        # --- Main Panel ---
        filtered_data = data[data['contID'] == selected_zone].copy()

        st.header(f"'{selected_zone}' 이상 탐지 결과")

        # --- Anomaly Statistics ---
        total_points = len(filtered_data)
        anomaly_points = filtered_data['is_anomaly'].sum()
        anomaly_rate = (anomaly_points / total_points) * 100 if total_points > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("총 데이터 포인트", f"{total_points:,}")
        col2.metric("탐지된 이상치", f"{anomaly_points:,}")
        col3.metric("이상치 비율", f"{anomaly_rate:.2f}%")
        
        st.markdown("---")

        if not filtered_data.empty:
            # --- Time Series Chart with Anomalies ---
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # 1. Temperature Line
            fig.add_trace(
                go.Scatter(
                    x=filtered_data['colDate'],
                    y=filtered_data['tempHot'],
                    name='Hot Aisle 온도',
                    mode='lines',
                    line=dict(color='#1f77b4')
                ),
                secondary_y=False,
            )

            # 2. Anomaly Points
            anomalies = filtered_data[filtered_data['is_anomaly'] == 1]
            fig.add_trace(
                go.Scatter(
                    x=anomalies['colDate'],
                    y=anomalies['tempHot'],
                    name='이상 탐지 지점',
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='x')
                ),
                secondary_y=False,
            )
            
            # (Optional) 3. Anomaly Score Line
            fig.add_trace(
                go.Scatter(
                    x=filtered_data['colDate'],
                    y=filtered_data['anomaly_score'],
                    name='이상 점수',
                    mode='lines',
                    line=dict(color='orange', dash='dash')
                ),
                secondary_y=True,
            )

            # 차트 레이아웃 업데이트
            fig.update_layout(
                title=f"'{selected_zone}'의 온도 및 이상 탐지 지점",
                legend_title_text='범례',
                hovermode="x unified"
            )
            fig.update_xaxes(title_text="타임스탬프")
            fig.update_yaxes(title_text="온도 (°C)", secondary_y=False)
            fig.update_yaxes(title_text="이상 점수", secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)

            # --- Anomaly Data Table ---
            st.subheader("이상 탐지 데이터 상세")
            st.dataframe(
                anomalies[['colDate', 'contID', 'tempHot', 'humiHot', 'anomaly_score']].sort_values('anomaly_score'),
                use_container_width=True
            )
        else:
            st.warning("선택된 존에 대한 데이터가 없습니다.")

if __name__ == "__main__":
    main()
