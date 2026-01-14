
import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="온도 예측 대시보드",
    page_icon="🌡️",
    layout="wide",
)

# --- Data Loading ---
@st.cache_data
def load_data(filepath):
    """
    CSV 파일에서 데이터를 로드하고, colDate를 datetime으로 변환합니다.
    """
    try:
        df = pd.read_csv(filepath)
        df['colDate'] = pd.to_datetime(df['colDate'])
        return df
    except FileNotFoundError:
        st.error(f"오류: '{filepath}' 파일을 찾을 수 없습니다. '02_train_forecast_model.py'를 먼저 실행했는지 확인하세요.")
        return None

# --- Main Application ---
def main():
    """
    Streamlit 대시보드의 메인 함수
    """
    st.title("🌡️ 컨테인먼트 온도 예측 대시보드")
    st.markdown("---")

    # 데이터 로드
    data = load_data('cont_forecast_data.csv')

    if data is not None:
        # --- Sidebar Filters ---
        st.sidebar.header("필터 설정")
        
        # Zone 선택
        all_zones = data['contID'].unique()
        selected_zone = st.sidebar.selectbox(
            "컨테인먼트 존(Zone) 선택:",
            options=all_zones,
            index=0
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info(
            """
            **안내:**
            - **현재 온도 (tempHot)**: 실제 측정된 Hot Aisle 온도입니다.
            - **30분 후 예측 (target_tempHot_30min)**: 모델이 예측한 30분 뒤의 온도입니다.
            """
        )

        # --- Main Panel ---
        # 선택된 Zone에 해당하는 데이터 필터링
        filtered_data = data[data['contID'] == selected_zone]

        st.header(f"'{selected_zone}' 온도 변화 추이")
        
        if not filtered_data.empty:
            # --- Time Series Chart ---
            fig = px.line(
                filtered_data,
                x='colDate',
                y=['tempHot', 'target_tempHot_30min'],
                title=f"'{selected_zone}'의 현재 온도 및 30분 후 예측",
                labels={
                    'colDate': '시간',
                    'value': '온도 (°C)',
                    'variable': '측정 항목'
                },
                color_discrete_map={
                    'tempHot': '#1f77b4',  # Muted Blue
                    'target_tempHot_30min': '#ff7f0e'  # Safety Orange
                }
            )

            # 차트 레이아웃 업데이트
            fig.update_layout(
                legend_title_text='범례',
                xaxis_title="타임스탬프",
                yaxis_title="온도 (°C)",
                hovermode="x unified"
            )
            
            # 범례 이름 변경
            new_names = {
                'tempHot': '현재 온도', 
                'target_tempHot_30min': '30분 후 예측'
            }
            fig.for_each_trace(lambda t: t.update(name = new_names[t.name]))


            st.plotly_chart(fig, use_container_width=True)

            # --- Data Table ---
            st.subheader("상세 데이터 보기")
            st.dataframe(
                filtered_data[['colDate', 'contID', 'tempHot', 'target_tempHot_30min']].tail(100),
                use_container_width=True
            )
        else:
            st.warning("선택된 존에 대한 데이터가 없습니다.")

if __name__ == "__main__":
    main()
