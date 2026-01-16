# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="온도 예측 대시보드",
    page_icon="🌡️",
    layout="wide",
)

# --- Pretendard 폰트 적용 ---
st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');

* {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif !important;
}

html, body, [class*="css"] {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# --- 설정값 ---
TEMP_THRESHOLD = 32.0  # 온도 임계값 (°C)
WARNING_DELTA = 0.5    # 경고 온도 델타 (°C)

# --- Data Loading ---
@st.cache_data
def load_data(filepath):
    """CSV 파일에서 데이터를 로드하고, colDate를 datetime으로 변환"""
    try:
        df = pd.read_csv(filepath)
        df['colDate'] = pd.to_datetime(df['colDate'])
        return df
    except FileNotFoundError:
        st.error(f"오류: '{filepath}' 파일을 찾을 수 없습니다.")
        return None

def calculate_metrics(df, zone_id):
    """KPI 지표 계산"""
    zone_data = df[df['contID'] == zone_id].copy()

    if len(zone_data) == 0:
        return None

    # 최신 데이터
    latest = zone_data.iloc[-1]

    # 현재 온도
    current_temp = latest['tempHot']

    # 30분 후 예측 온도
    predicted_temp = latest['target_tempHot_30min']

    # 온도 변화
    temp_delta = predicted_temp - current_temp

    # 최근 10분간 온도 변화 (마지막 1개 측정값과 그 전 비교)
    if len(zone_data) >= 2:
        recent_delta = zone_data.iloc[-1]['tempHot'] - zone_data.iloc[-2]['tempHot']
    else:
        recent_delta = 0

    # 예측 정확도 계산 (실제값 존재 시)
    # 이전 예측과 현재 실제값 비교
    if len(zone_data) >= 3:
        # 30분 전 예측값 (3 step 전)
        past_predicted = zone_data.iloc[-4]['target_tempHot_30min'] if len(zone_data) >= 4 else None
        past_actual = zone_data.iloc[-1]['tempHot']

        if past_predicted is not None:
            mae = abs(past_actual - past_predicted)
        else:
            mae = None
    else:
        mae = None

    # 경고 상태
    warning_status = "정상"
    warning_count = 0

    if predicted_temp >= TEMP_THRESHOLD:
        warning_status = "경고"
        warning_count += 1

    if temp_delta >= WARNING_DELTA:
        warning_status = "주의" if warning_status == "정상" else "경고"
        warning_count += 1

    return {
        'current_temp': current_temp,
        'predicted_temp': predicted_temp,
        'temp_delta': temp_delta,
        'recent_delta': recent_delta,
        'mae': mae,
        'warning_status': warning_status,
        'warning_count': warning_count,
        'latest_time': latest['colDate']
    }

def render_all_zones_kpi(all_zones_metrics, threshold):
    """모든 Zone의 KPI를 한 번에 표시"""

    st.markdown("#### 📊 전체 Zone 핵심 지표")

    # Zone별로 카드 생성 (4열)
    cols = st.columns(4)

    for idx, (zone_id, metrics) in enumerate(all_zones_metrics.items()):
        if metrics is None:
            continue

        with cols[idx % 4]:
            # 경고 상태에 따른 색상
            if metrics['warning_status'] == "경고":
                border_color = "#ff4b4b"  # 빨강
                status_emoji = "🚨"
            elif metrics['warning_status'] == "주의":
                border_color = "#ffa500"  # 주황
                status_emoji = "⚠️"
            else:
                border_color = "#00cc66"  # 초록
                status_emoji = "✅"

            # 전체 카드를 하나의 컨테이너로
            st.markdown(f"""
            <div style="
                border: 4px solid {border_color};
                border-radius: 10px;
                padding: 12px;
                margin-bottom: 10px;
                background-color: rgba(255,255,255,0.05);
            ">
                <div style="text-align: center; font-size: 0.9em; font-weight: bold; margin-bottom: 8px;">
                    {status_emoji} Zone {zone_id}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 2x2 그리드 구성
            # 첫 번째 행: 현재 온도 | 30분 후
            row1_col1, row1_col2 = st.columns(2)

            with row1_col1:
                st.metric(
                    label="현재",
                    value=f"{metrics['current_temp']:.1f}°C",
                    delta=f"{metrics['recent_delta']:+.1f}",
                    delta_color="inverse",
                    label_visibility="visible"
                )

            with row1_col2:
                delta_color = "off" if abs(metrics['temp_delta']) < 0.1 else ("inverse" if metrics['temp_delta'] > 0 else "normal")
                st.metric(
                    label="30분",
                    value=f"{metrics['predicted_temp']:.1f}°C",
                    delta=f"{metrics['temp_delta']:+.1f}",
                    delta_color=delta_color,
                    label_visibility="visible"
                )

            # 두 번째 행: MAE | 시간
            row2_col1, row2_col2 = st.columns(2)

            with row2_col1:
                if metrics['mae'] is not None:
                    mae_val = f"{metrics['mae']:.2f}°C"
                    mae_status = "우수" if metrics['mae'] < 0.5 else "양호"
                else:
                    mae_val = "-"
                    mae_status = ""

                st.metric(
                    label="MAE",
                    value=mae_val,
                    delta=mae_status if mae_status else None,
                    delta_color="normal" if mae_status == "우수" else "off",
                    label_visibility="visible"
                )

            with row2_col2:
                st.metric(
                    label="갱신",
                    value=f"{metrics['latest_time']:%H:%M}",
                    delta=None,
                    label_visibility="visible"
                )

    st.markdown("---")

def render_alert_banner(all_zones_metrics, threshold):
    """실시간 알림 배너 렌더링"""
    alerts = []

    for zone_id, metrics in all_zones_metrics.items():
        if metrics is None:
            continue

        # 임계값 초과 예상
        if metrics['predicted_temp'] >= threshold:
            alerts.append({
                'level': 'warning',
                'zone': zone_id,
                'message': f"Zone {zone_id}: 30분 후 {metrics['predicted_temp']:.1f}°C 예상 (임계값 {threshold}°C 초과 예정)",
                'action': "냉각 시스템 점검 권장"
            })

        # 급격한 온도 상승
        if metrics['temp_delta'] >= WARNING_DELTA:
            alerts.append({
                'level': 'warning',
                'zone': zone_id,
                'message': f"Zone {zone_id}: 30분간 {metrics['temp_delta']:+.1f}°C 상승 예상",
                'action': "서버 부하 확인 필요"
            })

    if len(alerts) == 0:
        st.success(f"✅ [{datetime.now():%H:%M}] 모든 존 정상 범위 유지 중")
    else:
        for alert in alerts:
            if alert['level'] == 'warning':
                st.warning(f"⚠️ [{datetime.now():%H:%M}] {alert['message']}\n   → {alert['action']}")
            elif alert['level'] == 'error':
                st.error(f"🔥 [{datetime.now():%H:%M}] {alert['message']}\n   → {alert['action']}")

def create_main_chart(filtered_data, zone_id):
    """메인 차트 생성 (실제 vs 예측 with 신뢰구간)"""

    # 서브플롯 생성
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Zone {zone_id} - 실제 vs 예측 온도',
            f'Zone {zone_id} - 예측 오차'
        ),
        vertical_spacing=0.12,
        row_heights=[0.7, 0.3]
    )

    # 1. 실제 온도
    fig.add_trace(
        go.Scatter(
            x=filtered_data['colDate'],
            y=filtered_data['tempHot'],
            name='실제 온도',
            line=dict(color='#1f77b4', width=2),
            mode='lines'
        ),
        row=1, col=1
    )

    # 2. 예측 온도
    fig.add_trace(
        go.Scatter(
            x=filtered_data['colDate'],
            y=filtered_data['target_tempHot_30min'],
            name='30분 후 예측',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            mode='lines'
        ),
        row=1, col=1
    )

    # 3. 임계값 라인
    fig.add_hline(
        y=TEMP_THRESHOLD,
        line_dash="dot",
        line_color="red",
        annotation_text=f"임계값 ({TEMP_THRESHOLD}°C)",
        annotation_position="right",
        row=1, col=1
    )

    # 4. 오차 계산 및 표시
    # 30분 전 예측과 현재 실제값 비교
    filtered_data['error'] = filtered_data['tempHot'] - filtered_data['target_tempHot_30min'].shift(3)

    # 오차 막대 그래프
    colors = ['green' if abs(e) < 0.5 else ('orange' if abs(e) < 1.0 else 'red')
              for e in filtered_data['error'].fillna(0)]

    fig.add_trace(
        go.Bar(
            x=filtered_data['colDate'],
            y=filtered_data['error'],
            name='예측 오차',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )

    # 오차 0 라인
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, row=2, col=1)

    # 레이아웃 업데이트
    fig.update_xaxes(title_text="시간", row=2, col=1)
    fig.update_yaxes(title_text="온도 (°C)", row=1, col=1)
    fig.update_yaxes(title_text="오차 (°C)", row=2, col=1)

    fig.update_layout(
        height=700,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def create_all_zones_chart(data, threshold):
    """모든 Zone을 4분할로 표시하는 차트"""

    all_zones = sorted(data['contID'].unique())

    # 2x2 그리드 생성
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'Zone {zone}' for zone in all_zones],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, zone_id in enumerate(all_zones):
        if idx >= 4:  # 최대 4개 Zone만 표시
            break

        row, col = positions[idx]
        zone_data = data[data['contID'] == zone_id].copy()

        if len(zone_data) == 0:
            continue

        # 실제 온도
        fig.add_trace(
            go.Scatter(
                x=zone_data['colDate'],
                y=zone_data['tempHot'],
                name=f'Zone {zone_id} 실제',
                line=dict(color='#1f77b4', width=2),
                mode='lines',
                showlegend=(idx == 0)  # 첫 번째만 범례 표시
            ),
            row=row, col=col
        )

        # 예측 온도
        fig.add_trace(
            go.Scatter(
                x=zone_data['colDate'],
                y=zone_data['target_tempHot_30min'],
                name=f'Zone {zone_id} 예측',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                mode='lines',
                showlegend=(idx == 0)  # 첫 번째만 범례 표시
            ),
            row=row, col=col
        )

        # 임계값 라인
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color="red",
            line_width=1,
            row=row, col=col
        )

        # Y축 범위 설정 (모든 차트 동일 범위)
        fig.update_yaxes(title_text="온도 (°C)", row=row, col=col)
        fig.update_xaxes(title_text="시간", row=row, col=col)

    fig.update_layout(
        height=800,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        title_text="📊 전체 Zone 실시간 모니터링",
        title_x=0.5
    )

    return fig

def create_zone_comparison_chart(data):
    """존별 비교 차트"""

    # 최신 데이터만
    latest_data = data.groupby('contID').tail(1)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('존별 현재 온도', '존별 30분 후 예측'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # 현재 온도
    colors_current = ['red' if t >= TEMP_THRESHOLD else 'orange' if t >= TEMP_THRESHOLD - 1 else 'green'
                      for t in latest_data['tempHot']]

    fig.add_trace(
        go.Bar(
            x=latest_data['contID'].astype(str),
            y=latest_data['tempHot'],
            name='현재 온도',
            marker_color=colors_current,
            text=latest_data['tempHot'].round(2),
            textposition='outside',
            showlegend=False
        ),
        row=1, col=1
    )

    # 예측 온도
    colors_predicted = ['red' if t >= TEMP_THRESHOLD else 'orange' if t >= TEMP_THRESHOLD - 1 else 'lightblue'
                       for t in latest_data['target_tempHot_30min']]

    fig.add_trace(
        go.Bar(
            x=latest_data['contID'].astype(str),
            y=latest_data['target_tempHot_30min'],
            name='예측 온도',
            marker_color=colors_predicted,
            text=latest_data['target_tempHot_30min'].round(2),
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )

    # 임계값 라인
    fig.add_hline(y=TEMP_THRESHOLD, line_dash="dash", line_color="red",
                  annotation_text=f"임계값", row=1, col=1)
    fig.add_hline(y=TEMP_THRESHOLD, line_dash="dash", line_color="red",
                  annotation_text=f"임계값", row=1, col=2)

    fig.update_xaxes(title_text="Zone", row=1, col=1)
    fig.update_xaxes(title_text="Zone", row=1, col=2)
    fig.update_yaxes(title_text="온도 (°C)", row=1, col=1)
    fig.update_yaxes(title_text="온도 (°C)", row=1, col=2)

    fig.update_layout(height=400)

    return fig

# --- Main Application ---
def main():
    # 데이터 로드
    data = load_data('cont_forecast_data.csv')

    if data is None:
        st.info("💡 **안내**: 'cont_forecast_data.csv' 파일이 필요합니다. '02_train_forecast_model.py'를 실행하여 생성하세요.")
        return

    # --- Sidebar ---
    st.sidebar.header("⚙️ 설정")

    # Zone 목록
    all_zones = sorted(data['contID'].unique())

    # 임계값 설정
    threshold = st.sidebar.slider(
        "온도 임계값 (°C)",
        min_value=28.0,
        max_value=35.0,
        value=TEMP_THRESHOLD,
        step=0.5,
        help="경고 알림 기준 온도입니다"
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **📊 대시보드 구성**
        - **전체 Zone KPI**: 4개 Zone 핵심 지표 한눈에 확인
        - **4분할 차트**: 실시간 온도 추이 동시 모니터링
        - **상세 분석**: Zone별 오차 분석

        **📈 지표 설명**
        - **현재 온도**: 최신 측정 온도
        - **30분 후**: 모델이 예측한 30분 뒤 온도
        - **MAE**: 평균 절대 오차 (낮을수록 우수)

        **🚨 경고 상태**
        - 🚨 빨강: 임계값 초과 또는 급격한 변화
        - ⚠️ 주황: 주의 필요
        - ✅ 초록: 정상
        """
    )

    # --- 타이틀 + 날짜 선택 (같은 줄) ---
    min_date = data['colDate'].min().date()
    max_date = data['colDate'].max().date()

    col_title, col_date_start, col_date_end, col_date_btn = st.columns([3, 2, 2, 1])

    with col_title:
        st.markdown("### 🌡️ 온도 예측 & 모니터링 대시보드")

    with col_date_start:
        start_date = st.date_input(
            "시작일",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key='start_date',
            label_visibility="visible"
        )

    with col_date_end:
        end_date = st.date_input(
            "종료일",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key='end_date',
            label_visibility="visible"
        )

    with col_date_btn:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("전체", use_container_width=True):
            st.session_state.start_date = min_date
            st.session_state.end_date = max_date
            st.rerun()

    # 날짜 필터링
    data = data[
        (data['colDate'].dt.date >= start_date) &
        (data['colDate'].dt.date <= end_date)
    ]

    st.caption(f"📊 {start_date} ~ {end_date} ({(end_date - start_date).days + 1}일)")
    st.markdown("---")

    # --- KPI Cards (전체 Zone) ---
    # 모든 존의 metrics 계산
    all_zones_metrics = {}
    for zone_id in all_zones:
        metrics = calculate_metrics(data, zone_id)
        all_zones_metrics[zone_id] = metrics

    # 모든 Zone의 KPI 카드 표시
    render_all_zones_kpi(all_zones_metrics, threshold)

    # --- Alert Banner ---
    st.markdown("#### 🔔 실시간 알림")
    render_alert_banner(all_zones_metrics, threshold)

    st.markdown("---")

    # --- Main Charts (4분할 - 모든 Zone 동시 표시) ---
    st.markdown("#### 📈 전체 Zone 실시간 모니터링")

    all_zones_fig = create_all_zones_chart(data, threshold)
    st.plotly_chart(all_zones_fig, width="stretch")

    st.markdown("---")

    # --- Zone 상세 분석 (오차 포함) ---
    with st.expander("🔍 Zone 상세 분석 (오차 포함)", expanded=False):
        st.markdown("**특정 Zone의 예측 오차를 상세히 분석합니다.**")

        # Zone 선택
        detail_zone = st.selectbox(
            "분석할 Zone 선택",
            options=all_zones,
            index=0,
            key='detail_zone_selector',
            help="상세 오차 분석을 볼 Zone을 선택하세요"
        )

        st.markdown("---")

        filtered_data = data[data['contID'] == detail_zone].copy()

        if not filtered_data.empty:
            # 상세 차트 (오차 포함)
            detail_fig = create_main_chart(filtered_data, detail_zone)
            st.plotly_chart(detail_fig, width="stretch")

            # 오차 통계
            filtered_data['error'] = filtered_data['tempHot'] - filtered_data['target_tempHot_30min'].shift(3)
            errors = filtered_data['error'].dropna()

            if len(errors) > 0:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("평균 오차", f"{errors.mean():.3f}°C")
                col2.metric("MAE", f"{errors.abs().mean():.3f}°C")
                col3.metric("최대 오차", f"{errors.abs().max():.3f}°C")
                col4.metric("표준편차", f"{errors.std():.3f}°C")
        else:
            st.warning("선택된 존에 대한 데이터가 없습니다.")

    st.markdown("---")

    # --- Zone Comparison ---
    st.markdown("#### 🏢 전체 Zone 비교")
    comparison_fig = create_zone_comparison_chart(data)
    st.plotly_chart(comparison_fig, width="stretch")

    st.markdown("---")

    # --- Data Table ---
    with st.expander("📋 상세 데이터 보기 (최근 100개)"):
        # 데이터 테이블용 Zone 선택
        table_zone = st.selectbox(
            "데이터를 볼 Zone 선택",
            options=all_zones,
            index=0,
            key='table_zone_selector'
        )

        table_data = data[data['contID'] == table_zone].copy()

        display_columns = ['colDate', 'contID', 'tempHot', 'target_tempHot_30min']
        # tempCold 컬럼이 있으면 추가
        if 'tempCold' in table_data.columns:
            display_columns.insert(3, 'tempCold')

        st.dataframe(
            table_data[display_columns].tail(100)
        )

    # --- Statistics ---
    with st.expander("📊 통계 정보"):
        # 통계용 Zone 선택
        stats_zone = st.selectbox(
            "통계를 볼 Zone 선택",
            options=all_zones,
            index=0,
            key='stats_zone_selector'
        )

        stats_data = data[data['contID'] == stats_zone].copy()

        if not stats_data.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Zone {stats_zone} 실제 온도 통계**")
                st.write(f"- 평균 온도: {stats_data['tempHot'].mean():.2f}°C")
                st.write(f"- 최고 온도: {stats_data['tempHot'].max():.2f}°C")
                st.write(f"- 최저 온도: {stats_data['tempHot'].min():.2f}°C")
                st.write(f"- 표준편차: {stats_data['tempHot'].std():.2f}°C")

            with col2:
                st.write("**예측 온도 통계**")
                st.write(f"- 평균 예측: {stats_data['target_tempHot_30min'].mean():.2f}°C")
                st.write(f"- 최고 예측: {stats_data['target_tempHot_30min'].max():.2f}°C")
                st.write(f"- 최저 예측: {stats_data['target_tempHot_30min'].min():.2f}°C")

                # 오차 통계
                stats_data['error'] = stats_data['tempHot'] - stats_data['target_tempHot_30min'].shift(3)
                errors = stats_data['error'].dropna()
                if len(errors) > 0:
                    st.write(f"- 평균 오차: {errors.mean():.2f}°C")
                    st.write(f"- MAE: {errors.abs().mean():.2f}°C")
        else:
            st.warning("선택된 존에 대한 데이터가 없습니다.")

if __name__ == "__main__":
    main()
