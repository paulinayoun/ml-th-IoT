# -*- coding: utf-8 -*-
"""
Azure AutoML 모델 구조 및 설정 확인
"""
import joblib
import pandas as pd

MODEL_PATH = "models/model.pkl"

print("="*60)
print("모델 정보 확인")
print("="*60)

# 모델 로드
model = joblib.load(MODEL_PATH)

print(f"\n[모델 타입]")
print(f"  {type(model)}")

print(f"\n[모델 속성]")
attrs = [attr for attr in dir(model) if not attr.startswith('_')]
for i, attr in enumerate(attrs, 1):
    print(f"  {i:2d}. {attr}")

print(f"\n[Forecasting 설정]")
if hasattr(model, 'time_column_name'):
    print(f"  time_column_name: {model.time_column_name}")
if hasattr(model, 'grain_column_names'):
    print(f"  grain_column_names: {model.grain_column_names}")
if hasattr(model, 'forecast_horizon'):
    print(f"  forecast_horizon: {model.forecast_horizon}")
if hasattr(model, 'freq'):
    print(f"  frequency: {model.freq}")
if hasattr(model, 'target_column_name'):
    print(f"  target_column_name: {model.target_column_name}")

print(f"\n[주요 메서드]")
methods = ['fit', 'predict', 'forecast', 'forecast_quantiles', 'rolling_forecast']
for method in methods:
    has_it = hasattr(model, method)
    print(f"  {method}: {'있음' if has_it else '없음'}")

# forecast 메서드 시그니처 확인
if hasattr(model, 'forecast'):
    import inspect
    sig = inspect.signature(model.forecast)
    print(f"\n[forecast 메서드 시그니처]")
    print(f"  {sig}")

print("\n" + "="*60)
