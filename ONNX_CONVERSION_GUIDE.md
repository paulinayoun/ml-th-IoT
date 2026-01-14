# ONNX 변환 가이드 (엣지 배포용)

## 1. Azure ML Studio에서 ONNX 변환

### 방법 A: Studio UI에서 변환
1. Azure ML Studio → Models → 해당 모델 선택
2. "Convert to ONNX" 또는 "Deploy" 옵션 확인
3. ONNX 형식으로 다운로드

### 방법 B: Python 스크립트로 변환 (Azure에서 실행)
```python
from azureml.core import Workspace, Model
from azureml.automl.runtime.onnx_convert import OnnxConverter

# Workspace 연결
ws = Workspace.from_config()

# 모델 다운로드
model = Model(ws, name='your_model_name')
model.download(target_dir='./model')

# ONNX 변환
onnx_model = OnnxConverter.convert_automl_to_onnx(
    model_path='./model',
    save_path='./model.onnx'
)
```

## 2. ONNX 모델로 예측 (엣지 환경)

### 설치 (매우 경량!)
```bash
pip install onnxruntime numpy pandas
```

### 예측 코드
```python
import onnxruntime as rt
import numpy as np
import pandas as pd

# ONNX 모델 로드
session = rt.InferenceSession("model.onnx")

# 데이터 준비
df = pd.read_csv("data.csv")
X = df[['tempHot', 'tempCold', 'humiHot', 'humiCold']].values

# 예측
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
predictions = session.run([output_name], {input_name: X.astype(np.float32)})[0]

print(predictions)
```

## 3. 리소스 비교

| 방법 | 패키지 수 | 설치 크기 | RAM 사용 | 속도 |
|------|----------|----------|---------|------|
| Azure ML 전체 | 200+ | 2-3GB | ~2GB | 느림 |
| ONNX Runtime | 3개 | ~100MB | ~500MB | 빠름 |

## 4. 엣지 배포 시나리오

```
[IoT 센서] → [데이터 수집] → [ONNX 모델 추론] → [결과 저장/전송]
           ↓
    [경량 Python 환경]
    - onnxruntime
    - numpy
    - pandas
```

## 5. 다음 단계

1. Azure ML Studio에서 ONNX 변환
2. model.onnx 파일 다운로드
3. 엣지 환경에 onnxruntime 설치
4. 예측 스크립트 실행
