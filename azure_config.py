"""
Azure ML Workspace 연결 설정
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# .env 파일 로드
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Workspace 정보 (환경 변수에서 로드)
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "rg-smartIoT-ml")
WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME", "ml-th-analysis")
LOCATION = os.getenv("AZURE_LOCATION", "Sweden Central")

def get_ml_client():
    """Azure ML Client 생성"""
    if not SUBSCRIPTION_ID:
        raise ValueError(
            "AZURE_SUBSCRIPTION_ID가 설정되지 않았습니다.\n"
            ".env 파일을 생성하고 구독 ID를 설정하세요.\n"
            ".env.example 파일을 참고하세요."
        )

    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME
        )
        print(f"[OK] Azure ML Workspace 연결 성공: {WORKSPACE_NAME}")
        print(f"     위치: {LOCATION}")
        print(f"     리소스 그룹: {RESOURCE_GROUP}")
        return ml_client
    except Exception as e:
        print(f"[ERROR] 연결 실패: {e}")
        print("\n해결 방법:")
        print("1. Azure CLI 로그인: az login")
        print(f"2. 올바른 구독 설정: az account set --subscription '{SUBSCRIPTION_ID}'")
        raise

if __name__ == "__main__":
    # 연결 테스트
    client = get_ml_client()
    print("\n워크스페이스 정보:")
    print(f"Storage: mlthanalysis1971033244")
    print(f"Key Vault: mlthanalysis8576821619")
    print(f"Application Insights: mlthanalysis8564505169")
