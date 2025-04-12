# connector.connect.py
from connector.config_loader import initialize_naver_api, create_request_body
from processed.cleaner import clean_data
from processed.validator import validate_data
from processed.monitor import update_dashboard
import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()  # .env 파일 로드

def connect():
    """프로그램 진입점"""
    try:
        print("\n === Phase 1: 데이터 수집 및 변환 ===")
        # 설정 파일 로드 및 API 초기화
        naver = initialize_naver_api()
        request_body = create_request_body("config.yaml")
        
        # 데이터 수집 및 변환
        response = naver.send_request(request_body)
        raw_df = naver.to_dataframe(response)
        
        # 데이터 수집 및 변환 후
        print("\n === 그룹별 데이터 수 ===")
        print(raw_df.groupby('group_name').size())
        
        print("\n === raw_df 상위 5개 데이터 ===")
        print(raw_df.head())
        print("\n === raw_df 하위 5개 데이터 ===")
        print(raw_df.tail())
        
        # 데이터 정제 및 저장
        print("\n === 데이터 정제 및 저장 ===")
        
        cleaned_df = clean_data(raw_df)
        validate_data(cleaned_df)
        
        # 데이터 정제 후 검증
        print("\n === 정제된 데이터 구조 확인 ===")
        print("컬럼 목록:", cleaned_df.columns.tolist())
        print("데이터 타입:\n", cleaned_df.dtypes)
        print("결측치 개수:\n", cleaned_df.isnull().sum())
        print("데이터 샘플:")
        print(cleaned_df.head())
        
        
        print("\n === 실시간 대시보드 업데이트 ===")
        update_dashboard(cleaned_df)

        print("Phase 1 완료!")        
        
        return cleaned_df
        
    except Exception as e:
        import traceback
        print(f"\033[91m심각한 오류:\033[0m")
        print(traceback.format_exc())  # 상세 에러 스택 출력
