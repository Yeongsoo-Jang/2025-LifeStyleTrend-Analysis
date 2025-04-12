# connector.config_loader.py
import yaml
from connector.naver_api import Naver_API
from typing import Dict, Any
import os
from dotenv import load_dotenv
load_dotenv()

def load_config(path: str) -> Dict[str, Any]:
    """YAML 설정 파일 로드"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def initialize_naver_api() -> Naver_API:
    """설정 기반 API 초기화"""
    
    # 클라이언트 인증
    naver_api = Naver_API(
        client_id=os.getenv("NAVER_CLIENT_ID"),
        client_secret=os.getenv("NAVER_SECRET")
    )
    
    return naver_api

def create_request_body(config_path: str) -> dict:
    """설정 기반 요청 본문 생성"""
    config = load_config(config_path)
    api_config = config['api_config']
    
    return {
        "startDate": api_config['start_date'],
        "endDate": api_config['end_date'],
        "timeUnit": api_config['time_unit'],
        "keywordGroups": [
            {"groupName": g['group_name'], "keywords": g['keywords']} 
            for g in config['keyword_groups']
        ],
        "device": api_config['device'],
        "ages": api_config['ages'],
        "gender": api_config['gender']
    }
