# naver_api_module.py (기능별 모듈 분리)
import os
import requests
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd

class KeywordGroup(BaseModel):
    groupName: str 
    keywords: List[str]

class NaverAPIRequest(BaseModel):
    start_date: str 
    end_date: str
    time_unit: str 
    keyword_groups: List[KeywordGroup]
    device: str = "pc"
    ages: List[str] = ["2", "3"]
    gender: str = "f"

def create_naver_request(config: NaverAPIRequest) -> Dict:
    """검색 요청 객체 생성"""
    return {
        "startDate": config.start_date,
        "endDate": config.end_date,
        "timeUnit": config.time_unit,
        "keywordGroups": [
            {"groupName": kg.groupName, "keywords": kg.keywords}
            for kg in config.keyword_groups 
        ],
        "device": config.device,
        "ages": config.ages,
        "gender": config.gender
    }

def fetch_naver_data(request_body: Dict) -> Dict:
    """API 데이터 수집"""
    headers = {
        "X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID"),
        "X-Naver-Client-Secret": os.getenv("NAVER_SECRET")
    }
    response = requests.post(
        "https://openapi.naver.com/v1/datalab/search",
        headers=headers,
        json=request_body
    )
    response.raise_for_status()
    return response.json()

def parse_to_dataframe(api_response: Dict) -> pd.DataFrame:
    """API 응답 → DataFrame 변환"""
    data_list = []
    
    for group in api_response.get('results', []):
        group_name = group.get('title', 'unknown_group')
        for data_point in group.get('data', []):
            data_list.append({
                "date": data_point.get('period', ''),
                "group": group_name,
                "ratio": float(data_point.get('ratio', 0))
            })
    
    return pd.DataFrame(data_list)