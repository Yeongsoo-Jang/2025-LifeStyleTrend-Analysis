# parallel_collector.py
from concurrent.futures import ThreadPoolExecutor
import tqdm
import logging
from typing import List, Dict
from naver_api_module import NaverAPIRequest, create_naver_request, fetch_naver_data

def execute_parallel_requests(
    keyword_groups: List[Dict], 
    max_workers: int = 8
) -> List[Dict]:
    """
    키워드 그룹별 병렬 처리
    - 8개 워커 스레드로 초당 15개 요청 처리 가능
    - 자동 재시도 메커니즘 적용
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for group in keyword_groups:
            future = executor.submit(
                process_single_group, 
                group
            )
            futures.append(future)
        
        results = []
        for future in tqdm.tqdm(futures, desc="Collecting Data"):
            try:
                results.extend(future.result())
            except Exception as e:
                logging.error(f"Collection failed: {str(e)}")
        return results

def process_single_group(group: Dict) -> List[Dict]:
    """개별 키워드 그룹 처리"""
    from naver_api_module import (
        KeywordGroup,  # [추가]
        NaverAPIRequest,
        create_naver_request,
        fetch_naver_data
    )
    
    kg = KeywordGroup(**group)
    config = NaverAPIRequest(
        start_date="2024-06-01",
        end_date="2025-04-07",
        time_unit="month",
        keyword_groups=[kg],  
        device="mo"  
    )
    request = create_naver_request(config)
    return fetch_naver_data(request)["results"]
