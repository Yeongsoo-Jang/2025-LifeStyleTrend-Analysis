import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
import requests
import ssl
import urllib
import certifi
import json

class Naver_API:
    def __init__(self, client_id: str, client_secret: str):
        """
        Naver DataLab API 초기화

        :param client_id: 네이버 API 클라이언트 ID
        :param client_secret: 네이버 API 클라이언트 Secret
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.url = "https://openapi.naver.com/v1/datalab/search"
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    def create_request_body(self, start_date: str, end_date: str, time_unit: str, keyword_groups: list, device: str, ages: list, gender: str) -> dict:
        """
        요청 데이터 생성

        :param start_date: 검색 시작 날짜 (YYYY-MM-DD 형식)
        :param end_date: 검색 종료 날짜 (YYYY-MM-DD 형식)
        :param time_unit: 데이터 집계 단위 ('date', 'week', 'month')
        :param keyword_groups: 키워드 그룹 리스트
        :param device: 검색 기기 ('pc', 'mobile')
        :param ages: 검색 연령대 리스트
        :param gender: 검색 성별 ('m', 'f')
        :return: 요청 데이터 (JSON 형식)
        """
        return {
            "startDate": start_date,
            "endDate": end_date,
            "timeUnit": time_unit,
            "keywordGroups": keyword_groups,
            "device": device,
            "ages": ages,
            "gender": gender
        }

    def send_request(self, body: dict) -> dict:
        """
        API 요청 및 응답 처리

        :param body: 요청 데이터 (JSON 형식)
        :return: API 응답 데이터 (JSON 형식)
        """
        # JSON 데이터를 문자열로 변환
        body = json.dumps(body)

        # 요청 객체 생성
        request = urllib.request.Request(self.url)
        request.add_header("X-Naver-Client-Id", self.client_id)
        request.add_header("X-Naver-Client-Secret", self.client_secret)
        request.add_header("Content-Type", "application/json")

        try:
            # API 요청
            response = urllib.request.urlopen(request, data=body.encode("utf-8"), context=self.ssl_context)
            rescode = response.getcode()

            if rescode == 200:
                # 성공적으로 응답을 받은 경우
                response_body = response.read()
                return json.loads(response_body.decode('utf-8'))
            else:
                # 에러 코드 반환 시 처리
                raise Exception(f"Error Code: {rescode}")
        
        except Exception as e:
            # 예외 처리
            raise Exception(f"API 요청 중 오류 발생: {e}")

    def to_dataframe(self, api_response: dict) -> pd.DataFrame:
        """
        API 응답 데이터를 pandas DataFrame으로 변환

        :param api_response: API 응답 데이터 (JSON 형식)
        :return: 변환된 pandas DataFrame
        """
        data_list = []

        # 각 키워드 그룹별로 데이터를 추출하여 리스트에 저장
        for group in api_response['results']:
            group_name = group['title']  # 키워드 그룹 이름
            for data in group['data']:
                data_list.append({
                    "date": data['period'],  # 날짜 (기간)
                    "group_name": group_name,  # 키워드 그룹 이름
                    "ratio": data['ratio']  # 검색 비율
                })

        df = pd.DataFrame(data_list)

        return df