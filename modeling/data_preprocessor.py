#modeling.data_preprocessor.py
import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """특성 공학"""
    df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')  # 숫자 변환 강화
    df = df.dropna(subset=['ratio'])  # 결측치 제거
    
    # 날짜 컬럼 명시적 생성
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # 월/년도/계절 추출
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['season'] = df['month'].apply(lambda x: (x % 12 + 3) // 3)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # 시차 특성(lag feature) 추가
    for lag in [1, 2, 4, 8]:
        df[f'ratio_lag_{lag}'] = df.groupby('group_name')['ratio'].shift(lag)
    
    # 롤링 통계 추가
    for window in [4, 8, 12]:
        df[f'ratio_ma_{window}'] = df.groupby('group_name')['ratio'].rolling(window).mean().values
    
    # 결측치 처리
    df['ratio'] = df['ratio'].replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().dropna()
    
    return df


def prepare_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """그룹별 주간 데이터 정규화"""
    # 날짜 형식 강제 변환
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'ratio'])
    
    # 주간 리샘플링 (그룹별 처리)
    return (
        df.set_index('date')
        .groupby('group_name')
        .resample('W')['ratio']
        .mean()
        .interpolate(method='linear')
        .reset_index()
        .rename(columns={'level_1': 'date'})
    )

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """변동성 기준 필터링 추가"""
    # 기존 정제 로직
    df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')
    df = df.dropna(subset=['ratio'])
    
    # 변동성 검증 (새로 추가된 부분)
    valid_groups = []
    for group, data in df.groupby('group_name'):
        if data['ratio'].std() > 0.01:  # 표준편차 0.01 이상 그룹만 선택
            valid_groups.append(group)
    
    return df[df['group_name'].isin(valid_groups)]

