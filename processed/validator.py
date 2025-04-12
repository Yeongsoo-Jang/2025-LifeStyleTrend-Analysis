# validator.py
import pandas as pd

def validate_data(df: pd.DataFrame):
    # 필수 컬럼 검증
    assert {'date', 'group_name', 'ratio'} <= set(df.columns)
    
    # 날짜 범위 검증 (최소 2년)
    date_range = pd.to_datetime(df['date']).max() - pd.to_datetime(df['date']).min()
    assert date_range.days >= 730, f"데이터 범위 부족: {date_range.days}일"
    
    # 변동성 기준 필터링
    valid_groups = [g for g, d in df.groupby('group_name') if d['ratio'].std() > 0.01]
    return df[df['group_name'].isin(valid_groups)]

