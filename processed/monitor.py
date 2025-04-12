# monitor.py 
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np

def update_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    """실시간 업데이트 대시보드"""
    df = df.copy()  # 원본 데이터프레임을 복사하여 수정
    df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')  # 날짜 변환
    if df['date'].isnull().any():
        print("⚠️ 잘못된 날짜 형식 존재:", df[df['date'].isnull()])
    df = df.dropna(subset=['date'])  # 잘못된 날짜 제거
    df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')  # 숫자로 변환 (잘못된 값은 NaN 처리)
    df = df.dropna(subset=['ratio'])  # NaN 값 제거
    
    # 개별 그룹 확인
    groups = df['group_name'].unique()
    print(f"표시할 그룹: {groups}")
    
    # 그래프 생성
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 6))
    # 데이터를 복사하여 처리
    norm_df = df.copy()

    # 그룹별 정규화 (0-1 범위)
    for group in norm_df['group_name'].unique():
        mask = norm_df['group_name'] == group
        min_val = norm_df.loc[mask, 'ratio'].min()
        max_val = norm_df.loc[mask, 'ratio'].max()
        range_val = max_val - min_val
        norm_df.loc[mask, 'norm_ratio'] = (norm_df.loc[mask, 'ratio'] - min_val) / range_val if range_val > 0 else 0

    # 정규화된 값으로 그래프 생성
    sns.lineplot(
        data=norm_df,
        x='date',
        y='norm_ratio',  # 정규화된 값 사용
        hue='group_name',
        style='group_name',
        marker='o',
        ci=None,  # 신뢰구간 비활성화
    )

    # 그래프 제목 추가
    plt.title(f"Normalized Search Trends ({datetime.now().strftime('%Y-%m-%d %H:%M')})", fontsize=16)
    plt.ylabel("Normalized Ratio (0-1 Scale)", fontsize=12)  # y축 레이블 수정

    # 원본 값 툴팁 표시를 위한 주석 추가
    for group in groups:
        data = norm_df[norm_df['group_name'] == group].iloc[-1]
        plt.annotate(
            f"{data['ratio']:.2f}",
            xy=(data['date'], data['norm_ratio']),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=9
        )
    
    
    # x축 레이블 회전
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)

    # 범례 꾸미기
    plt.legend(title="Group Name", fontsize=10, title_fontsize=12, loc="upper right")
    
    # 그래프 출력
    plt.tight_layout()
    plt.savefig('modeling/reports/real_time_search_trends.png', dpi=300)  # 그래프 저장
    plt.show(block = True)  # 블록 모드로 그래프 출력
    
    return df
