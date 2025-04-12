import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # 'date' 컬럼 존재 여부 검증
    if 'date' not in df.columns:
        raise ValueError("'date' 컬럼이 존재하지 않습니다.")
    
    df = df.dropna(subset=["ratio"])
    df = df.drop_duplicates(subset=["date", "group_name"])
    
    # 그룹별로 결측치 재확인
    clean_df = pd.DataFrame()
    for group in df['group_name'].unique():
        group_df = df[df['group_name'] == group]
        
        # 2차 결측치 검증
        group_df = group_df.dropna(subset=["ratio"])
        
        # 이상치 처리 로직
        Q1 = group_df["ratio"].quantile(0.05)
        Q3 = group_df["ratio"].quantile(0.95)
        IQR = Q3 - Q1
        filtered_group = group_df[~((group_df["ratio"] < (Q1 - 3 * IQR)) | 
                                    (group_df["ratio"] > (Q3 + 3 * IQR)))]
        
        clean_df = pd.concat([clean_df, filtered_group], axis = 0)
    
    # 최종 NaN 체크
    assert clean_df['ratio'].isna().sum() == 0, "정제 후 NaN 값이 존재합니다."
    return clean_df

