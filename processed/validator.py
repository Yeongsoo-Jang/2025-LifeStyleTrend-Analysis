import pandas as pd

def validate_data(df: pd.DataFrame) -> bool:
    """데이터 품질 검증"""
    assert not df.empty, "데이터가 존재하지 않습니다"
    assert df.duplicated().sum() == 0, "중복 데이터 존재"
    assert df["ratio"].isnull().sum() == 0, "결측값 존재"
    print("✓ 데이터 검증 완료")
    return True


# 실행 예시
# validate_data(cleaned_df)
