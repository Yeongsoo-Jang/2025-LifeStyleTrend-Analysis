# modeling.stl_decomposer.py
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple
import logging
logger = logging.getLogger(__name__)

def _calculate_acf(resid: pd.Series, nlags: int = 10) -> float:
    """잔차 자기상관 함수 계산"""
    acf_values = acf(resid, nlags=nlags, fft=False)
    return np.mean(np.abs(acf_values[1:]))  # 0차 항 제외

def _check_stationarity(series: pd.Series, threshold: float = 0.05) -> bool:
    """ADF 검정을 통한 정상성 확인"""
    result = adfuller(series.dropna())
    return result[1] < threshold

def _decompose_group(args: Tuple[str, pd.DataFrame, int]) -> Dict:
    """그룹별 분해 병렬 처리"""
    group_name, group_df, period = args
    
    # 1. NaN/Inf 값 최종 정제
    group_df['ratio'] = group_df['ratio'].replace([np.inf, -np.inf], np.nan)
    group_df = group_df.dropna(subset=['ratio'])
    
    # 2. 변동성 검증 (표준편차 0.01 이상)
    if group_df['ratio'].std() < 0.01:
        logger.error(f"[{group_name}] 변동성 부족 (표준편차: {group_df['ratio'].std():.4f})")
        return {group_name: None}
    
    # 3. 상수값 검증
    if (group_df['ratio'] == group_df['ratio'].iloc[0]).all():
        logger.error(f"[{group_name}] 데이터 변화 없음 (상수값)")
        return {group_name: None}
    
    # 4. 주기 재계산 (52주 고정)
    period = 52  # 계절성 주기 고정
    
    # 5. STL 파라미터 조정
    try:
        decomposition = STL(
            group_df.set_index('date')['ratio'],
            period=period,
            robust=True,
            seasonal_deg=0  # 계절성 차수 조정
        ).fit()
        
        # 잔차 유효성 검증 추가
        if decomposition.resid.isnull().all():
            raise ValueError("잔차 데이터가 모두 NaN입니다.")
            
        metrics = _calculate_decomposition_metrics(decomposition.resid)
        acf_score = _calculate_acf(decomposition.resid)
        
        return {
            group_name: {
                'observed': decomposition.observed,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'resid': decomposition.resid,
                'metrics': metrics,
                'acf_score': acf_score
            }
        }
    except Exception as e:
        logger.error(f"[{group_name}] 분해 실패 상세: {str(e)}")
        logger.debug(f"데이터 통계:\n{group_df['ratio'].describe()}")
        logger.debug(f"샘플 데이터:\n{group_df.head()}")
        return {group_name: None}

def _calculate_decomposition_metrics(resid: pd.Series) -> Dict:
    return {
        'resid_mean': resid.mean(skipna=True),
        'resid_std': resid.std(skipna=True),
        'resid_skew': resid.skew(skipna=True),
        'resid_kurtosis': resid.kurtosis(skipna=True)
    }


def decompose_trend(df: pd.DataFrame, period: int = None) -> Dict:
    """
    STL 분해를 통해 계절성, 추세, 잔차 분리
    Returns:
        {
            'decompositions': {
                '그룹명': {
                    'observed': pd.Series,
                    'trend': pd.Series,
                    'seasonal': pd.Series,
                    'resid': pd.Series,
                    'metrics': dict,
                    'acf_score': float
                }
            },
            'quality_report': pd.DataFrame
        }
    """
    results = {}
    
    # 그룹별 병렬 처리 준비
    tasks = []
    for style in df['group_name'].unique():
        style_df = df[df['group_name'] == style].copy()
        
        # ▼▼▼ 로깅 추가 ▼▼▼
        logger.info(f"[{style}] 데이터 길이: {len(style_df)}주")
        logger.debug(f"데이터 통계:\n{style_df['ratio'].describe().to_string()}")
        
        # 주기 자동 계산 (최소 1년 주기 보장)
        auto_period = period if period else max(52, len(style_df)//2)
        tasks.append((style, style_df, auto_period))
    
    # 병렬 처리
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_decompose_group, task) for task in tasks]
        
        # 결과 수집
        for future in futures:
            try:
                result = future.result()
                if result is not None:
                    results.update(result)
            except Exception as e:
                logger.error(f"분해 오류: {str(e)}", exc_info=True)

    # 품질 보고서 생성
    quality_data = []
    for style, data in results.items():
        if data is not None:
            quality_data.append({
                'group': style,
                **data['metrics'],
                'acf_score': data['acf_score']
            })
    
    return {
        'decompositions': results,
        'quality_report': pd.DataFrame(quality_data) if quality_data else pd.DataFrame()
    }
