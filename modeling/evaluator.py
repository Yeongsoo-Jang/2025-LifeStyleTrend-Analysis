# modeling/evaluator.py

from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import logging
logger = logging.getLogger(__name__)

def cross_validate(model, X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    return np.mean(scores)



def calculate_r2(actual: pd.Series, predicted: pd.Series) -> float:
    """R² 계산"""
    return r2_score(actual, predicted)

def calculate_trend_index(series: pd.Series) -> float:
    """선형 회귀를 사용한 트렌드 지수 계산"""
    x = range(len(series))
    slope, _, r_value, _, _ = linregress(x, series)
    return slope * r_value

def determine_trend_direction(trend_index: float) -> str:
    """트렌드 방향 결정"""
    if trend_index > 0:
        return "상승 추세"
    elif trend_index < 0:
        return "하락 추세"
    else:
        return "변동 없음"

def calculate_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """RMSE 계산"""
    return np.sqrt(mean_squared_error(actual, predicted))

def calculate_mape(actual: pd.Series, predicted: pd.Series) -> float:
    """NaN 방지 MAPE 계산"""
    # 인덱스 무시하고 배열로 변환
    actual_arr = actual.values
    predicted_arr = predicted.values
    
    mask = (actual_arr != 0)
    return np.mean(np.abs((actual_arr[mask] - predicted_arr[mask]) / actual_arr[mask])) * 100


def evaluate_forecasts(decomposed: dict, forecasts: dict) -> dict:
    """모델 성능 평가"""
    results = {}
    
    # 1. 그룹명 일치 검증
    common_styles = set(decomposed.keys()) & set(forecasts.keys())
    if not common_styles:
        raise ValueError("분해와 예측 그룹이 전혀 일치하지 않음")
    
    # 2. 실제 평가 가능한 그룹 필터링
    valid_styles = []
    for style in common_styles:
        valid_decomposed = (
            'trend' in decomposed[style] 
            and len(decomposed[style]['trend']) >= 26
        )
        valid_forecast = (
            'prophet' in forecasts[style] 
            and 'yhat' in forecasts[style]['prophet']
            and len(forecasts[style]['prophet']['yhat']) >= 26
        )
        
        if valid_decomposed and valid_forecast:
            valid_styles.append(style)
        else:
            logger.warning(f"[{style}] 검증 실패 - 분해: {valid_decomposed}, 예측: {valid_forecast}")
    
    # 3. 최종 검증
    if not valid_styles:
        available = {
            '분해 구조': {k: list(v.keys()) for k,v in decomposed.items()},
            '예측 구조': {k: list(v['prophet'].keys()) for k,v in forecasts.items()}
        }
        raise ValueError(
            f"평가 가능한 그룹 없음\n"
            f"분해 구조: {available['분해 구조']}\n"
            f"예측 구조: {available['예측 구조']}"
        )
    
    for style in common_styles:
        actual = decomposed[style]['trend'][-26:]  # 마지막 26주 (테스트 데이터)
        predicted = forecasts[style]['prophet']['yhat'][:26] # 예측 데이터
        
        # 데이터 길이 일치화 (최소 길이 기준)
        min_len = min(len(actual), len(predicted))
        actual = actual[-min_len:].reset_index(drop=True)  # 인덱스 재설정
        predicted = pd.Series(predicted[:min_len], name='yhat').reset_index(drop=True)
        
        # 트렌드 지수 및 방향 계산
        trend_index = calculate_trend_index(actual)
        trend_direction = determine_trend_direction(trend_index)
        
        # 성능 평가 지표 계산
        rmse = calculate_rmse(actual, predicted)
        mape = calculate_mape(actual, predicted)
        r2 = calculate_r2(actual, predicted)
        
        results[style] = {
            'trend_index': trend_index,
            'trend_direction': trend_direction,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
        
        # 앙상블 평가 추가
        if 'ensemble' in forecasts[style]:
            ensemble_actual = decomposed[style]['trend'][-26:]
            ensemble_pred = forecasts[style]['ensemble']
            
            # 데이터 길이 일치화
            min_len = min(len(ensemble_actual), len(ensemble_pred))
            ensemble_actual = ensemble_actual[-min_len:].reset_index(drop=True)
            ensemble_pred = pd.Series(ensemble_pred[:min_len]).reset_index(drop=True)
            
            ensemble_r2 = calculate_r2(ensemble_actual, ensemble_pred)
            results[style]['ensemble_r2'] = ensemble_r2  # 새로운 키 추가
    
    return results

