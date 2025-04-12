# modeling.prophet_model.py
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import pandas as pd
from typing import Dict, Any
import logging
logger = logging.getLogger(__name__)

# 로깅 설정
logging.getLogger('prophet').setLevel(logging.WARNING)

def validate_prophet(model: Prophet, df: pd.DataFrame) -> Dict[str, float]:
    """시간 순서 교차 검증"""
    try:
        # 1. 데이터 길이 검증
        if len(df) < 52:
            raise ValueError(f"교차 검증을 위한 충분한 데이터 없음 (필요: 52주, 현재: {len(df)}주)")
        
        # 2. 시간 단위 유효성 검증
        time_units = ['day']  # '364 days'에서 추출한 단위
        valid_units = ['day', 'hour', 'minute', 'second']
        if any(unit not in valid_units for unit in time_units):
            raise ValueError(f"잘못된 시간 단위: {time_units}")
            
        # 3. 교차 검증 실행
        df_cv = cross_validation(
            model,
            initial='728 days',
            period='91 days',
            horizon='182 days',
            parallel="processes"
        )
        
        metrics = performance_metrics(df_cv)
        return {
            'mape': metrics['mape'].mean(),
            'rmse': metrics['rmse'].mean(),
            'coverage': metrics['coverage'].mean()
        }
    except Exception as e:
        logging.error(f"검증 실패: {str(e)}")
        return {}

def analyze_changepoints(model: Prophet) -> pd.DataFrame:
    """트렌드 변화점 분석 (인덱스 변환 오류 해결)"""
    changepoints = pd.to_datetime(model.changepoints)
    
    # 변화점이 없거나 delta 파라미터 불일치 시 처리
    if len(changepoints) == 0 or len(model.params['delta']) != len(changepoints):
        return pd.DataFrame()
    
    # 디버깅 로그 추가
    logger.debug(f"변화점 수: {len(changepoints)}, delta 길이: {len(model.params['delta'])}")
    if len(changepoints) == 0:
        logger.warning("변화점 없음")
    
    return pd.DataFrame({
        'date': changepoints,
        'trend_change': model.params['delta'][:, 0]
    }).sort_values('trend_change', ascending=False)

def prophet_forecast(
    trend_series: pd.Series, 
    date_series: pd.Series, 
    periods: int = 26  # 기본값 설정
) -> Dict:
    
    # 날짜 컬럼 명시적 전달
    df = pd.DataFrame({
        'ds': date_series,
        'y': trend_series
    })
    
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df = df.dropna(subset=['ds'])
    
    # 데이터 품질 검증 추가
    if df['y'].std() < 0.01:
        raise ValueError("변동성 부족 (표준편차 < 0.01)")
    
    # 데이터 품질 검증
    if len(df) < 52:
        raise ValueError(f"데이터 부족 (필요: 52주, 현재: {len(df)}주)")
    
    # 모델 구성
    model = Prophet(
        changepoint_prior_scale=0.8,
        seasonality_prior_scale=8.0,
        changepoint_range=0.95,
        holidays_prior_scale=8.0,
        seasonality_mode='multiplicative',
        yearly_seasonality=6,
        weekly_seasonality=False,
        daily_seasonality=False,
        mcmc_samples=0
    )
    
    # 한국 특화 설정
    model.add_seasonality(name='monthly', period=30.5, fourier_order=8)
    model.add_country_holidays(country_name='KR')
    
    # 모델 훈련
    try:
        model.fit(df)
    except Exception as e:
        logging.error(f"모델 훈련 실패: {str(e)}")
        raise
    
    # 예측 생성
    future = model.make_future_dataframe(periods=periods, freq='W')
    forecast = model.predict(future)
    
    # 결과 포맷팅
    return {
        'yhat': forecast['yhat'].values,  
        'forecast_details': forecast[['ds', 'yhat_lower', 'yhat_upper']],
        'validation': validate_prophet(model, df),
        'changepoints': analyze_changepoints(model)
    }
