# modeling/arima_model.py
from pmdarima import auto_arima
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
import warnings
import joblib
import logging
from typing import Tuple, Dict, Any
from modeling.evaluator import calculate_mape, calculate_rmse, calculate_r2

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)

class ARIMAModel(BaseEstimator):
    def __init__(self, 
                 seasonal: bool = True,
                 max_p: int = 3,
                 max_q: int = 3,
                 max_d: int = 1,
                 m: int = None,
                 stepwise: bool = True):
        self.seasonal = seasonal
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.m = m
        self.stepwise = stepwise
        self.model = None
        
    def _detect_seasonality(self, series: pd.Series) -> int:
        """FFT 기반 계절성 주기 감지"""
        from scipy.fftpack import fft
        fft_vals = np.abs(fft(series))
        non_zero = fft_vals[:len(series)//2] > 0
        if non_zero.any():
            dominant_freq = np.argmax(fft_vals[:len(series)//2])
            return max(1, len(series) // dominant_freq)
        return 1

    def fit(self, X: pd.Series, exog=None):
        if self.m is None:
            self.m = self._detect_seasonality(X)
            
        self.model = auto_arima(
            X,
            seasonal=self.seasonal,
            max_p=self.max_p,
            max_q=self.max_q,
            max_d=self.max_d,
            m=self.m,
            stepwise=self.stepwise,
            suppress_warnings=True,
            error_action='ignore',
            exogenous=exog
        )
        return self

    def predict(self, n_periods: int, exog=None) -> Tuple[pd.Series, np.ndarray]:
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        pred, conf_int = self.model.predict(
            n_periods=n_periods,
            exogenous=exog,
            return_conf_int=True
        )
        return pd.Series(pred, name='forecast'), conf_int
    
    def save(self, path: str):
        """모델 저장 전 검증"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        joblib.dump(self.model, path)
        
    @classmethod
    def load(cls, path: str):
        loaded = joblib.load(path)
        model = cls()
        model.model = loaded
        return model

# 외부 유틸리티 함수 -------------------------------------------------
def prepare_exogenous_features(df: pd.DataFrame) -> pd.DataFrame:
    """외생변수 전처리"""
    return pd.get_dummies(
        df[['month', 'season', 'ratio_ma_4', 'ratio_lag_1']],
        columns=['month', 'season']
    )

def ensemble_forecast(
    prophet_pred: pd.Series, 
    arima_pred: pd.Series,
    weights: Tuple[float, float] = (0.6, 0.4)
) -> pd.Series:
    """앙상블 예측"""
    return (
        prophet_pred * weights[0] 
        + arima_pred * weights[1]
    ).rename('ensemble_forecast')

def train_arima(series: pd.Series, n_periods: int = 26) -> tuple:
    """모델 객체와 예측값 반환"""
    model = auto_arima(
        series,
        seasonal=True,
        m=52,
        stepwise=True,
        suppress_warnings=True
    )
    forecast = model.predict(n_periods=n_periods)
    return model, forecast

def evaluate_arima(model, test_data: pd.Series) -> dict:
    """ARIMA 성능 평가"""
    pred = model.predict(n_periods=len(test_data))
    return {
        'mape': calculate_mape(test_data, pred),
        'rmse': calculate_rmse(test_data, pred),
        'r2': calculate_r2(test_data, pred)
    }
