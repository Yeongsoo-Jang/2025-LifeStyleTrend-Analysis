# arima_model.py
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

def train_arima(series, order=None):
    """ARIMA 모델 학습 (자동 파라미터 선택 추가)"""
    if order is None:
        # 자동으로 최적 파라미터 선택
        model = auto_arima(
            series,
            start_p=0, start_q=0, max_p=3, max_q=3, m=52,
            seasonal=True,  # 계절성 추가
            d=None,  # 차분 차수 자동 결정
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        order = model.order
        seasonal_order = model.seasonal_order
        print(f"최적 ARIMA 파라미터: {order}, 계절성: {seasonal_order}")
    
    # 최적 파라미터로 모델 구축
    model = ARIMA(series, order=order, seasonal_order=seasonal_order)
    return model.fit()
