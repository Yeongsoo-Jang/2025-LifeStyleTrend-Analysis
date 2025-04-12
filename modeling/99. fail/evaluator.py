from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model(true, pred):
    """모델 평가 메트릭 계산"""
    return {
        'MAE': mean_absolute_error(true, pred),
        'RMSE': np.sqrt(mean_squared_error(true, pred)),
        'MAPE': np.mean(np.abs((true - pred)/true))*100
    }
