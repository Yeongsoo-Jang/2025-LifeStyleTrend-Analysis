# modeling.visualizer.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast(historical_data: pd.Series, forecast_data: list, style_name: str, df: pd.DataFrame):
    """트렌드 예측 시각화"""
    plt.figure(figsize=(12,6))
    
    # 실제 데이터 가져오기 (동일한 그룹 필터링)
    style_data = df[df['group_name']==style_name]
    
    # 날짜와 값을 명시적으로 사용
    plt.plot(style_data['date'], historical_data.values, label='Historical')
    
    # 예측 데이터 날짜 생성
    last_date = style_data['date'].iloc[-1]
    
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=7),
        periods=len(forecast_data),
        freq='W'
    )
    
    plt.plot(forecast_dates, forecast_data, label='Forecast', linestyle='--')
    plt.title(f"{style_name} Trend Forecast")
    plt.xlabel("Date")
    plt.ylabel("Search Ratio")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

