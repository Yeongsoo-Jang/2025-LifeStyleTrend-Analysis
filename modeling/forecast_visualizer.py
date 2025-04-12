# modeling/forecast_visualizer.py
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Dict
from matplotlib.dates import DateFormatter, MonthLocator
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def _validate_inputs(decomposed: dict, forecasts: dict) -> None:
    """입력 데이터 검증"""
    if not isinstance(decomposed, dict) or not isinstance(forecasts, dict):
        raise TypeError("decomposed와 forecasts는 딕셔너리 형태여야 합니다.")
        
    common_styles = set(decomposed.keys()) & set(forecasts.keys())
    if not common_styles:
        raise ValueError("공통 스타일이 존재하지 않습니다.")

def _prepare_forecast_dates(last_date: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    """예측 날짜 생성"""
    return pd.date_range(
        start=last_date + pd.DateOffset(weeks=1),
        periods=periods,
        freq='W'
    )

def _create_standard_plots(style: str, decomposed: dict, forecast: pd.DataFrame) -> plt.Figure:
    """4분할 기본 시각화"""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    fig.suptitle(f'{style} 트렌드 분석 리포트', y=1.02, fontsize=16)
    
    # 1. 관측치 vs 트렌드
    axes[0].plot(decomposed['date'], decomposed['observed'], label='Observed', color='royalblue')
    axes[0].plot(decomposed['trend'], label='Trend', color='crimson')
    axes[0].set_title('원본 데이터 및 추세')
    axes[0].legend()
    
    # 2. 트렌드 + 예측
    axes[1].plot(decomposed['trend'], label='Historical Trend', color='crimson')
    axes[1].plot(forecast['date'], forecast['yhat'], '--', label='Forecast', color='darkorange')
    if 'yhat_lower' in forecast and 'yhat_upper' in forecast:
        axes[1].fill_between(forecast['date'], forecast['yhat_lower'], forecast['yhat_upper'], 
                           color='orange', alpha=0.2, label='신뢰구간')
    axes[1].set_title('추세 및 예측')
    axes[1].legend()
    
    # 3. 계절성
    axes[2].plot(decomposed['seasonal'], label='Seasonality', color='forestgreen')
    axes[2].set_title('계절성 패턴')
    
    # 4. 잔차
    axes[3].plot(decomposed['resid'], label='Residuals', color='purple')
    axes[3].axhline(0, color='gray', linestyle='--')
    axes[3].set_title('잔차 분석')
    
    return fig

def plot_forecasts(decomposed: dict, forecasts: dict, confidence: bool = True):
    """고도화된 시계열 시각화"""
    plt.style.use('seaborn-darkgrid')
    _validate_inputs(decomposed, forecasts)
    
    for style in set(decomposed.keys()) & set(forecasts.keys()):
        # 데이터 추출
        observed = decomposed[style]['observed']
        forecast = forecasts[style]
        
        # 날짜 생성
        forecast_dates = _prepare_forecast_dates(observed.index[-1], len(forecast))
        forecast['ds'] = forecast_dates  # 날짜 정보 강제 적용
        
        # 기본 플롯 생성
        fig = _create_standard_plots(style, decomposed[style], forecast)
        
        # 리포트 저장
        os.makedirs('modeling/reports', exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'modeling/reports/{style}_analysis.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # 추가 리포트 생성
        generate_html_report(style)
        create_trend_animation(style, decomposed[style], forecast)
        plot_3d_trend(style, decomposed[style])

def generate_html_report(style: str):
    """대시보드용 HTML 리포트 생성"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{style} 트렌드 리포트</title>
        <style>
            .report {{ margin: 20px; padding: 20px; border: 1px solid #ddd; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="report">
            <h2>{style} 트렌드 분석</h2>
            <img src="./{style}_analysis.png" alt="트렌드 분석 차트">
            <img src="./{style}_3d_trend.png" alt="3D 트렌드 차트">
        </div>
    </body>
    </html>
    """
    with open(f'modeling/reports/{style}_report.html', 'w') as f:
        f.write(html_content)

def create_trend_animation(style: str, decomposed: dict, forecast: pd.DataFrame):
    """시간 경과 애니메이션 생성"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    def animate(i):
        ax.clear()
        ax.plot(decomposed['trend'].iloc[:i], label='Historical Trend')
        if i > len(decomposed['trend']):
            offset = i - len(decomposed['trend'])
            ax.plot(forecast['ds'].iloc[:offset], forecast['yhat'].iloc[:offset], 
                   '--', label='Forecast')
        ax.set_title(f'{style} 트렌드 변화 ({i}주)')
        ax.legend()
    
    ani = FuncAnimation(fig, animate, frames=range(len(decomposed['trend']) + len(forecast)), 
                       interval=100)
    ani.save(f'modeling/reports/{style}_animation.gif', writer='imagemagick', dpi=100)
    plt.close()

def plot_3d_trend(style: str, decomposed: dict):
    """3D 트렌드 시각화"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 데이터 준비
    x = decomposed['trend'].index.to_julian_date()
    y = decomposed['seasonal'].values
    z = decomposed['trend'].values
    
    ax.plot(x, y, z, linewidth=2, color='blue')
    ax.set_xlabel('Time (Julian Date)')
    ax.set_ylabel('Seasonality')
    ax.set_zlabel('Trend Value')
    ax.set_title(f'{style} 3D Trend Analysis')
    
    plt.savefig(f'modeling/reports/{style}_3d_trend.png', bbox_inches='tight')
    plt.close()
