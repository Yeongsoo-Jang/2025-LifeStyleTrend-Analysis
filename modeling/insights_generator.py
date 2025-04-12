# modeling/insights_generator.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy.stats import linregress
import os
from typing import Dict, Any
import seaborn as sns

def _calculate_trend_metrics(series: pd.Series) -> Dict[str, Any]:
    """트렌드 메트릭 계산"""
    x = np.arange(len(series))
    slope, intercept, r_value, _, _ = linregress(x, series.values)
    return {
        'growth_rate': slope * 100,  # 주간 성장률(%)
        'r_squared': max(r_value**2, 0),  # 결정계수(R²)
        'current_value': series.iloc[-1],
        'baseline': intercept
    }

def _detect_seasonal_peaks(seasonal: pd.Series, n_peaks: int = 3) -> Dict[str, float]:
    """계절성 피크 감지 (인덱스 강제 변환 추가)"""
    try:
        # 인덱스 변환 보강
        if not isinstance(seasonal.index, pd.DatetimeIndex):
            seasonal = seasonal.copy()
            seasonal.index = pd.to_datetime(seasonal.index)
        
        # 월별 그룹화 및 피크 검출
        monthly_avg = (
            seasonal
            .groupby(seasonal.index.month)
            .mean()
            .sort_values(ascending=False)
        )
        
        return {
            f"{int(month)}월": round(monthly_avg[month], 3) 
            for month in monthly_avg.index[:n_peaks]
        }
    except Exception as e:
        print(f"계절성 분석 오류: {str(e)}")
        return {}

def _compare_groups(metrics: Dict[str, Dict]) -> pd.DataFrame:
    """그룹 간 비교 분석"""
    return pd.DataFrame({
        group: {
            'growth_rate': data.get('growth_rate', 0),
            'r_squared': data.get('r_squared', 0),
            'current_value': data.get('current_value', 0)
        }
        for group, data in metrics.items()
    }).T

def generate_insights(decomposed: dict, forecasts: dict, results: dict) -> None:
    """종합 인사이트 리포트 생성 (구조 검증 강화)"""
    # 입력 데이터 구조 검증
    required_keys = ['trend', 'seasonal', 'resid']
    for group, data in decomposed.items():
        if not all(key in data for key in required_keys):
            raise ValueError(f"{group} 데이터에 {required_keys} 누락")
        if not isinstance(data['trend'], pd.Series):
            raise TypeError(f"{group} 트렌드 데이터가 Series가 아닙니다.")
    
    # 트렌드 메트릭 계산 로직 개선
    trend_metrics = {}
    for group, data in decomposed.items():
        try:
            # 트렌드 분석
            trend_metrics[group] = _calculate_trend_metrics(data['trend'])
            
            # 계절성 피크 분석 (인덱스 보존)
            seasonal_peaks = _detect_seasonal_peaks(data['seasonal'].copy())
            
            # 예측 성장률 계산 (인덱스 정렬)
            last_trend = data['trend'].iloc[-1]
            forecast_value = forecasts[group]['prophet']['yhat'][-1]
            
            trend_metrics[group].update({
                'seasonal_peaks': seasonal_peaks,
                'forecast_growth': forecast_value - last_trend,
                'reliability': '높음' if results[group]['ensemble_r2'] > 0.5 else '주의 요망'  # 신뢰도 레이블
            })
        except KeyError as e:
            print(f"{group} 데이터 키 누락: {str(e)}")
            continue
    
    # 2. 그룹별 비교 분석
    comparison_df = _compare_groups(trend_metrics)
    
    # 3. 시각화 생성
    plt.style.use('seaborn-v0_8-darkgrid') 
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 성장률 비교 차트
    comparison_df['growth_rate'].plot.bar(ax=axes[0], color='royalblue')
    axes[0].set_title('Comparison of weekly search volume growth rates by group')
    axes[0].set_ylabel('Growth rate (%)')

    # 예측 정확도 산점도
    
    colors = []
    sizes = []
    for group in trend_metrics:
        r2 = trend_metrics[group]['r_squared']
        colors.append('green' if r2 > 0.5 else 'orange' if r2 > 0 else 'red')
        sizes.append(abs(trend_metrics[group]['growth_rate'])*10)

    axes[1].scatter(
        comparison_df['growth_rate'], 
        comparison_df['r_squared'],
        s=sizes,
        c=colors,  # 신뢰도에 따른 색상
        alpha=0.6
    )
    axes[1].set_xlabel('Search volume growth rate (%)')
    axes[1].set_ylabel('Prediction accuracy (R²)')
    
    plt.tight_layout()
    os.makedirs('modeling/reports', exist_ok=True)
    plt.savefig('modeling/reports/group_comparison.png', bbox_inches='tight')
    plt.close()
    
    # 4. 리포트 생성
    html_report = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>2025 라이프스타일 트렌드 리포트</title>
        <style>
            .section { margin: 30px 0; padding: 20px; border: 1px solid #eee; }
            img { max-width: 100%; height: auto; }
            .highlight { color: #e74c3c; font-weight: bold; }
            .alert { background-color: #fdf5e6; padding: 15px; border-left: 5px solid #e74c3c; }
        </style>
    </head>
    <body>
        <h1>2025 라이프스타일 트렌드 인사이트</h1>
        <div class="section">
            <h2>실시간 검색 트렌드</h2>
            <img src="./real_time_search_trends.png">
            <p>각 그룹별 정규화된 검색 비율 추이 (2022-04 ~ 2025-04)</p>
        </div>
        <div class="section">
            <h2>그룹별 비교 분석</h2>
            <img src="./group_comparison.png">
    """
    
    # 그룹별 상세 분석 추가
    for group, metrics in trend_metrics.items():
        html_report += f"""
        <div class="section">
            <h3>{group} 트렌드 분석</h3>
            <ul>
                <li>주간 성장률: <span class="highlight">{metrics['growth_rate']:.2f}%</span></li>
                <li>계절성 피크: {', '.join([f'{k} ({v}%)' for k,v in metrics['seasonal_peaks'].items()])}</li>
                <li>6개월 예측 성장: <span class="highlight">{metrics['forecast_growth']:.2f}%</span></li>
                <li>모델 정확도 (R²): {metrics['r_squared']:.2f}</li>
            </ul>
        </div>
        """
    html_report += """
    <div class="alert alert-warning mt-4">
        <h4>※ 결과 해석 주의사항</h4>
        <ul>
            <li>R² 값이 음수인 경우: 기본 예측 모델(평균)보다 성능 낮음</li>
            <li>트렌드 지수: 실제 관측치 기반 계산 (예측 신뢰도와 무관)</li>
            <li>붉은색 표시: R² < 0 (모델 개선 필요)</li>
        </ul>
    </div>
    """
    html_report += "</body></html>"
    
    with open('modeling/reports/trend_insights.html', 'w', encoding='utf-8') as f:
        f.write(html_report)
