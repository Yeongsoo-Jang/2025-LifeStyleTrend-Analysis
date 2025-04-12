def generate_insights(arima_models: dict, ml_scores: dict) -> dict:
    """트렌드 분석 리포트 생성"""
    insights = {}
    
    for style, forecast_data in arima_models.items():
        # ARIMA 예측값 추출
        forecast_values = forecast_data['forecast']
        
        # 트렌드 방향 계산
        trend_direction = "상승" if forecast_values[-1] > forecast_values[0] else "하락"
        
        # 머신러닝 정확도
        ml_accuracy = ml_scores.get(style, 0)
        
        insights[style] = {
            'trend_direction': trend_direction,
            'model_accuracy': ml_accuracy,
            'peak_season': forecast_data.get('peak_season', 'N/A')
        }
    
    return insights
