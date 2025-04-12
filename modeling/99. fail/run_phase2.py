from modeling.data_preprocessor import *
from modeling.arima_model import *
from modeling.ml_model import train_rf
from modeling.evaluator import *
from modeling.visualizer import *
from modeling.insights_generator import *
import joblib
import warnings
warnings.filterwarnings("ignore")

def analyze_seasonal_peaks(series: pd.Series, dates: pd.Series) -> list:
    """계절성 피크 분석 (컬럼 기반)"""
    monthly_avg = series.groupby(dates.dt.month).mean()
    return monthly_avg.nlargest(2).index.tolist()



def run_phase2(cleaned_df):
    """Phase 2 메인 파이프라인"""
    # 1. 데이터 준비
    ts_df = prepare_time_series(cleaned_df)
    feature_df = add_features(ts_df)
    
    # 데이터 확인
    if len(feature_df) < 20:
        print("⚠️ 경고: 데이터가 너무 적습니다. 정확도가 낮을 수 있습니다.")
    
    # 계절성 탐지
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    for style in feature_df['group_name'].unique():
        style_df = feature_df[feature_df['group_name'] == style]
        
        # 충분한 데이터가 있는 경우에만 계절성 분해
        if len(style_df) >= 12:
            try:
                # 주기 설정 (주간 데이터는 52를 사용)
                result = seasonal_decompose(style_df['ratio'], model='additive', period=52)
                
                # 계절성 시각화
                plt.figure(figsize=(10, 8))
                plt.subplot(411)
                plt.plot(style_df['date'], result.observed)
                plt.title("Observed")
                plt.subplot(412)
                plt.plot(style_df['date'], result.trend)
                plt.title("Trend")
                plt.subplot(413)
                plt.plot(style_df['date'], result.seasonal)
                plt.title("Seasonal")
                plt.subplot(414)
                plt.plot(style_df['date'], result.resid)
                plt.title("Residual")
                plt.tight_layout()
                plt.savefig(f"modeling/models/{style}_seasonal.png")
                plt.close()
            except:
                print(f"계절성 분해 실패: {style}")
    
    print("\n=== 시계열 데이터 검증 ===")
    print("최소 날짜:", ts_df['date'].min())
    print("최대 날짜:", ts_df['date'].max())

    # 2. 모델링
    arima_models = {}
    ml_models = {}
    
    for style in feature_df['group_name'].unique():
        style_data = feature_df[feature_df['group_name'] == style]['ratio']
        style_dates = feature_df[feature_df['group_name'] == style]['date']
        
        # ARIMA 모델 학습
        model = ARIMA(style_data, order=(1,1,1))
        try:
            model_fit = model.fit()
        except Exception as e:
            print(f"ARIMA 학습 실패: {str(e)}")
        
        # 12주 예측
        forecast = model_fit.forecast(steps=12)
        
        # 계절성 피크 분석 (날짜 컬럼 전달)
        peak_season = analyze_seasonal_peaks(style_data, style_dates)
        
        arima_models[style] = {
            'forecast': forecast.tolist(),
            'model': model_fit,
            'peak_season': peak_season
        }
        
        # 머신러닝 학습
        # 특성 및 타깃 준비
        style_df = feature_df[feature_df['group_name'] == style]
        X = style_df[['month', 'season']]
        y = style_df['ratio']
        
        # 모델 학습
        ml_model, score = train_rf(X, y)
        ml_models[style] = score  # 정확도 저장
        joblib.dump(ml_model, f'modeling/models/{style}_model.pkl')  # 모델 파일 저장
        
    
    # 3. 결과 분석
    insights = generate_insights(arima_models, ml_models)
    
    # 4. 시각화
    for style in feature_df['group_name'].unique():
        style_df = feature_df[feature_df['group_name']==style]
        plot_forecast(
            historical_data=style_df['ratio'],
            forecast_data=arima_models[style]['forecast'],
            style_name=style,
            df=feature_df
        )

    # 결과 출력
    print("2025 트렌드 예측 리포트:")
    for style, data in insights.items():
        print(f"- {style}: {data['trend_direction']} 추세 (정확도: {data['model_accuracy']:.2f})")
        print(f"  - 예측 피크 시즌: {data['peak_season']}")
    
    
    return insights

