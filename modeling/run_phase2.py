# modeling.run_phase2.py
import logging
from typing import Dict
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import joblib
# 모듈 임포트
from modeling.data_preprocessor import add_features, prepare_time_series, clean_data
from modeling.stl_decomposer import decompose_trend
from modeling.prophet_model import prophet_forecast
from modeling.arima_model import ARIMAModel, train_arima, evaluate_arima
from modeling.forecast_visualizer import plot_forecasts
from modeling.evaluator import evaluate_forecasts
from modeling.insights_generator import generate_insights

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_phase2(cleaned_df: pd.DataFrame) -> Dict:
    """고도화된 트렌드 분석 파이프라인"""
    
    # 1. 데이터 전처리 --------------------------------------------------------
    logger.info("=== Phase 2: 데이터 전처리 시작 ===")
    try:
        processed_df = add_features(cleaned_df)
        ts_df = prepare_time_series(processed_df)
        ts_df = clean_data(ts_df)
        
        # 데이터 검증
        if ts_df.empty:
            raise ValueError("전처리된 데이터가 없습니다.")
            
        logger.info("\n=== 전처리 데이터 통계 ===")
        logger.info(f"그룹 수: {ts_df['group_name'].nunique()}")
        logger.info(f"시간 범위: {ts_df['date'].min()} ~ {ts_df['date'].max()}")
        
    except Exception as e:
        logger.error(f"전처리 실패: {str(e)}", exc_info=True)
        raise

    logger.info("전처리 후 데이터 컬럼: %s", processed_df.columns.tolist())
    logger.info("date 컬럼 샘플 값: %s", processed_df['date'].head())
    
    # 2. STL 분해 -----------------------------------------------------------
    logger.info("\n=== Phase 2: STL 분해 실행 ===")
    decomposition_result = decompose_trend(ts_df)
    decomposed_groups = decomposition_result['decompositions']  
    
    # 그룹별 분해 결과 로깅
    logger.info("STL 분해 그룹 목록: %s", list(decomposed_groups.keys()))
    
    # 분해 결과 상세 로깅
    logger.info("분해 품질 보고서:\n%s", 
        decomposition_result['quality_report'].to_markdown())
    
    
    
    
    # 3. 병렬 예측 처리 ------------------------------------------------------
    logger.info("\n=== Phase 2: 병렬 예측 시작 ===")
    forecasts = {}
    
    def _process_group(style: str, data: pd.DataFrame) -> Dict:
        """입력 데이터 검증 추가"""
        required_cols = ['date', 'trend', 'ratio']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"{style} 데이터에 {required_cols} 누락")
        """그룹별 예측 처리"""
        try:
            # 데이터 분할 (컬럼 기반)
            data = data.sort_values('date')
            cutoff = data['date'].iloc[-26]
            train = data[data['date'] < cutoff]
            
            # Prophet 예측
            prophet_fcst = prophet_forecast(train['ratio'], train['date'], periods = 26)
            
            # ARIMA 예측
            arima_model, arima_fcst = train_arima(train['ratio'], n_periods=26)
            
            return {
                'style': style,
                'prophet': prophet_fcst,
                'arima_model': arima_model,  # 모델 객체 반환
                'arima_forecast': arima_fcst,
                'train_data': train
            }
        except Exception as e:
            logger.error(f"{style} 예측 실패 상세: {str(e)}", exc_info=True)
            return None


    # 병렬 실행
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for style, data in decomposed_groups.items():
            # 1. None 체크
            if data is None:
                logger.error(f"{style} - 분해 실패 데이터")
                continue
                
            # 2. 키 존재 여부
            if 'trend' not in data or 'observed' not in data:
                logger.error(f"{style} - 필수 데이터 누락")
                continue
                
            # 3. 데이터 길이 검증
            if len(data['trend']) < 104:
                logger.error(f"{style} - 데이터 부족 ({len(data['trend'])}주)")
                continue
            
                
            input_df = pd.DataFrame({
                'date': pd.to_datetime(data['trend'].index),
                'trend': data['trend'].values,
                'ratio': data['observed'].values  # 실제 사용될 값 추가
            })
            futures.append(executor.submit(_process_group, style, input_df))
            
        # ▼▼▼ 결과 수집 추가 ▼▼▼
        for future in tqdm(futures, desc="예측 진행률"):
            result = future.result()
            if result and result['style']:
                forecasts[result['style']] = result

    # 예측 전 데이터 샘플 출력
    logger.debug("예측 입력 데이터 샘플:\n%s", input_df.head(10).to_markdown())
    
    # 검증
    logger.info("분해 그룹: %s", list(decomposed_groups.keys()))
    logger.info("예측 그룹: %s", list(forecasts.keys()))

    # 4. 성능 평가 및 리포트 생성 ---------------------------------------------
    logger.info("\n=== Phase 2: 최종 평가 ===")
    logger.info("분해 그룹 구조: %s", 
    {k: list(v.keys()) for k,v in decomposed_groups.items()})
    logger.info("예측 그룹 구조 상세: %s", 
    {k: list(v['prophet'].keys()) for k,v in forecasts.items()})
    
    results = evaluate_forecasts(decomposed_groups, forecasts)
    if results:
        sample_style = next(iter(results.keys()))
        print("평가 결과 샘플:", results[sample_style].keys())
    else:
        print("평가 결과 없음")
    
    # 앙상블 가중치 계산 (zero division 방지)
    for style in forecasts:
        # 가중치 계산 로직
        prophet_score = max(results[style].get('r2', 0), 0)
        arima_score = max(evaluate_arima(
            forecasts[style]['arima_model'], 
            decomposed_groups[style]['trend'][-26:]
        ).get('r2', 0), 0)
        
        total = prophet_score + arima_score
        weights = (prophet_score/total, arima_score/total) if total != 0 else (0.5, 0.5)
        
        # 데이터 길이 검증
        prophet_len = len(forecasts[style]['prophet']['yhat']) 
        arima_len = len(forecasts[style]['arima_forecast']) 
        assert prophet_len >= 26, f"Prophet 예측 부족: {prophet_len}"
        assert arima_len == 26, f"ARIMA 예측 길이 오류: {arima_len}"

        # 앙상블 생성
        forecasts[style]['ensemble'] = (
            forecasts[style]['prophet']['yhat'][-26:] * weights[0] + 
            forecasts[style]['arima_forecast'] * weights[1]
        )
    
    # 앙상블 후 검증
    ensemble = forecasts[style]['ensemble']
    if len(ensemble) != 26:
        raise ValueError(f"{style} 앙상블 길이 오류: {len(ensemble)}")
    if ensemble.isnull().any():
        raise ValueError(f"{style} 앙상블에 NaN 값 존재")

    # 예측 시각화 ---------------------------------------------------
    # 앙상블 생성 후 재평가
    updated_results = evaluate_forecasts(decomposed_groups, forecasts)  # 앙상블 포함 평가
    generate_insights(decomposed_groups, forecasts, updated_results)    # 최신 결과 사용
    
    # 모델 저장 (안전한 버전)
    for style in forecasts:
        if 'arima_model' in forecasts[style]:
            joblib.dump(
                forecasts[style]['arima_model'], 
                f'modeling/models/{style}_arima.pkl'
            )
        else:
            logger.warning(f"{style} ARIMA 모델 저장 실패: 모델 객체 없음")
    
    return decomposed_groups, forecasts, updated_results
