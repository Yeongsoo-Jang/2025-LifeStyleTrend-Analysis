**📘 프로젝트 개요**  
2025 라이프스타일 트렌드 예측 시스템은 **Naver 검색 데이터**를 기반으로 한 과학적 시계열 분석 파이프라인입니다. STL 분해, Prophet, ARIMA 모델을 결합해 3가지 키워드(`Cost-Effective`, `HomeAppliances`, `UncommonStyle`)의 2025년 검색량 패턴을 예측하며, **데이터 검증 → 모델링 → 앙상블 최적화**로 2025년 트렌드를 예측합니다.

- **계절성 패턴 분석**: 52주 주기 STL 분해
- **다중 모델 병렬 예측**: Prophet (계절성 강화) + ARIMA (단기 패턴)
- **앙상블 최적화**: R² 기반 가중치 자동 계산
- **인터랙티브 리포트**: HTML/PNG 형식의 시각화 결과 자동 생성

---

## 📂 프로젝트 구조
```
├── data/                  
│ ├── .env # API 키 등 민감정보(**개인 생성 및 NAVER API 키 작성 필요)
│ ├── config.yaml # 분석 대상 키워드 설정
│ └── processed/ # 정제된 데이터
├── connector/
│ ├── naver_api.py # Naver 검색 API 연동 모듈
│ ├── config_loader.py # YAML 설정파일 파싱
│ └── connect.py # 메인 실행 (데이터 수집)
├── processed/
│ ├── cleaner.py # 데이터 정제 (이상치 처리)
│ ├── validator.py # 데이터 무결성 검증
│ └── monitor.py # 실시간 대시보드
├── modeling/
│ ├── models/ # 학습된 모델 저장
│ ├── reports/ # HTML 리포트 & 시각화 결과
│ ├── arima_model.py # ARIMA 모델링
│ ├── prophet_model.py # Prophet 모델링
│ ├── run_phase2.py # 메인 실행 (분석 파이프라인)
│ └── stl_decomposer.py # STL 시계열 분해
└── requirements.txt # 패키지 의존성
```

---

## 📈 주요 결과
### 트렌드 리포트 예시

# 📊 트렌드 리포트

[![HTML Preview](https://img.shields.io/badge/HTML_Preview-Open_in_Tab-green)](https://htmlpreview.github.io/?https://github.com/Yeongsoo-Jang/2025-LifeStyleTrend-Analysis/blob/main/modeling/reports/trend_insights.html)

![정규화된 트렌드 시각화](./modeling/reports/real_time_search_trends.png)

> **Cost-Effective**  
> - 주간 성장률: **10.84%**  
> - 계절성 피크: **2월 (3.605%)**, **1월 (3.153%)**, **9월 (0.287%)**  
> - 모델 정확도 (R²): **0.98**

> **HomeAppliances**  
> - 주간 성장률: **10.82%**  
> - 계절성 피크: **7월 (7.204%)**, **5월 (6.731%)**, **2월 (1.956%)**  
> - 모델 정확도 (R²): **0.96**

> **UncommonStyle**  
> - 주간 성장률: **-0.07%**  
> - 계절성 피크: **8월 (0.067%)**, **2월 (0.057%)**, **1월 (0.056%)**  
> - 모델 정확도 (R²): **0.94**




### 성능 비교표 (R²)
| 그룹           | Prophet | ARIMA | 앙상블 |
|----------------|---------|-------|--------|
| Cost-Effective | 0.98    | 0.94  | 0.98   |
| HomeAppliances | 0.96    | 0.92  | 0.96   |
| UncommonStyle  | 0.94    | 0.91  | 0.94   |

---

🚀 프로젝트 실행 가이드
1. **데이터 수집 및 전처리**
```bash
# Naver API 키 설정
echo "NAVER_CLIENT_ID=your_id" > data/.env
echo "NAVER_CLIENT_SECRET=your_secret" > data/.env

# 데이터 수집 및 정제
python connector/connect.py
```

2. **분석 파이프라인 실행**
```bash
# STL 분해 + Prophet/ARIMA 병렬 예측
python modeling/run_phase2.py
```

🛠 핵심 기술 및 데이터 검증 프로세스
1. **데이터 검증 강화**
```python
# validator.py - 데이터 무결성 검증
def validate_data(df: pd.DataFrame):
    # 필수 컬럼 검증
    assert {'date', 'group_name', 'ratio'} <= set(df.columns), "필수 컬럼 누락"
    
    # 날짜 범위 검증 (최소 2년 데이터)
    date_range = pd.to_datetime(df['date']).max() - pd.to_datetime(df['date']).min()
    assert date_range.days >= 730, f"데이터 범위 부족: {date_range.days}일"
    
    # 그룹별 데이터 밸런스 검증
    group_counts = df['group_name'].value_counts()
    assert group_counts.std() / group_counts.mean() < 0.1, "그룹 간 데이터 불균형"
```

2. **트러블슈팅 사례**
🔍 문제 1: Prophet-ARIMA 예측 불일치
증상: Prophet 예측값(158주)과 ARIMA 예측값(26주) 길이 불일치 → 앙상블 시 ValueError

해결:

```python
# 앙상블 생성 전 데이터 정렬
prophet_pred = forecasts[style]['prophet']['yhat'].iloc[-26:]  # 최근 26주만 선택
arima_pred = forecasts[style]['arima_forecast']
```

🔍 문제 2: Naver API 401 오류
증상: HTTP 401 Unauthorized 지속 발생

원인: 환경변수(.env) 키 이름 불일치 + SSL 검증 문제

해결:

```python
# naver_api.py 수정
self.client_id = os.getenv("NAVER_CLIENT_ID")  # 기존: CLIENT_KEY
self.ssl_context = ssl._create_unverified_context()  # SSL 검증 비활성화
```

🔍 문제 3: 계절성 피크 검출 실패
증상: seasonal_peaks에서 월별 그룹화 실패

원인: DatetimeIndex 미적용

해결:

```python
data['seasonal'].index = pd.to_datetime(data['seasonal'].index)
```

📈 모델 성능 검증 체계
1. **교차 검증 강화**
```python
# evaluator.py - Prophet 검증
def validate_prophet(model, train_data):
    # 3년 초기 데이터 → 13주 단위 검증
    df_cv = cross_validation(
        model, 
        initial='728 days', 
        period='91 days', 
        horizon='182 days',
        parallel="processes"
    )
    return performance_metrics(df_cv)
```

2. **앙상블 가중치 계산 로직**
```python
# run_phase2.py - R² 기반 가중치 최적화
prophet_score = max(results[style].get('r2', 0), 0)  # 음수 값 방지
arima_score = max(evaluate_arima(...).get('r2', 0), 0)
total = prophet_score + arima_score
weights = (prophet_score/total, arima_score/total) if total !=0 else (0.5, 0.5)
```

---

### 📜 라이선스
[MIT License](LICENSE) © 장영수  
문의: [GitHub](https://github.com/Yeongsoo-Jang) | 이메일: `9135jys@gmail.com`




