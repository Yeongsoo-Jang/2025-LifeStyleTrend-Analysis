# 2025 LifeStyleTrend Analysis
 2025년 라이프스타일 트렌드 검색어 및 시계열 분석


**📂 GitHub Repository 구조**
```
2025-lifestyle-trend-forecast/
├── data/
│ ├── .env # API 키 등 민감정보
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
```

**📄 README.md 작성 가이드**

```markdown
# 2025 라이프스타일 트렌드 예측 시스템

![대시보드 예시](./reports/real_time_search_trends.png)

> **인테리어 트렌드 예측을 위한 데이터 기반 의사결정 시스템**  
> STL 분해 · Prophet · ARIMA · 앙상블 기법 적용

## 🌟 주요 기능
- **다중 계절성 분해**: 52주 주기 기준 STL 분해
- **병렬 예측 처리**: 4-Worker 병렬 처리로 효율성 극대화
- **앙상블 최적화**: R² 기반 Prophet-ARIMA 가중치 조정
- **실시간 대시보드**: 정규화된 트렌드 비교 시각화

## 🛠 기술 스택
| 분야 | 도구 |
|------|------|
| 언어 | Python 3.10 |
| 데이터 처리 | Pandas · NumPy |
| 시계열 분해 | statsmodels.STL |
| 머신러닝 | Prophet · pmdarima |
| 시각화 | Matplotlib · Seaborn |

## 🔧 설치 및 실행
```
git clone https://github.com/your-username/2025-lifestyle-trend-forecast.git
cd 2025-lifestyle-trend-forecast

# 가상환경 설정
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt

# 메인 파이프라인 실행
python modeling/run_phase2.py
```

## 📊 데이터 파이프라인
1. **원본 데이터 형식**
   ```
   date,group_name,ratio
   2022-04-03,Cost-Effective,0.76693
   2022-04-10,HomeAppliances,50.28760
   ```
2. **전처리 단계**
   - 주간 리샘플링
   - 계절성 변수(월/분기) 추가
   - 변동성 기준(표준편차 >0.5) 필터링

## 📈 방법론
### 1. STL 분해
```
decomposition = STL(series, period=52).fit()
trend = decomposition.trend  # 추세 성분
```

### 2. Prophet & ARIMA 병렬 예측
| 모델 | 장점 | 한계 |
|------|------|------|
| Prophet | 강력한 계절성 처리 | 변동성 민감 |
| ARIMA | 복잡한 패턴 포착 | 장기 예측 불안정 |

### 3. 앙상블 최적화
```
weights = (prophet_r2 / total_r2, arima_r2 / total_r2)
ensemble = prophet_pred * weights + arima_pred * weights[1]
```

## 📌 결과 해석
| 지표 | Cost-Effective | UncommonStyle |
|------|----------------|---------------|
| RMSE | 13.89          | 0.11          |
| MAPE | 93.12%         | 43.90%        |
| R²   | 0.98           | 0.94          |

> 💡 **주의사항**: R² <0 인 경우 기본 예측 모델 대비 성능 저하

---

**문의**: 장영수 · 9135jys@gmail.com  
**최종 업데이트**: 2025-04-12
```

**🔗 포함해야 할 추가 요소**
1. `requirements.txt`에 주요 패키지 버전
   ```
   pandas==2.2.0
   numpy==1.24.0
   prophet==1.1.5
   pmdarima==2.0.4
   ```
2. `.gitignore` 파일에 불필요 파일 제외 설정
3. LICENSE 파일 추가
4. CONTRIBUTING.md 상세 기여 가이드