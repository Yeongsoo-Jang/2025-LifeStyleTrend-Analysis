**📘 프로젝트 개요**  
2025 라이프스타일 트렌드 예측 시스템은 **시계열 분석과 머신러닝을 결합**한 인테리어 트렌드 예측 파이프라인입니다. STL 분해, Prophet, ARIMA를 활용해 3가지 키워드(`Cost-Effective`, `HomeAppliances`, `UncommonStyle`)의 검색량 패턴을 분석하고 2025년 트렌드를 예측합니다.

- **계절성 패턴 분석**: 52주 주기 STL 분해
- **다중 모델 병렬 예측**: Prophet (계절성 강화) + ARIMA (단기 패턴)
- **앙상블 최적화**: R² 기반 가중치 자동 계산
- **인터랙티브 리포트**: HTML/PNG 형식의 시각화 결과 자동 생성

---

## 🛠 기술 스택
```bash
# 1. 저장소 복제 및 환경 설정
git clone https://github.com/Yeongsoo-Jang/2025-LifeStyleTrend-Analysis.git
cd 2025-LifeStyleTrend-Analysis
python -m venv .venv && source .venv/bin/activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 메인 파이프라인 실행
python modeling/run_phase2.py
```

---

## 📂 프로젝트 구조
```
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
└── requirements.txt # 패키지 의존성
```

---

## 📈 주요 결과
### 트렌드 리포트 예시

# 📊 트렌드 리포트

![2025 라이프스타일 트렌드 리포트](./trend_insights.svg)

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
| Cost-Effective | 0.82    | 0.78  | 0.85   |
| UncommonStyle  | 0.89    | 0.81  | 0.91   |

---

## 🧠 핵심 기술
1. **STL 분해 최적화**  
```python
# 52주 주기 고정 + 강건성 향상
STL(series, period=52, robust=True).fit()
```

2. **앙상블 가중치 계산**  
```python
weights = (prophet_r2 / total_r2, arima_r2 / total_r2)
```

3. **트렌드 지수 공식**  
```math
\text{Trend Index} = \beta \times R^2
```

---

## 📜 라이선스
[MIT License](LICENSE) © 2025 장영수  
**문의**: [포트폴리오](https://github.com/Yeongsoo-Jang) | 9135jys@gmail.com

---

> "데이터는 미래를 보는 창입니다. 이 프로젝트는 2025 인테리어 시장의 전략 수립을 지원합니다."  
---

✨ **Tip for Interviewers**:  
- [코어 로직 살펴보기](modeling/run_phase2.py)  
- [최종 리포트 샘플 확인](modeling/reports/trend_insights.html)  
- [기술적 도전 사항 분석](https://github.com/Yeongsoo-Jang/2025-LifeStyleTrend-Analysis/issues/1)