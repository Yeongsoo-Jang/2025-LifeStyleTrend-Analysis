**📘 프로젝트 개요**  
2025 라이프스타일 트렌드 예측 시스템은 **Naver 검색 데이터**를 기반으로 인테리어 트렌드를 분석/예측하는 E2E 파이프라인입니다. STL 분해, Prophet, ARIMA 모델을 활용해 3가지 키워드(`Cost-Effective`, `HomeAppliances`, `UncommonStyle`)의 2025년 검색량 패턴을 예측합니다.

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB MIT](https://img.shields.io/badge/License-MIT-yellowHub Last Commit](https://img.shields.io/github/last-commit/Yeongsoo-Jang/2025-LifeStyleTrendb.com/Yeongsoo-Jang/2025-LifeStyleTrend-Analysis/commits/main심 기능
![](https://img.shields.io/badge/-Data_Pipeline-007ACC?style=flat&logoimg.shields.io/badge/-Time_Series-4A154B?style=flatimg.shields.io/badge/-ML_OPS-FF6F00?style=flat&logo실시간 데이터 수집**: Naver API 연동 → 주간 단위 검색량 자동 수집
- **다중 계절성 분석**: STL 분해를 통한 추세/계절성/잔차 분리
- **앙상블 예측**: Prophet (계절성 강화) + ARIMA (단기 패턴) 조합
- **자동 리포트**: HTML 리포트 & 대시보드 생성 (예시: [트렌드 리포트](./modeling/reports/trend_insights.html))

---

## 🛠 설치 및 실행
```bash
# 1. 저장소 복제
git clone https://github.com/Yeongsoo-Jang/2025-LifeStyleTrend-Analysis.git
cd 2025-LifeStyleTrend-Analysis

# 2. 가상환경 설정
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 메인 파이프라인 실행
python modeling/run_phase2.py
```

---

## 📊 시스템 아키텍처
```mermaid
graph TD
    A[Naver API] --> B[Raw Data]
    B --> C[Data Preprocessing]
    C --> D[STL Decomposition]
    D --> E[Prophet Model]
    D --> F[ARIMA Model]
    E --> G[Ensemble Forecast]
    F --> G
    G --> H[Interactive Report]
```

---

## 🔧 기술 스택
![](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white.io/badge/Prophet-FF6F00?logo=facebook&logoColor=white.io/badge/StatsModels-8C8C8C?logo=python&logoColor=lds.io/badge/Matplotlib-11557C야 | 도구 |
|------|------|
| **데이터 수집** | Naver API, Python-Requests |
| **전처리** | Pandas, NumPy |
| **시계열 분석** | STL, ACF/PACF |
| **예측 모델** | Prophet, pmdarima |
| **시각화** | Matplotlib, Seaborn |

---

## 📈 주요 결과
### 예측 정확도 (R²)
| 그룹           | Prophet | ARIMA | 앙상블 |
|----------------|---------|-------|--------|
| Cost-Effective | 0.82    | 0.78  | 0.85   |
| UncommonStyle  | 0.89    | 0.81  | 0.91   |

![트렌드 비교](./modeling/reports/real 📝 학습 경험
- **데이터 파이프라인 구축**: 주간 단위 자동화 시스템 설계
- **계절성 처리**: 52주 주기 STL 분해 최적화
- **모델 통합**: Prophet-ARIMA 앙상블 가중치 자동 계산 알고리즘 개발
- **문제 해결**: Negative R² 이슈 → 데이터 정제 강화로 해결

---

## 📄 라이선스
[MIT License](LICENSE) © 2025 장영수  
**문의**: [포트폴리오](https://github.com/Yeongsoo-Jang) | 9135jys@gmail.com