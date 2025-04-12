# modeling.ml_model.py
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

def train_rf(X, y):
    """랜덤 포레스트 모델 학습"""
    assert y.isna().sum() == 0, "타겟 변수에 NaN이 존재합니다."
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 그리드 서치로 최적 파라미터 탐색
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    # 모델 선택: 데이터 크기에 따라 랜덤 포레스트 또는 그래디언트 부스팅
    if len(X) < 50:  # 데이터가 적은 경우 그래디언트 부스팅 사용
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    else:  # 데이터가 충분한 경우 랜덤 포레스트 사용
        model = RandomForestRegressor(random_state=42)
    
    # 그리드 서치로 최적 파라미터 찾기
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2')
    grid_search.fit(X_train, y_train)
    
    # 최적 모델 추출
    best_model = grid_search.best_estimator_
    score = best_model.score(X_test, y_test)
    
    print(f"최적 파라미터: {grid_search.best_params_}, 정확도: {score:.3f}")
    return best_model, score