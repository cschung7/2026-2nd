# -*- coding: utf-8 -*-
"""
Sogang University - AI Finance Seminar
[실습 가이드] 달러 바(Dollar Bars) + 트리플 배리어 기법(TBM) + 메타 레이블링(Meta-Labeling) ML 파이프라인

본 스크립트는 마르코스 로페즈 데 프라도(Marcos López de Prado) 교수의 저서 
'Advances in Financial Machine Learning' 2장과 3장의 핵심 방법론을 하나로 연결한 종합 실습 코드입니다.

[학습 흐름]
Step 1. 가상 틱 데이터 생성 (일중 U자형 거래량 및 변동성 패턴 반영)
Step 2. 달러 바(Dollar Bars) 전처리 및 생성
Step 3. 피처 엔지니어링 (머신러닝 피처 생성: 이동평균 괴리율, 변동성, RSI 등)
Step 4. 1차 모델(Primary Model) 수립: 단순 이동평균 골든크로스 전략 (포지션 방향 'side' 결정)
Step 5. 동적 트리플 배리어 기법(Triple-Barrier Method) 적용 (익절/손절/만기 배리어 설정)
Step 6. 메타 레이블(Meta-Label) 생성 (1차 모델 예측 성공 시 1, 실패/만기 시 0)
Step 7. 2차 모델(Secondary Model) 수립: Random Forest Classifier 학습 (피처 -> 메타 레이블 예측)
Step 8. 성과 평가: 메타 레이블링이 '정밀도(Precision)'와 '샤프 지수'를 어떻게 개선하는지 실증 비교
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 난수 시드 고정
np.random.seed(42)

# ==============================================================================
# STEP 1. 가상 틱(Tick) 데이터 시뮬레이션
# ==============================================================================
print("Step 1. 고빈도 틱 데이터 시뮬레이션 중...")
n_ticks = 150000
days = 10
day_seconds = 23400  # 6.5시간 영업일

# U-shaped 일중 거래 밀도 모사 (장초반/장마감 활성화)
u_draws = np.random.beta(0.5, 0.5, size=n_ticks)
time_offsets = np.sort(u_draws.reshape(days, n_ticks // days), axis=1)

timestamps = []
for d in range(days):
    day_times = d * 86400 + time_offsets[d] * day_seconds
    timestamps.extend(day_times)
timestamps = np.array(timestamps)

# 변동성 스케일 설정 (U자형 반영)
vol_scale = 0.0008 * (1.0 + 3.0 * (1.0 - 2.0 * np.abs(u_draws - 0.5)))
price_changes = np.random.normal(loc=0.0, scale=vol_scale)
prices = 100.0 * np.exp(np.cumsum(price_changes))

# 거래량 설정 (거래가 몰릴 때 거래대금 급증)
mean_volume = 150 + 850 * (1.0 - 2.0 * np.abs(u_draws - 0.5))
volumes = np.random.exponential(scale=mean_volume).astype(int) + 10
dollar_values = prices * volumes

df_ticks = pd.DataFrame({
    'timestamp': pd.to_datetime(timestamps, unit='s'),
    'price': prices,
    'volume': volumes,
    'dollar_value': dollar_values
})

# ==============================================================================
# STEP 2. 달러 바(Dollar Bars) 생성
# ==============================================================================
print("Step 2. 달러 바(Dollar Bars) 생성 중...")
total_val = df_ticks['dollar_value'].sum()
# 하루 평균 약 200개의 바가 생성되도록 임계값 설정
dollar_threshold = total_val / (days * 200)

bar_indices = []
cum_val = 0
for i, val in enumerate(df_ticks['dollar_value']):
    cum_val += val
    if cum_val >= dollar_threshold:
        bar_indices.append(i)
        cum_val = 0

db = df_ticks.iloc[bar_indices].copy()
db = db.set_index('timestamp')
print(f"-> 생성 완료: 총 틱수 {n_ticks}개 -> 달러 바 {len(db)}개로 압축.")

# ==============================================================================
# STEP 3. 피처 엔지니어링 (Feature Engineering)
# ==============================================================================
print("Step 3. 머신러닝 학습용 피처 생성 중...")
# 1. 수익률 및 변동성 피처
db['ret'] = db['price'].pct_change()
db['vol_10'] = db['ret'].rolling(window=10).std()
db['vol_50'] = db['ret'].rolling(window=50).std()

# 2. 이동평균선 괴리율 (MA Divergence)
db['ma_5'] = db['price'].rolling(window=5).mean()
db['ma_20'] = db['price'].rolling(window=20).mean()
db['ma_60'] = db['price'].rolling(window=60).mean()
db['div_5'] = db['price'] / db['ma_5'] - 1
db['div_20'] = db['price'] / db['ma_20'] - 1
db['div_60'] = db['price'] / db['ma_60'] - 1

# 3. 상대강도지수 (RSI - 14)
delta = db['price'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / (avg_loss + 1e-9)
db['rsi'] = 100 - (100 / (1 + rs))

db = db.dropna()

# ==============================================================================
# STEP 4. 1차 모델 (Primary Model): 단순 기술적 트레이딩 규칙 적용
# ==============================================================================
print("Step 4. 1차 모델(SMA Crossover) 거래 방향 수립 중...")
# 단기 이평선(5)이 장기 이평선(20)을 돌파하면 매수(+1), 하향 돌파하면 매도(-1)
db['side'] = 0
db.loc[db['ma_5'] > db['ma_20'], 'side'] = 1
db.loc[db['ma_5'] < db['ma_20'], 'side'] = -1

# 포지션에 진입한 이벤트 필터링
events = db[db['side'] != 0].copy()

# ==============================================================================
# STEP 5. 동적 트리플 배리어 기법 (Triple-Barrier Method) 적용
# ==============================================================================
print("Step 5. 동적 트리플 배리어 기법 적용 중...")

# A. 지수 가중 이동 표준편차 기반 동적 타겟(임계치) 설정
span = 50
events['target'] = events['ret'].rolling(window=span).std()
events = events.dropna(subset=['target'])

# B. 수직 장벽 (Vertical Barrier, 만기) 생성: 진입 후 달러 바 15개 경과 시점
vertical_barrier_span = 15
t1_series = []
for i in range(len(events)):
    entry_idx = db.index.get_loc(events.index[i])
    exit_idx = min(entry_idx + vertical_barrier_span, len(db) - 1)
    t1_series.append(db.index[exit_idx])
events['t1'] = t1_series

# C. 트리플 배리어 경로 추적
# pt_sl 승수 설정 (익절: 변동성의 1.5배, 손절: 변동성의 1.0배)
pt_multiplier = 1.5
sl_multiplier = 1.0

final_t1 = []
for loc, row in events.iterrows():
    side = row['side']
    target = row['target']
    t_end = row['t1']
    
    # 해당 이벤트 기간의 가격 경로 발췌
    path = db.loc[loc:t_end, 'price']
    path_returns = (path / db.loc[loc, 'price'] - 1) * side
    
    pt_barrier = target * pt_multiplier
    sl_barrier = -target * sl_multiplier
    
    # 첫 도달 지점 계산
    first_pt = path_returns[path_returns >= pt_barrier].index.min()
    first_sl = path_returns[path_returns <= sl_barrier].index.min()
    
    # 최선 청산 시점 판단
    if first_pt is pd.NaT and first_sl is pd.NaT:
        final_t1.append(t_end)  # 시간 만기 청산
    elif first_pt is pd.NaT:
        final_t1.append(first_sl) # 손절 청산
    elif first_sl is pd.NaT:
        final_t1.append(first_pt) # 익절 청산
    else:
        final_t1.append(min(first_pt, first_sl)) # 둘 중 먼저 닿는 것

events['t1'] = final_t1

# ==============================================================================
# STEP 6. 메타 레이블(Meta-Label) 생성
# ==============================================================================
print("Step 6. 2차 모델 학습용 메타 레이블 생성 중...")
# 실제 최종 도달한 가격을 기준으로 수익률 정산
events['real_ret'] = (db.loc[events['t1'], 'price'].values / db.loc[events.index, 'price'].values - 1) * events['side']

# 메타 레이블: 실제 수익이 났으면 1(실행), 손실 혹은 무수익 청산이면 0(거절/패스)
events['meta_label'] = (events['real_ret'] > 0).astype(int)

# ==============================================================================
# STEP 7. 2차 모델 (Secondary Model) 수립 및 학습
# ==============================================================================
print("Step 7. 2차 모델(Random Forest) 학습 및 예측 실행 중...")
# 학습에 사용할 피처셋 구성
feature_cols = ['vol_10', 'vol_50', 'div_5', 'div_20', 'div_60', 'rsi']

X = events[feature_cols]
y = events['meta_label']

# 학습 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Random Forest Classifier 훈련
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# 테스트 세트 예측
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# ==============================================================================
# STEP 8. 메타 레이블링 효과 검증 (핵심 결과 요약)
# ==============================================================================
print("\n" + "="*80)
print("                       [ ML PIPELINE PERFORMANCE EVALUATION ]")
print("="*80)

# 1. 1차 모델만의 단독 성과 (테스트 셋에 매칭되는 시그널 기준)
test_events = events.loc[X_test.index]
primary_precision = (test_events['real_ret'] > 0).mean()

# 2. 2차 모델(메타 레이블링) 필터링이 적용된 최종 성과
# 2차 모델이 예측한 성공 확률이 높은 신호만 선별하여 베팅을 실행했다고 가정 (임계치 0.5)
bet_decision = (y_pred_proba > 0.5).astype(int)
executed_trades = test_events[bet_decision == 1]

if len(executed_trades) > 0:
    meta_precision = (executed_trades['real_ret'] > 0).mean()
    rejected_count = len(test_events) - len(executed_trades)
else:
    meta_precision = 0.0
    rejected_count = len(test_events)

# 통계 출력
print(f"1. 전체 가용 거래 시그널 수          : {len(test_events)}건")
print(f"2. 2차 모델이 필터링한 신호 (거절)     : {rejected_count}건")
print(f"3. 2차 모델 승인 후 실제 실행한 거래 수 : {len(executed_trades)}건")
print("-"*80)
print(f"★ 1차 모델 단독 사용시 정밀도(Precision) : {primary_precision*100:.2f}%")
print(f"★ 2차 모델 결합시 최종 정밀도(Precision) : {meta_precision*100:.2f}%")
print(f"☞ 정밀도 개선 폭 (Precision Lift)       : +{(meta_precision - primary_precision)*100:.2f}%p")
print("="*80)

print("\n[2차 모델 분류 성능 리포트 (Classification Report)]")
print(classification_report(y_test, y_pred, target_names=['Pass (0)', 'Bet (1)']))

print("\n* 실습 팁: 'y_pred_proba'의 임계값(0.5)을 더 높이면(예: 0.6) 더 보수적으로 완벽한 기회에만")
print("  베팅하게 되며, 이 경우 정밀도는 추가로 상승하지만 거래 횟수(Recall)는 감소합니다.")
print("  이것이 마르코스 데 프라도 교수가 제시한 '정밀도와 재현율의 트레이드오프' 원리입니다.\n")
