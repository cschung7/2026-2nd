# ==============================================================================
# [실습 가이드] 모듈 4: 행동경제학 AI 에이전트 시장 시뮬레이터 (Behavioral Investors)
# 과목명: AI금융 세미나 (2026년 가을학기)
# 담당교수: 정재식 교수 (서강대학교)
# 대상: MBA/GSE 학생 실습용
# ==============================================================================
# 
# [설명]
# 이 스크립트는 전통 금융학의 '합리적 투자자' 가정과 행동경제학의 '편향된 투자자' 가정이
# 금융 시장의 변동성, 버블(Bubble) 형성, 그리고 패닉 셀(Crash)에 미치는 영향을
# 에이전트 기반 모델(Agent-Based Model, ABM)로 시뮬레이션하고 비교합니다.
#
# [주요 편향 구현]
# 1. 과잉 확신 (Overconfidence): 최근의 가격 추세(모멘텀)를 과도하게 추정하여 추격 매수/매도를 집행합니다.
# 2. 손실 회피 (Loss Aversion): 자산 가격이 자신의 매입 단가 대비 특정 임계치 이하로 하락하면 
#    공포심에 사로잡혀 자산을 시장에 급하게 매도(Panic Selling)합니다.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 헤드리스 서버 구동용 설정 (GUI 창을 띄우지 않음)
import matplotlib.pyplot as plt

# 시드 값 고정으로 매번 일관된 시뮬레이션 결과 재현 보장 (Mimic & Reproduce)
np.random.seed(42)

class BehavioralInvestorAgent:
    """
    시장 거래에 참여하는 개별 투자자 에이전트 클래스
    """
    def __init__(self, agent_id, agent_type, cash=10000, holdings=100, 
                 overconfidence=0.1, loss_aversion=1.5):
        self.agent_id = agent_id
        self.agent_type = agent_type # 'Rational', 'Overconfident', 'LossAverse'
        self.cash = cash
        self.holdings = holdings
        self.overconfidence = overconfidence # 모멘텀(가격 추세) 추종 강도
        self.loss_aversion = loss_aversion   # 손실 시 패닉 강도
        
        # 에이전트의 최초 평단가는 초기 시장가(100)로 설정
        self.average_cost = 100.0
        
    def calculate_demand(self, current_price, prev_price, fundamental_value):
        """
        에이전트의 심리적 상태와 가격 정량 정보를 바탕으로 순수요(Buy/Sell)를 계산
        """
        # 1. 합리적 가치 기반 수요: 가격이 내재가치보다 낮으면 매수, 높으면 매도 (평균 회귀 성향)
        value_gap = fundamental_value - current_price
        rational_demand = 0.1 * value_gap
        
        # 2. 과잉 확신 편향 (추세 추종/모멘텀): 가격이 오르면 더 오를 거라 믿고 사고, 내리면 더 내릴 거라 믿고 팖
        price_trend = current_price - prev_price
        momentum_demand = self.overconfidence * price_trend * 1.5
        
        # 3. 손실 회피 편향 (손절매/패닉셀): 평단가 대비 손실이 일정 수준 이상 발생하면 시장가로 전량 매도 유도
        panic_sell_demand = 0.0
        if self.agent_type == 'LossAverse' and self.holdings > 0:
            current_loss_rate = (self.average_cost - current_price) / self.average_cost
            # 손실률이 8%를 초과할 경우 손실 회피 강도에 비례해 패닉셀 매물 출하
            if current_loss_rate > 0.08:
                panic_sell_demand = -self.holdings * 0.5
                
        # 에이전트 타입별 가중치 조율을 통한 최종 수요 결정
        if self.agent_type == 'Rational':
            demand = rational_demand + (0.1 * momentum_demand)
        elif self.agent_type == 'Overconfident':
            # 과잉 확신 에이전트는 가격 추세를 맹목적으로 따름
            demand = (0.1 * rational_demand) + (momentum_demand * 1.8)
        elif self.agent_type == 'LossAverse':
            # 손실 회피 투자자는 평상시에는 합리적으로 행동하다가, 패닉셀 조건 충족 시 폭발적인 매도 주문 제출
            if panic_sell_demand < 0:
                demand = panic_sell_demand
            else:
                demand = rational_demand + (0.2 * momentum_demand)
        else:
            demand = rational_demand
            
        # 가용한 현금 및 주식 잔고 범위 내에서 주문 제약 적용 (안전 장치)
        if demand > 0: # 매수 시
            max_buyable = self.cash / current_price
            demand = min(demand, max_buyable)
        elif demand < 0: # 매도 시
            demand = max(demand, -self.holdings) # 가지고 있는 주식 이상 팔 수 없음
            
        return demand

    def update_portfolio(self, executed_amount, execution_price):
        """
        체결 결과를 에이전트 포트폴리오(현금, 잔고, 평단가)에 반영
        """
        if executed_amount > 0: # 매수 체결 시
            total_cost = (self.holdings * self.average_cost) + (executed_amount * execution_price)
            self.holdings += executed_amount
            self.cash -= executed_amount * execution_price
            self.average_cost = total_cost / self.holdings if self.holdings > 0 else execution_price
        elif executed_amount < 0: # 매도 체결 시
            self.holdings += executed_amount # executed_amount가 음수이므로 더해줌
            self.cash -= executed_amount * execution_price # 현금 유입


def run_market_simulation(steps=60, market_mix={'Rational': 80, 'Overconfident': 10, 'LossAverse': 10}):
    """
    주어진 투자자 에이전트 조합에 따라 시장 가격 결정 과정을 시뮬레이션
    """
    # 초기화
    agents = []
    agent_id = 0
    for agent_type, count in market_mix.items():
        for _ in range(count):
            if agent_type == 'Rational':
                agents.append(BehavioralInvestorAgent(agent_id, 'Rational', overconfidence=0.05))
            elif agent_type == 'Overconfident':
                agents.append(BehavioralInvestorAgent(agent_id, 'Overconfident', overconfidence=0.8))
            elif agent_type == 'LossAverse':
                agents.append(BehavioralInvestorAgent(agent_id, 'LossAverse', overconfidence=0.2, loss_aversion=2.5))
            agent_id += 1
            
    # 시계열 가격 데이터 저장 리스트
    fundamental_path = []
    price_path = []
    
    # 초기 시장 가격 설정
    current_price = 100.0
    prev_price = 100.0
    fundamental_value = 100.0
    
    price_path.append(current_price)
    fundamental_path.append(fundamental_value)
    
    # 시뮬레이션 루프
    for t in range(1, steps):
        # 1. 내재 가치(Fundamental Value) 업데이트 (무작위 행보 및 특정 시점의 큰 충격)
        noise = np.random.normal(0, 0.4)
        if t == 15:
            # 15시점에 예상치 못한 거시 악재로 내재가치가 급락하는 시나리오
            fundamental_value = fundamental_value - 15.0 + noise
        else:
            fundamental_value = fundamental_value + noise
            
        fundamental_path.append(fundamental_value)
        
        # 2. 모든 에이전트들로부터 매수/매도 순수요 수집
        total_net_demand = 0.0
        demands = {}
        for agent in agents:
            dem = agent.calculate_demand(current_price, prev_price, fundamental_value)
            demands[agent.agent_id] = dem
            total_net_demand += dem
            
        # 3. 시장 균형 가격(Market Clearing Price) 결정
        # 가격 변동폭을 안정적으로 통제하기 위해 클리핑(clipping) 적용
        price_change = np.clip(0.12 * total_net_demand, -12, 12)
        next_price = current_price + price_change + np.random.normal(0, 0.1)
        next_price = max(10.0, next_price) # 가격 하한선 설정
        
        # 4. 개별 에이전트들의 체결 집행 및 자산 업데이트
        for agent in agents:
            dem = demands[agent.agent_id]
            agent.update_portfolio(dem, next_price)
            
        # 가격 변수 갱신
        prev_price = current_price
        current_price = next_price
        price_path.append(current_price)
        
    return price_path, fundamental_path


def execute_and_plot():
    """
    합리적 시장 vs 편향이 가득한 시장(행동경제학적 시장)의 시뮬레이션을 돌려 비교 차트 작성
    """
    steps = 55
    
    # 시나리오 1: 합리적 투자자가 지배적인 시장 (시장 자정 능력이 뛰어남)
    rational_market_mix = {'Rational': 85, 'Overconfident': 10, 'LossAverse': 5}
    prices_rational, fundamentals = run_market_simulation(steps, rational_market_mix)
    
    # 시나리오 2: 과잉 확신과 손실 회피 투자자가 혼재된 시장 (행동경제학적 시장)
    behavioral_market_mix = {'Rational': 30, 'Overconfident': 40, 'LossAverse': 30}
    # 동일한 노이즈 패스를 보장하기 위해 시드 재설정
    np.random.seed(42)
    prices_behavioral, _ = run_market_simulation(steps, behavioral_market_mix)
    
    # 두 시나리오 비교 차트 드로잉 (학습 서버 폰트 제약으로 텍스트는 영문 표기)
    plt.figure(figsize=(12, 6))
    
    plt.plot(fundamentals, label="Asset Fundamental Value", color='black', linestyle='--', linewidth=2)
    plt.plot(prices_rational, label="Rational Dominant Market", color='blue', linewidth=2.5)
    plt.plot(prices_behavioral, label="Behavioral Biased Market", color='red', linewidth=2.5)
    
    # 시나리오 충격 및 버블 구간 시각적 주석 추가
    plt.axvline(x=15, color='gray', linestyle=':', label='Macro Shock Event')
    
    # 버블 및 패닉셀 영역 강조 표시
    plt.annotate('Price Bubble\n(Overconfident Momentum)', xy=(13, 105), xytext=(3, 115),
                 arrowprops=dict(facecolor='orange', shrink=0.05, width=1.5, headwidth=6))
    
    plt.annotate('Panic Selling\n(Loss Averse Triggered)', xy=(18, 75), xytext=(22, 60),
                 arrowprops=dict(facecolor='purple', shrink=0.05, width=1.5, headwidth=6))
    
    # 차트 레이아웃 정밀 가공
    plt.title("Agent-Based Financial Market Simulation: Rational vs Behavioral Investors", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Simulation Timesteps (Weeks / Epochs)", fontsize=11, labelpad=10)
    plt.ylabel("Asset Price", fontsize=11, labelpad=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # 이미지 파일로 우선 세이브 (scratch에 가공)
    chart_path = '/workspace/scratch/behavioral_simulation.png'
    plt.savefig(chart_path, dpi=150)
    plt.close()
    
    print(f"[성공] 시뮬레이션 비교 차트가 생성되었습니다: {chart_path}")
    
    # 학생 배포를 위해 시뮬레이션 요약 데이터 리포트 출력
    print(f"\n[시뮬레이션 통계 요약 (총 {steps}시점)]")
    print(f"- 내재 가치 최종값: {fundamentals[-1]:.2f}")
    print(f"- 합리적 시장 최종값: {prices_rational[-1]:.2f} (내재 가치 대비 괴리도: {abs(prices_rational[-1]-fundamentals[-1])/fundamentals[-1]*100:.2f}%)")
    print(f"- 행동 편향 시장 최종값: {prices_behavioral[-1]:.2f} (내재 가치 대비 괴리도: {abs(prices_behavioral[-1]-fundamentals[-1])/fundamentals[-1]*100:.2f}%)")
    print(f"- 합리적 시장 변동성(표준편차): {np.std(prices_rational):.2f}")
    print(f"- 행동 편향 시장 변동성(표준편차): {np.std(prices_behavioral):.2f}")

if __name__ == "__main__":
    execute_and_plot()
