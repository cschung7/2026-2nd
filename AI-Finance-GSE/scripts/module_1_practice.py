# -*- coding: utf-8 -*-
"""
================================================================================
[AI금융 세미나 - 2026년 가을학기] 
모듈 1: 금융 AI 입문 및 환경 점검 (학생 실습용 핸즈온 가이드)
--------------------------------------------------------------------------------
담당교수: 정재식 교수 (서강대학교)
지향점: "Mimic & Reproduce" (모방과 재현)을 통한 금융 데이터 분석 도구 마스터
================================================================================

이 스크립트는 1주차 수업에서 다루는 필수 금융 라이브러리(Pandas, yfinance, Matplotlib)의
기초 점검을 위한 실습 코드입니다. Google Colab 또는 로컬 Python 환경(VS Code 등)에서
차례대로 복사하여 실행해 보시기 바랍니다.

* 실습 전에 필수 라이브러리를 설치해 주세요:
  !pip install pandas yfinance matplotlib
"""

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 실습 1. Pandas를 활용한 금융 데이터프레임 핸들링 및 일별 수익률 계산
# ------------------------------------------------------------------------------
def run_pandas_practice():
    print("\n" + "="*55)
    print("[실습 1] Pandas 기초 핸들링 & 일별 수익률 계산")
    print("="*55)
    
    # 1.1 샘플 시계열 데이터 생성 (날짜와 KOSPI 지수)
    data = {
        'Date': ['2026-09-01', '2026-09-02', '2026-09-03', '2026-09-04', '2026-09-07'],
        'KOSPI_Close': [2550.50, 2562.10, 2548.90, 2570.30, 2585.10]
    }
    df = pd.DataFrame(data)
    print("1. 원본 데이터프레임 (시계열 인덱스 설정 전):")
    print(df)
    print()
    
    # 1.2 Date 컬럼을 datetime 타입으로 변환 및 인덱스 설정 (금융 시계열 분석의 약속)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # 1.3 일별 수익률(Daily Return) 계산: pct_change() 사용
    # 공식: (오늘 가격 - 어제 가격) / 어제 가격
    df['Daily_Return'] = df['KOSPI_Close'].pct_change()
    
    print("2. 변환 후 데이터프레임 (시계열 인덱스 적용 및 일별 수익률 산출):")
    print(df)
    print("-"*55)


# ------------------------------------------------------------------------------
# 실습 2. yfinance를 활용한 실시간 및 역사적 시장 데이터 수집
# ------------------------------------------------------------------------------
def run_yfinance_practice():
    print("\n" + "="*55)
    print("[실습 2] yfinance 활용 금융 데이터 수집")
    print("="*55)
    
    # 2.1 삼성전자(005930.KS) 티커 설정
    # *꿀팁: 한국 코스피 종목은 뒤에 '.KS', 코스닥 종목은 '.KQ'를 붙입니다. (예: 카카오 '035720.KS')
    ticker_symbol = "005930.KS"
    samsung = yf.Ticker(ticker_symbol)
    
    # 2.2 최근 1개월(1mo) 동안의 일별 시세 데이터 다운로드
    print(f"야후 파이낸스(Yahoo Finance)에서 '{ticker_symbol}'의 최근 1개월 데이터를 수집하는 중...")
    hist = samsung.history(period="1mo")
    
    # 2.3 수집된 데이터의 형태 및 상위 5행 출력
    print(f"\n데이터 크기 (행, 열): {hist.shape}")
    print("\n삼성전자 주가 데이터 (상위 5행):")
    print(hist[['Open', 'High', 'Low', 'Close', 'Volume']].head())
    print("-"*55)
    
    return hist


# ------------------------------------------------------------------------------
# 실습 3. Matplotlib를 활용한 금융 시계열 데이터 시각화
# ------------------------------------------------------------------------------
def run_visualization_practice(hist, save_only=False):
    print("\n" + "="*55)
    print("[실습 3] Matplotlib를 활용한 주가 차트 시각화")
    print("="*55)
    
    if hist.empty:
        print("시각화할 데이터가 존재하지 않습니다.")
        return
        
    print("수집한 삼성전자 종가(Close) 추이를 시각화합니다...")
    
    # 3.1 캔버스 생성 및 크기 지정 (가로 10인치, 세로 5인치)
    plt.figure(figsize=(10, 5))
    
    # 3.2 선 그래프(Line Plot) 생성 및 디자인 설정
    plt.plot(hist.index, hist['Close'], marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Close Price (KRW)')
    
    # 3.3 차트 부가 요소 구성 (제목, 레이블, 범례, 그리드)
    plt.title("Samsung Electronics (005930.KS) - Past 1 Month Trend", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Price (KRW)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    
    # 3.4 x축 날짜 레이블 가독성 개선 (기울기 조절)
    plt.gcf().autofmt_xdate()
    
    # 3.5 실행 환경에 따른 차트 출력 분기
    if save_only:
        # 헤드리스(샌드박스) 환경용 이미지 저장
        output_path = "/workspace/scratch/samsung_close_price.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"차트가 이미지 파일('{output_path}')로 정상 저장되었습니다.")
    else:
        # 학생들의 로컬 PC 또는 Google Colab용 화면 출력
        plt.show()
        print("차트가 화면에 정상 출력되었습니다.")
        
    print("="*55)


if __name__ == "__main__":
    print("="*80)
    print(" 서강대학교 MBA/GSE [AI금융 세미나] - 1주차 파이썬 실습 코드 검증 시작")
    print("="*80)
    
    # 1. Pandas 실습 실행
    run_pandas_practice()
    
    # 2. yfinance 데이터 다운로드 실습 실행
    samsung_hist = run_yfinance_practice()
    
    # 3. 시각화 실습 실행 (샌드박스 검증을 위해 save_only=True로 설정하여 실행)
    if not samsung_hist.empty:
        # 샌드박스 환경에서는 GUI 창을 띄울 수 없으므로, 내부 검증 시에는 이미지 저장 옵션을 켭니다.
        # 학생들이 복사해서 개인 Colab 등에서 실행할 때는 save_only=False(기본값)로 두시면 화면에 바로 그래프가 출력됩니다.
        run_visualization_practice(samsung_hist, save_only=True)
        
    print("\n[성공] 모든 실습 코드 블록이 정상 작동하고 검증을 완료했습니다.")
    print("="*80)
