# ==============================================================================
# [실습 가이드] 모듈 2: OpenRouter API를 활용한 다중 모델 금융 시장 분석 및 비교
# 과목명: AI금융 세미나 (2026년 가을학기)
# 담당교수: 정재식 교수 (서강대학교)
# 대상: MBA/GSE 학생 실습용
# ==============================================================================
# 
# [설명]
# 이 파이썬 스크립트는 오픈라우터(OpenRouter) 플랫폼을 사용하여, 
# 단 하나의 OpenAI 패키지만으로 상용 모델(Proprietary)과 오픈소스 경량 모델(SLM)을 
# 유연하게 전환(Swap)하며 동일한 금융 분석 질의에 대한 답변 품질을 비교 평가합니다.
#
# [사전 필수 조건]
# 1. API 키 발급: https://openrouter.ai/ 에 가입 후 API Key를 발급받으세요.
# 2. 라이브러리 설치: 구글 코랩(Google Colab) 또는 로컬 환경에서 아래 명령어를 실행하세요.
#    !pip install openai
# ==============================================================================

import os
from openai import OpenAI

def run_openrouter_comparison():
    # --------------------------------------------------------------------------
    # 1. OpenRouter API 키 설정
    # --------------------------------------------------------------------------
    # 실제 실습 시에는 'YOUR_OPENROUTER_API_KEY_HERE' 부분에 발급받은 API 키를 입력하세요.
    # 보안을 위해 로컬 환경 변수(Environment Variable)에 등록하여 사용하는 것을 권장합니다.
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")
    
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        print("[주의] 실제 OpenRouter API 키가 입력되지 않았습니다.")
        print("스크립트를 실행하려면 29라인의 'YOUR_OPENROUTER_API_KEY_HERE'를 실제 키로 변경하세요.")
        return

    # --------------------------------------------------------------------------
    # 2. OpenAI SDK 호환 클라이언트 초기화
    # --------------------------------------------------------------------------
    # OpenRouter는 OpenAI의 API 명세(Spec)와 100% 호환됩니다.
    # 따라서 base_url을 openrouter.ai로 변경하는 것만으로 모든 모델을 제어할 수 있습니다.
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    # --------------------------------------------------------------------------
    # 3. 금융 경제 가상 시나리오 및 질문(Prompt) 정의
    # --------------------------------------------------------------------------
    financial_prompt = """
    다음은 가상의 신흥국 거시경제 시나리오다:
    '인플레이션율이 전년 대비 8.2% 급등했으며, 중앙은행은 기준금리를 150bp 인상했다. 그러나 정부의 대규모 재정 적자로 인해 국채 금리가 폭등하고 있으며 외환보유고가 빠르게 고갈되고 있다.'

    위 시나리오가 '국내 채권 투자 포트폴리오'와 '외환 리스크 관리'에 미치는 파급 효과를 각각 1줄씩 핵심만 요약하십시오.
    """

    # --------------------------------------------------------------------------
    # 4. 테스트 및 비교할 대상 모델 정의 (상용 vs 오픈소스 SLM)
    # --------------------------------------------------------------------------
    # OpenRouter에서 지원하는 수많은 모델 경로 중 대표적인 두 가지를 선정합니다.
    # - Proprietary: 독점형 상용 최고 성능 모델인 Anthropic Claude 3.5 Sonnet
    # - Open-Source SLM: 메타의 오픈소스 소형 모델인 LLaMA 3.1 8B Instruct
    models_to_test = {
        "1. Proprietary Model (Anthropic Claude 3.5 Sonnet)": "anthropic/claude-3.5-sonnet",
        "2. Open-Source SLM (Meta LLaMA 3.1 8B Instruct)": "meta-llama/llama-3.1-8b-instruct"
    }

    # --------------------------------------------------------------------------
    # 5. 다중 모델 쿼리 수행 및 비교 출력
    # --------------------------------------------------------------------------
    print("\n[시작] 거시경제 시나리오 분석 모델 비교 분석\n")
    print(f"**분석 시나리오**:\n{financial_prompt.strip()}\n")

    for label, model_path in models_to_test.items():
        print(f"\n" + "="*70)
        print(f" {label}")
        print(f" 모델 ID: {model_path}")
        print("="*70)
        
        try:
            # chat.completions.create 호출 방식을 그대로 사용합니다.
            response = client.chat.completions.create(
                model=model_path,
                messages=[
                    {
                        "role": "system", 
                        "content": "너는 글로벌 거시경제 채권 펀드매니저이자 리스크 관리 전문가이다. 불필요한 서론 없이 본론만 명확히 답변하라."
                    },
                    {
                        "role": "user", 
                        "content": financial_prompt
                    }
                ],
                # OpenRouter 전용 선택적 메타데이터 추가 (통계 및 랭킹 분석용)
                extra_headers={
                    "HTTP-Referer": "https://sogang.ac.kr",
                    "X-Title": "Sogang AI Finance Seminar",
                }
            )
            # 답변 출력
            answer = response.choices[0].message.content.strip()
            print(answer)
            
        except Exception as e:
            print(f"모델 [{model_path}] 호출 중 오류 발생: {e}")
            print("API 키의 유효성을 점검하거나, OpenRouter 크레딧 잔액을 확인해 주세요.")

if __name__ == "__main__":
    run_openrouter_comparison()
