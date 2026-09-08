# ==============================================================================
# [실습 가이드] 모듈 2: OpenRouter API 및 Hugging Face 로컬 금융 뉴스 감성 분석
# 과목명: AI금융 세미나 (2026년 가을학기)
# 담당교수: 정재식 교수 (서강대학교)
# 대상: MBA/GSE 학생 실습용 (버전 2)
# ==============================================================================
# 
# [설명]
# 이 파이썬 스크립트는 오픈소스 및 대규모 언어 모델 실습을 위한 두 가지 파트로 구성되어 있습니다.
#
# 파트 1: OpenRouter API 활용 (상용 모델 vs 오픈소스 초소형 모델 품질 비교)
# 파트 2: Hugging Face transformers 라이브러리 활용 (로컬 보안망 내 금융 감성 분석 구동)
#
# [설치해야 할 패키지]
# 구글 코랩(Google Colab) 또는 로컬 아나콘다 환경에서 아래 명령어를 복사해 실행해 주세요:
# !pip install openai transformers torch sentencepiece
# ==============================================================================

import os
import sys

# --------------------------------------------------------------------------
# [파트 1] OpenRouter API를 활용한 다중 모델 분석 비교
# --------------------------------------------------------------------------
def run_openrouter_comparison():
    try:
        from openai import OpenAI
    except ImportError:
        print("[오류] 'openai' 라이브러리가 설치되어 있지 않습니다.")
        print("설치 명령어: pip install openai")
        return

    # 1. OpenRouter API 키 설정
    # 아래 'YOUR_OPENROUTER_API_KEY_HERE'를 발급받은 실제 API 키로 교체해 주세요.
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")
    
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        print("\n" + "="*80)
        print("[파트 1 건너뜀] 실제 OpenRouter API 키가 입력되지 않았습니다.")
        print("API 키를 적용하려면 코드의 'YOUR_OPENROUTER_API_KEY_HERE' 부분을 변경하십시오.")
        print("="*80 + "\n")
        return

    # OpenAI-Compatible SDK 설정
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    # 거시경제 가상 시나리오 및 프롬프트 정의
    financial_prompt = """
    다음은 가상의 신흥국 거시경제 시나리오다:
    '인플레이션율이 전년 대비 8.2% 급등했으며, 중앙은행은 기준금리를 150bp 인상했다. 그러나 정부의 대규모 재정 적자로 인해 국채 금리가 폭등하고 있으며 외환보유고가 빠르게 고갈되고 있다.'

    위 시나리오가 '국내 채권 투자 포트폴리오'와 '외환 리스크 관리'에 미치는 파급 효과를 각각 1줄씩 핵심만 요약하십시오.
    """

    # 비교할 상용 모델과 오픈소스 경량 모델 정의
    models_to_test = {
        "1. Proprietary Model (Anthropic Claude 3.5 Sonnet)": "anthropic/claude-3.5-sonnet",
        "2. Open-Source SLM (Meta LLaMA 3.1 8B Instruct)": "meta-llama/llama-3.1-8b-instruct"
    }

    print("\n" + "="*80)
    print(" [파트 1] OpenRouter 다중 모델 비교 분석 수행")
    print("="*80)
    print(f"* 분석 시나리오:\n{financial_prompt.strip()}\n")

    for label, model_path in models_to_test.items():
        print(f"\n>> {label}")
        print(f"   모델 경로: {model_path}")
        print("-" * 50)
        try:
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
                extra_headers={
                    "HTTP-Referer": "https://sogang.ac.kr",
                    "X-Title": "Sogang AI Finance Seminar",
                }
            )
            print(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"API 호출 중 오류 발생: {e}")


# --------------------------------------------------------------------------
# [파트 2] Hugging Face Transformers를 활용한 로컬 감성 분석 (FinBERT)
# --------------------------------------------------------------------------
def run_huggingface_sentiment_analysis():
    print("\n" + "="*80)
    print(" [파트 2] Hugging Face 로컬 금융 뉴스 감성 분석 (FinBERT)")
    print("="*80)
    print("* 최초 구동 시 모델 다운로드가 진행되므로 수십 초~수 분의 시간이 소요될 수 있습니다.")
    print("* 이 기능은 별도의 API 키나 과금 없이 로컬 환경에서 완전히 독립적으로 동작합니다.\n")

    try:
        from transformers import pipeline
    except ImportError:
        print("[오류] 'transformers' 라이브러리가 설치되어 있지 않습니다.")
        print("설치 명령어: pip install transformers torch sentencepiece")
        return

    try:
        # 1. Hugging Face 허브로부터 금융 뉴스 전용 감성 분석 모델(FinBERT) 파이프라인 생성
        # 모델명: 'ProsusAI/finbert' (금융 리서치 분석가들이 가장 범용적으로 쓰는 벤치마크 모델 중 하나)
        print("... FinBERT 모델 가중치 및 토크나이저 로딩 중 ...")
        classifier = pipeline(
            "sentiment-analysis", 
            model="ProsusAI/finbert"
        )
        print("... 모델 로딩 완료! 분석을 시작합니다 ...\n")

        # 2. 분석 대상 금융 뉴스 헤드라인 정의
        headlines = [
            "Samsung Electronics reports record-high quarterly revenue driven by strong semiconductor demand.",
            "Concerns grow over rising household debt and potential interest rate hikes by the central bank.",
            "The stock price remained unchanged ahead of the Federal Reserve's monetary policy announcement."
        ]

        # 3. 로컬 분석 수행
        results = classifier(headlines)

        # 4. 결과 출력 (감성 태그 및 매칭 예측 스코어)
        for i, (headline, result) in enumerate(zip(headlines, results), 1):
            print(f"뉴스 #{i}: {headline}")
            # FinBERT의 기본 라벨: positive(긍정), negative(부정), neutral(중립)
            label = result['label'].upper()
            score = result['score']
            print(f"   ▶ 감성 분석 결과: {label} (신뢰 확률: {score:.4f})\n")

    except Exception as e:
        print(f"Hugging Face 모델 구동 중 오류 발생: {e}")
        print("만약 로컬망 환경이나 메모리 제약이 있다면 가동이 중단될 수 있습니다.")


if __name__ == "__main__":
    # 파트 2(Hugging Face)를 먼저 실행해 로컬 무상 실습의 직관을 제공하고,
    # 파트 1(OpenRouter API)로 다중 상용 모델과의 유연한 클라우드 결합을 비교하는 흐름으로 구성하면 효과적입니다.
    
    print("서강대학교 AI금융 세미나 - 모듈 2 실습 스크립트 가동")
    run_huggingface_sentiment_analysis()
    run_openrouter_comparison()
