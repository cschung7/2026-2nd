# ==============================================================================
# [실습 가이드] 모듈 3: 파이썬으로 구현하는 금융 RAG(검색 증강 생성) 핵심 파이프라인
# 과목명: AI금융 세미나 (2026년 가을학기)
# 담당교수: 정재식 교수 (서강대학교)
# 대상: MBA/GSE 학생 실습용
# ==============================================================================
# 
# [설명]
# 이 스크립트는 RAG(Retrieval-Augmented Generation)의 핵심 4단계 아키텍처를
# 외부 복잡한 벡터 DB(Vector Database) 없이, 파이썬 기본 라이브러리와 scikit-learn만으로
# 로컬에서 직관적으로 재현하여 이해할 수 있도록 설계되었습니다.
#
# [실습 흐름]
# 1. Chunking (문서 분할): 가상의 기업 공시 주석 문서를 의미 있는 단락 단위로 쪼갭니다.
# 2. Embedding & Indexing (TF-IDF 벡터화): 각 단락을 수학적 벡터로 변환하여 검색 가능한 상태로 만듭니다.
# 3. Retrieval (유사도 검색): 질문과 가장 연관성이 높은 단락을 코사인 유사도(Cosine Similarity)로 찾아냅니다.
# 4. Grounding & Generation (출처 기반 답변): 찾아낸 단락만을 LLM에 참고 자료로 제공하여 환각 없는 정확한 답변을 유도합니다.
#
# [사전 필수 조건]
# !pip install openai scikit-learn pandas numpy
# ==============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# --------------------------------------------------------------------------
# [준비 단계] 가상의 금융 공시 데이터 (Mock Financial Document)
# --------------------------------------------------------------------------
# 실제 실무에서의 PDF나 금감원 DART 공시 텍스트 역할을 하는 가상의 문서입니다.
# 고의로 다양한 주제의 재무 정보와 단서 조항(Caveats)을 섞어 두었습니다.
CORPORATE_FILING = """
[문서 ID: CHUNK_1 - 개요 및 실적]
서강전자의 2026년 2분기 영업이익은 전년 동기 대비 14.5% 증가한 4,250억 원을 기록하였습니다. 반도체 신규 라인 가동율 상승이 주된 실적 개선 요인입니다.

[문서 ID: CHUNK_2 - 설비투자 및 R&D]
당분기 설비투자(CAPEX)는 총 1조 2,300억 원이 집행되었습니다. 연구개발(R&D) 비용은 1,200억 원으로 전년 대비 50% 급증한 것처럼 보이나, 이 중 400억 원은 특허 소송 합의금 일시 반영분입니다. 생산 설비 투자는 자금 확보 문제로 당분기 중 일부 라인이 잠정 중단되었습니다.

[문서 ID: CHUNK_3 - 재무 건전성 및 특별조항]
사채 발행 계약서상의 재무약정 비율(Covenants)인 부채비율 150% 이하 유지 조항은 안정적으로 준수되고 있습니다. 현재 부채비율은 98.4%입니다. 단, 주력 납품처의 대금 지급이 3개월 이상 지연될 경우 유동성 경고 레벨로 진입하게 됩니다.

[문서 ID: CHUNK_4 - 거시경제 리스크]
최근 신흥국의 인플레이션율 급등(8.2%) 및 원자재 가격 불안정은 해외 공장의 부품 조달 원가를 상승시키는 잠재적 위협 요인입니다.
"""

def simple_rag_demo():
    print("==============================================================================")
    print(" 서강대학교 MBA AI금융 세미나 - 모듈 3 RAG 실전 파이프라인 시뮬레이터")
    print("==============================================================================\n")

    # --------------------------------------------------------------------------
    # 1. Chunking (문서 분할)
    # --------------------------------------------------------------------------
    # 원본 문서를 단락(Chunk) 단위로 쪼개어 리스트로 만듭니다.
    chunks = [chunk.strip() for chunk in CORPORATE_FILING.strip().split("\n\n") if chunk.strip()]
    
    print(f"[Step 1] Chunking 완료: 총 {len(chunks)}개의 단락으로 문서를 분할했습니다.")
    for i, chunk in enumerate(chunks):
        print(f"  └─ Chunk {i+1} ({len(chunk)}자): {chunk[:40]}...")
    print("-" * 80)

    # --------------------------------------------------------------------------
    # 2. Embedding & Indexing (수학적 벡터 인덱싱)
    # --------------------------------------------------------------------------
    # 문장 간의 '의미 체계'와 '키워드 매칭'을 동시에 계산하기 위해 TF-IDF 벡터라이저를 생성합니다.
    # 실무 벡터 DB에서는 이 부분을 'OpenAI Embedding'이나 'BGE Embedding' 등으로 대체합니다.
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)
    
    print(f"[Step 2] Embedding & Indexing 완료: 단락들을 수학적 고차원 공간 벡터로 임베딩했습니다.")
    print(f"  └─ 인덱싱된 단어 사전 크기: {len(vectorizer.get_feature_names_out())}개 단어")
    print("-" * 80)

    # --------------------------------------------------------------------------
    # 3. Retrieval (유사도 검색 구현)
    # --------------------------------------------------------------------------
    # 사용자 질문을 접수하고, 코사인 유사도가 가장 높은 문단을 검색해 내는 핵심 엔진입니다.
    def retrieve_context(query, top_k=1):
        # 질문을 동일한 벡터 공간으로 변환
        query_vector = vectorizer.transform([query])
        # 인덱싱된 모든 단락 벡터들과의 유사도 계산
        similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
        # 가장 높은 유사도를 가진 인덱스 추출
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        retrieved_results = []
        for idx in top_indices:
            retrieved_results.append({
                "chunk_id": idx + 1,
                "text": chunks[idx],
                "score": similarities[idx]
            })
        return retrieved_results

    # 실습용 테스트 질문 정의
    user_query = "연구개발비용 R&D가 갑자기 크게 늘어난 구체적인 원인이 뭐야?"
    print(f"[Step 3] Retrieval 실행: 분석가의 질문을 분석합니다.")
    print(f"  └─ 질문(Query): \"{user_query}\"")
    
    retrieved = retrieve_context(user_query, top_k=1)[0]
    print(f"  └─ 검색 성공! (가장 유사한 단락 포착)")
    print(f"     [선정된 Chunk {retrieved['chunk_id']} (유사도 점수: {retrieved['score']:.4f})]")
    print(f"     내용: {retrieved['text']}")
    print("-" * 80)

    # --------------------------------------------------------------------------
    # 4. Grounding & Generation (출처 기반 프롬프트 주입 및 답변 생성)
    # --------------------------------------------------------------------------
    # 검색된 사실(Grounding Context)을 LLM에 주입하여, 환각 현상을 원천 차단하는 장벽을 칩니다.
    
    # OpenRouter API 설정 (이전 모듈 실습 기반)
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")
    
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        print("[주의] OpenRouter API 키가 등록되지 않았습니다.")
        print("이후 LLM 답변 생성(Step 4) 실습을 진행하려면 코드 95라인에 실제 API 키를 입력해 주세요.")
        print("검색된 단락(Chunk 2)의 실물 내용 비교로 이번 실습을 종료합니다.")
        return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    # [중요] 금융 RAG 전용 '근거 그라운딩(In-context Grounding)' 프롬프트 템플릿
    # 오직 제공된 검색 결과 범위 내에서만 답변하고, 상상력을 발휘하지 말 것을 지시(Constraint)합니다.
    system_instruction = """
    너는 최고 권위의 금융 투자 분석가이자 금융 리스크 감사원이다.
    반드시 제공된 [Reference Context] 안의 구체적인 사실에만 기반하여 답변하십시오.
    만약 제공된 콘텍스트 내에서 질문에 대한 명확한 답을 찾을 수 없거나 수치가 언급되어 있지 않다면, 
    스스로 추정하거나 지식을 지어내지 말고 "제공된 공시 문서에서 근거를 찾을 수 없습니다"라고 정직하게 답하십시오.
    답변 끝에는 반드시 인용한 문서 ID(예: [CHUNK_N])를 표기하여 출처를 명시하십시오.
    """

    rag_prompt = f"""
    [Reference Context]
    {retrieved['text']}

    [User Question]
    {user_query}
    """

    print(f"[Step 4] Grounding & Generation 실행: 최신 LLM(Claude 3.5 Sonnet)에 검색된 지식을 밀어 넣습니다.")
    try:
        response = client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[
                {"role": "system", "content": system_instruction.strip()},
                {"role": "user", "content": rag_prompt.strip()}
            ],
            extra_headers={
                "HTTP-Referer": "https://sogang.ac.kr",
                "X-Title": "Sogang AI Finance Seminar Module 3 RAG",
            }
        )
        print("\n" + "="*35 + " [최종 LLM 답변 출력] " + "="*35)
        print(response.choices[0].message.content.strip())
        print("="*92 + "\n")
        
        # --------------------------------------------------------------------------
        # 비판적 성찰 유도 (MBA 세미나용 디스커션 포인트)
        # --------------------------------------------------------------------------
        print("[교수님 주석 및 비판적 성찰 포인트]")
        print(" * 단순 LLM 질문 시: 'R&D 투자액이 증가한 것은 혁신 성장과 미래 동력을 확보하기 위한 적극적 경영 활동...' 이라는 그럴듯한 소설을 씁니다.")
        print(" * RAG 파이프라인 가동 시: '실제로는 R&D 비용 1,200억 중 400억이 소송 합의금 일시 반영분이며, 생산 설비 투자는 잠정 중단되었다'는 숨겨진 내부 사정을 완벽히 집어내 환각을 차단합니다.")
        
    except Exception as e:
        print(f"API 호출 중 에러가 발생했습니다: {e}")

if __name__ == "__main__":
    simple_rag_demo()
