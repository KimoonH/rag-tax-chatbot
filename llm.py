import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_ai_response(user_question):
    # 1. 환경 변수 로드
    load_dotenv()
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

    # 2. Vector DB 연결
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    index_name = "tax-markdown-index"

    pc = Pinecone(api_key=pinecone_api_key)

    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding
    )

    # 3. LLM 및 프롬프트 설정
    llm = ChatOllama(model="llama3.2")

    # [A] 사전 기반 질문 변환 프롬프트
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    dictionary_prompt = ChatPromptTemplate.from_template(f"""
        당신은 사용자의 질문에서 특정 표현을 사전에 따라 '전문 용어'로 교체하는 변환기입니다.
        
        [변환 규칙]
        {dictionary}
        
        [지시사항]
        1. 질문에서 "사람을 나타내는 표현"이라는 문구가 보이면 무조건 "거주자"로 바꾸세요.
        2. "5천만원" 같은 숫자나 다른 상세 조건은 절대로 건드리지 마세요.
        3. 설명 없이 오직 '변환된 문장'만 다시 출력하세요.

        질문: {{question}}
    """)

    # [B] 세무 전문가 페르소나 RAG 프롬프트
    custom_prompt = ChatPromptTemplate.from_template("""
        당신은 한국의 소득세 전문가입니다. 다음에 제공된 [관련 법령/문서] 내용만을 근거로 사용자의 [질문]에 답변하세요.
        
        [지시사항]
        1. 수치를 계산하거나 조항을 설명할 때는 반드시 문서에 있는 내용만 사용하세요.
        2. 문서에 없는 내용은 "제공된 정보에서는 알 수 없습니다"라고 명확히 답하세요.
        3. 가능한 한 읽기 쉽게 결론부터 말하고, 필요하면 글머리 기호를 사용하세요.

        [관련 법령/문서]: 
        {context}

        [질문]: {question}
        
        [답변]:
    """)

    # 4. LCEL 체인 구성 (Pipeline)
    dictionary_chain = dictionary_prompt | llm | StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    full_chain = (
        {
            "context": dictionary_chain | database.as_retriever() | format_docs, 
            "question": RunnablePassthrough()
        }
        | custom_prompt 
        | llm 
        | StrOutputParser()
    )

    # 5. 실행
    print(f"--- {user_question} ---\n")
    print(f"✅ 검색용 변환 질문: {dictionary_chain.invoke(user_question)}\n")

    result = full_chain.invoke(user_question)
    print(f"✨ 최종 AI 답변:\n{result}")
    
    return result
