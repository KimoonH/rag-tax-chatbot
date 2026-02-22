import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from config import answer_examples

def get_dictionary_chain(llm):
    # [A] 사전 기반 질문 변환 (Query Transformation) 프롬프트
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
    return dictionary_prompt | llm | StrOutputParser()

def get_llm(model_name="llama3.2"):
    return ChatOllama(model=model_name)

def get_retriever():
    load_dotenv()
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    
    # Vector DB 연결
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    index_name = "tax-markdown-index"

    pc = Pinecone(api_key=pinecone_api_key)

    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding
    )
    return database.as_retriever()

def get_history_aware_retriever(llm, retriever):
    # [A] 질문 재구성 프롬프트
    contextualize_q_system_prompt = """
    당신은 채팅 기록과 최신 사용자 질문을 사용하여 질문을 재구성하는 AI 조우입니다.
    사용자 질문이 채팅 기록의 맥락을 참조하는 경우, 채팅 기록 없이도 이해할 수 있는 단독 질문으로 재구성하세요.
    질문에 답변하지 말고 필요에 따라 질문을 재구성하기만 하고 그렇지 않으면 질문을 그대로 반환하세요.
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

def get_qa_chain(llm):
    # [B] 세무 전문가 페르소나 RAG 프롬프트
    system_prompt = """
    당신은 한국의 소득세 전문가입니다. 다음에 제공된 [관련 법령/문서] 내용만을 근거로 사용자의 [질문]에 답변하세요.
    
    [지시사항]
    1. 수치를 계산하거나 조항을 설명할 때는 반드시 문서에 있는 내용만 사용하세요.
    2. 문서에 없는 내용은 "제공된 정보에서는 알 수 없습니다"라고 명확히 답하세요.
    3. 가능한 한 읽기 쉽게 결론부터 말하고, 필요하면 글머리 기호를 사용하세요.

    [관련 법령/문서]: 
    {context}
    """
    
    # 예시 프롬프트 템플릿 정의
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )

    # Few-shot 프롬프트 템플릿 생성
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt, # Few-shot 예시 주입
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return create_stuff_documents_chain(llm, qa_prompt)

def get_ai_response(user_question, chat_history):
    # 1. Retriever 및 LLM 설정
    retriever = get_retriever()
    llm = get_llm()

    # 변환 체인 및 QA 체인 가져오기
    # dictionary_chain = get_dictionary_chain(llm) # 질문 변환 체인은 기록 기반 검색기로 대체 가능하거나 조합 필요
    
    # 2. History-aware Retriever 생성
    history_aware_retriever = get_history_aware_retriever(llm, retriever)
    
    # 3. QA Chain (Documents Chain) 생성
    question_answer_chain = get_qa_chain(llm)
    
    # 4. 최종 Conversational RAG 체인 구성
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 5. 실행
    print(f"--- {user_question} ---\n")
    # print(f"✅ 검색용 변환 질문: {dictionary_chain.invoke(user_question)}\n")

    result = rag_chain.invoke({"input": user_question, "chat_history": chat_history})
    print(f"✨ 최종 AI 답변:\n{result['answer']}")
    
    return result['answer']
