# 🚀 RAG 기반 세무 챗봇 구축 완벽 가이드라인

로컬 및 클라우드(Streamlit Cloud) 환경 모두에서 안정적으로 작동하는 **RAG 기반 세무 챗봇(Memory + Few-shot 적용)**을 처음부터 다시 만든다면, 다음과 같은 단계별 가이드라인을 따르시는 것을 권장합니다.

특히 상용 서비스 배포 시 발생할 수 있는 '로컬 LLM 접속 오류'를 방지하기 위해, **클라우드 API(OpenAI 등) 사용**을 전제로 작성되었습니다.

---

## 1단계: 프로젝트 초기 환경 설정 (환경의 독립성 보장)

가장 먼저 할 일은 프로젝트 폴더를 만들고, 파이썬 환경을 격리하며, 필수 라이브러리를 설치하는 것입니다.

1. **프로젝트 폴더 생성 및 이동**
2. **가상환경(Virtual Environment) 생성 및 활성화**
    - 예: `python -m venv .venv` 실행 후 `source .venv/bin/activate` (Mac/Linux) 또는 `.venv\Scripts\activate` (Windows)
3. **핵심 패키지 설치 (`requirements.txt` 관리)**
    - 초기부터 패키지 목록을 관리해야 나중에 클라우드 배포 시 에러가 나지 않습니다.
    - 패키지 스택: `streamlit`, `langchain-openai`, `langchain-pinecone`, `langchain-classic`, `pinecone-client`, `python-dotenv` 등
    - 패키지 설치 후 `pip freeze > requirements.txt` 명령어로 의존성 파일 저장.
4. **환경 변수 파일 (`.env`) 생성**
    - `OPENAI_API_KEY`, `PINECONE_API_KEY` 등을 저장하여 보안 유지.
5. **`.gitignore` 설정**
    - `.env`, `.venv`, `__pycache__` 등이 GitHub에 올라가지 않도록 가장 먼저 설정합니다.

---

## 2단계: 데이터 전처리 및 벡터 DB 저장 (1회성 스크립트 분리)

챗봇 접속 속도를 높이기 위해, 데이터를 자르고(Chunking) Pinecone에 넣는 작업은 `chat.py`가 아닌 **별도의 파일(`ingest.py` 등)**에서 단 한 번만 실행되도록 구성합니다.

1. **문서 로드 (Document Loaders)**: 법령 등의 문서(PDF, Word, Markdown 등)를 파이썬으로 불러옵니다.
2. **문서 분할 (Text Splitters)**: `RecursiveCharacterTextSplitter`를 사용해 문서를 적절한 크기(Chunk)로 일정하게 자릅니다.
3. **임베딩 및 Pinecone 저장**: 
    - OpenAI 임베딩 모델(예: `text-embedding-3-small`)을 사용합니다.
    - `PineconeVectorStore.from_documents()`를 사용하여 쪼갠 내용들을 벡터 DB에 업로드합니다.
    - *이 작업이 끝나면 해당 스크립트는 더 이상 실행할 필요가 없습니다.*

---

## 3단계: AI 뇌(LLM & RAG 로직) 구축 (모듈화)

UI 로직과 분리하여 독립적으로 테스트 및 유지보수 할 수 있도록 `llm.py` 와 `config.py`를 만듭니다.

1. **`config.py` 구성 (Few-shot)**
    - AI가 대답할 이상적인 포맷(`answer_examples` 딕셔너리 리스트)을 분리하여 저장합니다.
2. **`llm.py` 구성 (체인 컴포넌트 분리)**
    - `get_llm()`: OpenAI 모델(`gpt-4o-mini` 등) 호출.
    - `get_retriever()`: 저장된 Pinecone 인덱스를 `from_existing_index()`로 연결.
    - `get_history_aware_retriever()`: 이전 질문(예: "그럼 저축은요?")의 맥락을 고려한 독립적인 질문을 생성하도록 설정.
    - `get_qa_chain()`: `config.py`의 Few-shot 예시를 주입하고, 시스템 프롬프트(페르소나)를 더해 답변 생성 체인 구성.
    - `get_ai_response(user_question, chat_history)`: 위 요소들을 `create_retrieval_chain`으로 통합하고 파라미터를 받아 최종 텍스트만 리턴.

---

## 4단계: Streamlit 사용자 인터페이스 (UI) 구축

복잡한 LangChain 로직은 숨기고, `chat.py`에서는 사용자와의 대화 렌더링에만 집중합니다.

1. **페이지 기본 설정**: 제목(`st.title`), 캡션, 아이콘 등을 설정합니다.
2. **세션 상태(`st.session_state`) 초기화**: 대화 기록(`message_list`)을 담을 리스트 객체 생성.
3. **과거 대화 렌더링**: 초기화된 리스트를 `st.chat_message` 반복문으로 돌며 화면에 말풍선들을 다시 그립니다.
4. **사용자 입력 처리 (`st.chat_input`)**: 
    - 사용자 말풍선을 맨 밑에 그리고 세션 상태 리스트에 dict 형태로 추가합니다.
    - 세션 상태에 저장된 기록들을 순회하며 LangChain 객체(`HumanMessage`, `AIMessage`)로 포맷팅하여 `chat_history` 리스트로 만듭니다.
    - 스피너(`st.status`)를 띄우고 `llm.py`의 `get_ai_response()` 함수를 모델 백그라운드에서 돌립니다.
    - AI 결과가 도착하면 스피너를 끄고 AI 말풍선을 출력한 뒤, 성공한 결과를 세션 상태에 추가합니다.

---

## 5단계: 클라우드 배포 (Streamlit Cloud 기준)

완성된 코드를 대중에게 공개합니다. OpenAI 모델을 사용하므로 외부 서버에서도 원활하게 돌아갑니다.

1. **GitHub Push**: 지금까지 작성한 문서들(`chat.py`, `llm.py`, `config.py`, `requirements.txt`)을 `main` 브랜치로 푸시합니다. **(주의: `.env` 파일 제외 필수!)**
2. **Streamlit Cloud 연결**: Streamlit 계정을 GitHub와 연동하고, 레포지토리를 선택한 후 메인 파일을 `chat.py`로 지정합니다.
3. **Secrets 설정 (가장 중요 ⭐️)**:
    - Streamlit 배포 세팅(Advanced Settings ➡️ Secrets) 란에 로컬 `.env` 에 있던 내용을 복사/붙여넣기 합니다. 
    - 이 단계가 누락되면 클라우드 서버에서 API 권한 오류가 발생합니다.
4. **Deploy 및 테스트**: 배포 스크립트 실행 후 에러 없이 웹 화면이 나오고 RAG 및 메모리가 정상 작동하는지 확인합니다.
