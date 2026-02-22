import streamlit as st
from llm import get_ai_response
# 페이지 컴픽

st.set_page_config(page_title="소득세 챗봇", page_icon="💰")

st.title("💰 소득세 챗봇")
st.caption("소득세 관련된 모든것을 답해드립니다.")

if "message_list" not in st.session_state:
    st.session_state.message_list = []

# 이전에 있던 채팅 내용을 기억
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자가 채팅을 입력
if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요."):
    with st.chat_message("user"):
        st.write(user_question)
    # 여기서 다시 그려준다.
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.chat_message("ai"):
        with st.status("답변을 생성 중입니다...", expanded=True) as status:
            ai_message = get_ai_response(user_question)
            status.update(label="답변 생성 완료!", state="complete", expanded=False)
        st.write(ai_message)
    # 여기서 다시 그려준다.
    st.session_state.message_list.append({"role": "ai", "content": ai_message})
