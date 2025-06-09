import streamlit as st
import speech_recognition as sr
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("마이크에 말씀해 주세요...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language="ko-KR")
            st.success(f"인식 결과: {text}")
            return text
        except sr.UnknownValueError:
            st.error("음성을 인식하지 못했습니다.")
            return None
        except sr.RequestError as e:
            st.error(f"음성 인식 서비스 오류: {e}")
            return None

st.title("개인용 비서 서비스")
st.subheader("텍스트 또는 음성 입력 → 라마 3.1 기반 답변")

input_mode = st.radio("입력 방식", ("텍스트", "음성"))
input_text = ""

if input_mode == "텍스트":
    input_text = st.text_input("비서에게 할 말을 입력하세요:")
elif input_mode == "음성":
    if st.button("음성 입력 시작"):
        input_text = recognize_speech()

if input_text:
    prompt = ChatPromptTemplate.from_template(
        "당신은 개인 비서입니다. 사용자의 요청을 이해하고, 친절하고 명확하게 답변하세요.\n요청: {message}\n답변:"
    )
    llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"message": input_text})
    st.write("비서 답변:")
    st.write(response)

    # 대화 이력 저장 (선택)
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(("사용자", input_text))
    st.session_state.history.append(("비서", response))

    # 대화 이력 출력 (선택)
    st.subheader("대화 이력")
    for role, msg in st.session_state.history:
        st.write(f"{role}: {msg}")

