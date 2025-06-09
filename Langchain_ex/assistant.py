################################################################################
# 1. 라이브러리 임포트 및 기본 설정
################################################################################
import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import io
import os
import platform
import pyttsx3

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

################################################################################
# 2. Whisper 음성 인식 모델 초기화
################################################################################
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

def recognize_speech_web(audio_bytes):
    try:
        audio_bio = io.BytesIO(audio_bytes)
        audio_bio.name = 'audio.webm'
        result = whisper_model.transcribe(audio_bio, language="ko")
        return result["text"]
    except Exception as e:
        st.error(f"음성 인식 오류: {e}")
        return None

################################################################################
# 3. LLM 및 기본 대화 체인 구성
################################################################################
llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434", temperature=0.7)

prompt = ChatPromptTemplate.from_template(
    """당신은 전문 개인 비서입니다. 다음 규칙을 준수하세요:
    1. 사용자 요청을 정확히 이해해 답변
    2. 친절한 말투 사용 (존댓말)
    3. 답변은 3문장 이내로 간결하게

    요청: {message}
    답변:"""
)

base_chain = prompt | llm | StrOutputParser()

################################################################################
# 4. 대화 메모리 및 메시지 기반 체인 구성
################################################################################
memory = ConversationBufferMemory(return_messages=True)

conversation = RunnableWithMessageHistory(
    base_chain,
    lambda session_id: memory,
    input_messages_key="message",
    history_messages_key="history"
)

################################################################################
# 5. RAG 문서 검색 체인 구성
################################################################################
documents = [
    Document(page_content="회의는 매주 월요일 오후 3시에 진행됩니다."),
    Document(page_content="개인 비서는 사용자의 요청을 친절하게 처리합니다.")
]

embeddings = OllamaEmbeddings(model="llama3.1", base_url="http://localhost:11434")
db = FAISS.from_documents(documents, embeddings)
retriever = db.as_retriever()

rag_prompt = ChatPromptTemplate.from_template("문맥: {context}\n질문: {question}\n답변:")
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt | llm | StrOutputParser()
)

################################################################################
# 6. LangChain Agent Tool 구성
################################################################################
tools = [
    Tool(
        name="DraftEmail",
        func=lambda input: base_chain.invoke({"message": f"다음 내용으로 이메일을 작성해주세요: {input}"}),
        description="내용을 입력하면 이메일을 작성해드립니다."
    ),
    Tool(
        name="ScheduleMeeting",
        func=lambda input: base_chain.invoke({"message": f"{input} 일정 잡아줘"}),
        description="회의 일정을 조율합니다."
    ),
    Tool(
        name="QnA",
        func=lambda input: base_chain.invoke({"message": input}),
        description="일반 질문에 답변합니다."
    )
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory)

################################################################################
# 7. Streamlit UI 구성
################################################################################
st.title("🤖 개인용 AI 비서 서비스")
st.subheader("텍스트/음성 입력 → Llama 3.1 기반 답변")
st.write("마이크 사용 시 브라우저 권한을 허용해주세요.")

input_mode = st.radio("입력 방식을 선택하세요", ("텍스트", "음성"), horizontal=True)
input_text = ""

if input_mode == "텍스트":
    input_text = st.text_input("비서에게 할 말을 입력하세요:", placeholder="예: 오늘 일정 알려줘")
elif input_mode == "음성":
    st.write("아래 버튼을 눌러 마이크로 말하세요 👇")
    audio = mic_recorder(start_prompt="🎤 말하기 시작", stop_prompt="⏹️ 녹음 종료", key="recorder")
    if audio:
        st.audio(audio['bytes'], format="audio/webm")
        input_text = recognize_speech_web(audio['bytes'])

################################################################################
# 8. 사용자 요청 분기 처리 및 응답 생성
################################################################################
if input_text:
    if "이메일" in input_text or "일정" in input_text:
        response = agent.run(input_text)
    elif "회의" in input_text or "정보" in input_text:
        response = rag_chain.invoke(input_text)
    else:
        response = conversation.invoke(
            {"message": input_text},
            config={"configurable": {"session_id": "user-session"}}
        )

################################################################################
# 9. 응답 출력
################################################################################
    st.subheader("📝 비서 답변")
    st.write(response)

################################################################################
# 10. 음성 출력 (TTS)
################################################################################
    st.subheader("🔊 음성 출력")
    try:
        tts_engine = pyttsx3.init()
        tts_engine.say(response)
        tts_engine.runAndWait()
    except Exception as e:
        st.warning("TTS 실행 오류: pyttsx3 모듈이 정상 작동하지 않습니다.")

################################################################################
# 11. 대화 이력 저장 및 출력
################################################################################
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(("👤 사용자", input_text))
    st.session_state.history.append(("🤖 비서", response))

    st.subheader("🗂️ 대화 이력")
    for role, msg in st.session_state.history:
        st.markdown(f"**{role}:** {msg}")
