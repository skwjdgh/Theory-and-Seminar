import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import io
import os
import platform
import threading
import subprocess
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

# 1. 리소스 캐시와 세션 스테이트 초기화 분리
@st.cache_resource
def load_whisper_model():
    model = whisper.load_model("base")
    return model

@st.cache_resource
def load_embeddings():
    return OllamaEmbeddings(model="llama3.1", base_url="http://localhost:11434")

@st.cache_resource
def load_faiss_index(embeddings):
    docs = [Document(page_content="회의는 매주 월요일 오후 3시에 진행됩니다."),
            Document(page_content="개인 비서는 사용자의 요청을 친절하게 처리합니다.")]
    return FAISS.from_documents(docs, embeddings)

# 버전 고정은 requirements.txt에서 관리하세요

whisper_model = load_whisper_model()
embeddings = load_embeddings()
faiss_db = load_faiss_index(embeddings)
retriever = faiss_db.as_retriever()

# 2. LLM 및 프롬프트 체인 구성 (타임아웃 및 재시도 로직 포함)
llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434", temperature=0.7, timeout_seconds=10)

prompt = ChatPromptTemplate.from_template(
    """당신은 전문 개인 비서입니다. 다음 규칙을 준수하세요:
    1. 사용자 요청을 정확히 이해해 답변
    2. 친절한 말투 사용 (존댓말)
    3. 답변은 최대 3문장 이내로 간결하게

    요청: {message}
    답변:"""
)

base_chain = prompt | llm | StrOutputParser()

# 3. 세션별 메모리
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
conversation = RunnableWithMessageHistory(
    base_chain,
    lambda session_id: st.session_state.memory,
    input_messages_key="message",
    history_messages_key="history"
)

# 4. RAG 체인 구성 (invoke 입력 수정)
rag_prompt = ChatPromptTemplate.from_template("문맥: {context}\n질문: {question}\n답변:")
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt | llm | StrOutputParser()
)

# 5. Tool 함수 정의 (람다 대신 함수)
def draft_email_tool(input_text: str) -> str:
    return base_chain.invoke({"message": f"다음 내용으로 이메일을 작성해주세요: {input_text}"})

def schedule_meeting_tool(input_text: str) -> str:
    return base_chain.invoke({"message": f"{input_text} 일정 잡아줘"})

def qna_tool(input_text: str) -> str:
    return base_chain.invoke({"message": input_text})

tools = [
    Tool(name="DraftEmail", func=draft_email_tool, description="내용을 입력하면 이메일을 작성해드립니다."),
    Tool(name="ScheduleMeeting", func=schedule_meeting_tool, description="회의 일정을 조율합니다."),
    Tool(name="QnA", func=qna_tool, description="일반 질문에 답변합니다.")
]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=st.session_state.memory)

# 6. Whisper 음성 인식 (ffmpeg 변환 지원)
def transcode_to_wav(audio_bytes: bytes) -> io.BytesIO:
    temp_in = '/tmp/input_audio'
    temp_out = '/tmp/output.wav'
    with open(temp_in, 'wb') as f:
        f.write(audio_bytes)
    subprocess.run(['ffmpeg', '-y', '-i', temp_in, temp_out], check=False)
    wav_bio = io.BytesIO(open(temp_out, 'rb').read())
    wav_bio.name = 'audio.wav'
    return wav_bio


def recognize_speech_web(audio_bytes):
    try:
        audio_bio = transcode_to_wav(audio_bytes)
        result = whisper_model.transcribe(audio_bio, language="ko")
        return result.get("text", None)
    except Exception as e:
        st.error(f"음성 인식 오류: {e}")
        return None

# 7. Streamlit UI
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

# 8. 요청 분기 처리 및 예외 처리
if input_text:
    if "이메일" in input_text or "일정" in input_text:
        response = agent.run(input_text)
    elif "회의" in input_text or "정보" in input_text:
        response = rag_chain.invoke({"question": input_text})
    else:
        response = conversation.invoke(
            {"message": input_text},
            config={"configurable": {"session_id": "user-session"}}
        )

    # 9. 응답 출력
    st.subheader("📝 비서 답변")
    st.write(response)

    # 10. 비동기 TTS 처리
    def tts_play(text: str):
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

    st.subheader("🔊 음성 출력")
    threading.Thread(target=tts_play, args=(response,), daemon=True).start()

    # 11. 대화 이력 관리 (최대 50개)
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(("👤 사용자", input_text))
    st.session_state.history.append(("🤖 비서", response))
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[-50:]

    st.subheader("🗂️ 대화 이력")
    for role, msg in st.session_state.history:
        st.text(f"{role}: {msg}")
