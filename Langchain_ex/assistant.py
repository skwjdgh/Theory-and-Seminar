################################################################################
# 1. Whisper 모델 초기화
################################################################################
import streamlit as st
from streamlit_mic_recorder import mic_recorder  # 웹 마이크 입력
import whisper  # 음성 인식
import io  # 오디오 바이트 처리
import platform
import os

################################################################################
# 2. LangChain 및 Ollama 모듈 로딩
################################################################################
from langchain_ollama import ChatOllama, OllamaEmbeddings  # Ollama 모델 연동, 임베딩
from langchain_core.prompts import ChatPromptTemplate  # 프롬프트 템플릿
from langchain_core.output_parsers import StrOutputParser  # 출력 파서
from langchain.agents import Tool, initialize_agent, AgentType  # 에이전트 도구
from langchain.memory import ConversationBufferMemory  # 대화 메모리 저장
from langchain.chains import ConversationChain  # 대화형 체인
from langchain_community.vectorstores import FAISS  # 벡터 검색
from langchain_core.documents import Document  # 문서
from langchain_core.runnables import RunnablePassthrough  # RAG용 흐름 구성

################################################################################
# 3. Whisper 음성 인식 모델 캐시 로딩
################################################################################
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

################################################################################
# 4. 음성 인식 함수 정의
################################################################################
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
# 5. LLM 및 기본 대화 체인 구성
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

chain = prompt | llm | StrOutputParser()

################################################################################
# 6. 대화 메모리 및 ConversationChain
################################################################################
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

################################################################################
# 7. 문서 기반 RAG 체인 구성
################################################################################
documents = [
    Document(page_content="회의는 매주 월요일 오후 3시에 진행됩니다."),
    Document(page_content="개인 비서는 사용자의 요청을 친절하게 처리합니다.")
]
embeddings = OllamaEmbeddings(model="llama3.1", base_url="http://localhost:11434")
db = FAISS.from_documents(documents, embeddings)
retriever = db.as_retriever()

rag_prompt = ChatPromptTemplate.from_template("문맥: {context}\n질문: {question}\n답변:")
rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser())

################################################################################
# 8. 툴 정의 (에이전트가 활용할 수 있는 기능)
################################################################################
tools = [
    Tool(
        name="DraftEmail",
        func=lambda ctx: chain.invoke({"message": f"다음 내용으로 이메일을 작성해주세요: {ctx}"}),
        description="주어진 내용으로 이메일을 작성합니다."
    ),
    Tool(
        name="ScheduleMeeting",
        func=lambda date, topic: chain.invoke({"message": f"{date}에 {topic} 회의 일정을 잡아주세요."}),
        description="특정 날짜와 주제로 회의 일정을 잡습니다."
    ),
    Tool(
        name="QnA",
        func=lambda question: chain.invoke({"message": question}),
        description="일반 질문에 답변합니다."
    )
]

################################################################################
# 9. 에이전트 초기화
################################################################################
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory)

################################################################################
# 10. Streamlit UI 구성
################################################################################
st.title("🤖 개인용 AI 비서 서비스")
st.subheader("텍스트/음성 입력 → Llama 3.1 기반 답변")
st.write("마이크 사용 시 브라우저 권한을 허용해주세요. (최초 1회 필요)")

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
# 11. 사용자 요청 분기 처리 및 응답 생성
################################################################################
if input_text:
    if "이메일" in input_text or "일정" in input_text:
        response = agent.run(input_text)
    elif "회의" in input_text or "정보" in input_text:
        response = rag_chain.invoke(input_text)
    else:
        response = conversation.predict(input=input_text)

    ################################################################################
    # 12. 응답 출력 및 TTS 처리
    ################################################################################
    st.subheader("📝 비서 답변")
    st.write(response)

    st.subheader("🔊 음성 출력")
    tts_command = None
    if platform.system() == "Darwin":
        tts_command = f'say "{response}"'
    elif platform.system() == "Windows":
        tts_command = f'edge-tts --text "{response}" --write-media output.mp3 && start output.mp3'
    elif platform.system() == "Linux":
        tts_command = f'tts --text "{response}" --out_path output.wav && aplay output.wav'

    if tts_command:
        os.system(tts_command)

    ################################################################################
    # 13. 대화 이력 저장 및 출력
    ################################################################################
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(("👤 사용자", input_text))
    st.session_state.history.append(("🤖 비서", response))

    st.subheader("🗂️ 대화 이력")
    for role, msg in st.session_state.history:
        st.markdown(f"**{role}:** {msg}")
