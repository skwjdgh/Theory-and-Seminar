# whisper_module.py
import io
import subprocess
import whisper
import streamlit as st

class WhisperRecognizer:
    def __init__(self):
        self.model = self.load_model()

    @st.cache_resource
    def load_model(self):
        return whisper.load_model("base")

    def transcode_to_wav(self, audio_bytes: bytes) -> io.BytesIO:
        temp_in = '/tmp/input_audio'
        temp_out = '/tmp/output.wav'
        with open(temp_in, 'wb') as f:
            f.write(audio_bytes)
        subprocess.run(['ffmpeg', '-y', '-i', temp_in, temp_out], check=False)
        wav_bio = io.BytesIO(open(temp_out, 'rb').read())
        wav_bio.name = 'audio.wav'
        return wav_bio

    def recognize(self, audio_bytes):
        try:
            wav = self.transcode_to_wav(audio_bytes)
            result = self.model.transcribe(wav, language="ko")
            return result.get("text", None)
        except Exception as e:
            st.error(f"음성 인식 오류: {e}")
            return None


# chain_module.py
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import streamlit as st

class ChainBuilder:
    def __init__(self):
        self.embeddings = self.load_embeddings()
        self.vectorstore = self.load_vectorstore(self.embeddings)
        self.retriever = self.vectorstore.as_retriever()
        self.llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434", temperature=0.7, timeout_seconds=10)
        self.memory = self.get_memory()

    @st.cache_resource
    def load_embeddings(self):
        return OllamaEmbeddings(model="llama3.1", base_url="http://localhost:11434")

    @st.cache_resource
    def load_vectorstore(self, embeddings):
        docs = [
            Document(page_content="회의는 매주 월요일 오후 3시에 진행됩니다."),
            Document(page_content="개인 비서는 사용자의 요청을 친절하게 처리합니다.")
        ]
        return FAISS.from_documents(docs, embeddings)

    def get_memory(self):
        if 'memory' not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(return_messages=True)
        return st.session_state.memory

    def build_base_chain(self):
        prompt = ChatPromptTemplate.from_template(
            """당신은 전문 개인 비서입니다. 다음 규칙을 준수하세요:
            1. 사용자 요청을 정확히 이해해 답변
            2. 친절한 말투 사용 (존댓말)
            3. 답변은 최대 3문장 이내로 간결하게

            요청: {message}
            답변:"""
        )
        return prompt | self.llm | StrOutputParser()

    def build_rag_chain(self):
        rag_prompt = ChatPromptTemplate.from_template("문맥: {context}\n질문: {question}\n답변:")
        return ( {"context": self.retriever, "question": RunnablePassthrough()} | rag_prompt | self.llm | StrOutputParser())

    def build_conversation_chain(self, base_chain):
        return RunnableWithMessageHistory(
            base_chain,
            lambda session_id: self.memory,
            input_messages_key="message",
            history_messages_key="history"
        )


# tool_module.py
from langchain.agents import Tool, initialize_agent, AgentType

class AssistantTools:
    def __init__(self, base_chain, llm, memory):
        self.agent = self.build_agent(base_chain, llm, memory)

    def build_agent(self, base_chain, llm, memory):
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
        return initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory)


# ui_module.py
import streamlit as st
import threading
from streamlit_mic_recorder import mic_recorder
from whisper_module import WhisperRecognizer

class AssistantUI:
    def __init__(self, agent, rag_chain, conversation_chain):
        self.agent = agent
        self.rag_chain = rag_chain
        self.conversation_chain = conversation_chain
        self.recognizer = WhisperRecognizer()

    def tts_play(self, text: str):
        import pyttsx3
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

    def run_ui(self):
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
                input_text = self.recognizer.recognize(audio['bytes'])

        if input_text:
            if "이메일" in input_text or "일정" in input_text:
                response = self.agent.run(input_text)
            elif "회의" in input_text or "정보" in input_text:
                response = self.rag_chain.invoke({"question": input_text})
            else:
                response = self.conversation_chain.invoke(
                    {"message": input_text},
                    config={"configurable": {"session_id": "user-session"}}
                )

            st.subheader("📝 비서 답변")
            st.write(response)

            st.subheader("🔊 음성 출력")
            threading.Thread(target=self.tts_play, args=(response,), daemon=True).start()

            if 'history' not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append(("👤 사용자", input_text))
            st.session_state.history.append(("🤖 비서", response))
            if len(st.session_state.history) > 50:
                st.session_state.history = st.session_state.history[-50:]

            st.subheader("🗂️ 대화 이력")
            for role, msg in st.session_state.history:
                st.text(f"{role}: {msg}")


# main.py
from chain_module import ChainBuilder
from tool_module import AssistantTools
from ui_module import AssistantUI

if __name__ == "__main__":
    chain_builder = ChainBuilder()
    base_chain = chain_builder.build_base_chain()
    rag_chain = chain_builder.build_rag_chain()
    conversation_chain = chain_builder.build_conversation_chain(base_chain)

    tools = AssistantTools(base_chain, chain_builder.llm, chain_builder.memory)
    ui = AssistantUI(tools.agent, rag_chain, conversation_chain)
    ui.run_ui()