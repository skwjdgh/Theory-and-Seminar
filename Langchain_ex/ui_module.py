# ui_module.py
import streamlit as st
from streamlit_mic_recorder import mic_recorder
import pyttsx3
from datetime import datetime
import threading

class AssistantUI:
    def __init__(self, agent, rag_chain, conversation_chain):
        self.agent = agent
        self.rag_chain = rag_chain
        self.conversation_chain = conversation_chain
        self.tts_engine = self._init_tts()
        self.whisper_recognizer = None

    def _init_tts(self):
        """TTS 엔진 초기화"""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            return engine
        except:
            return None

    def _get_whisper_recognizer(self):
        """지연 로딩으로 Whisper 인식기 초기화"""
        if self.whisper_recognizer is None:
            from whisper_module import WhisperRecognizer
            self.whisper_recognizer = WhisperRecognizer()
        return self.whisper_recognizer

    def speak_text(self, text: str):
        """텍스트를 음성으로 출력 (별도 스레드)"""
        if self.tts_engine:
            def _speak():
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except:
                    pass
            
            thread = threading.Thread(target=_speak)
            thread.daemon = True
            thread.start()

    def run_ui(self):
        st.title("🤖 AI 개인 비서")
        st.markdown("---")

        # 사이드바 설정
        with st.sidebar:
            st.header("⚙️ 설정")
            
            # 모드 선택
            mode = st.selectbox(
                "응답 모드",
                ["일반 대화", "문서 검색", "도구 사용"],
                help="원하는 응답 방식을 선택하세요"
            )
            
            # TTS 설정
            use_tts = st.checkbox("음성 출력", value=False)
            
            # 대화 기록 초기화
            if st.button("대화 기록 초기화"):
                if 'conversation_history' in st.session_state:
                    st.session_state.conversation_history = []
                st.success("대화 기록이 초기화되었습니다.")

        # 메인 영역을 두 개 컬럼으로 분할
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("💬 대화")
            
            # 입력 방식 선택
            input_method = st.radio(
                "입력 방식",
                ["텍스트", "음성"],
                horizontal=True
            )

            user_input = ""
            
            if input_method == "텍스트":
                user_input = st.text_input(
                    "메시지를 입력하세요:",
                    placeholder="안녕하세요! 무엇을 도와드릴까요?",
                    key="text_input"
                )
            else:
                st.write("🎤 음성 녹음:")
                audio_bytes = mic_recorder(
                    start_prompt="녹음 시작",
                    stop_prompt="녹음 중지",
                    key="voice_recorder"
                )
                
                if audio_bytes:
                    with st.spinner("음성을 인식하는 중..."):
                        try:
                            recognizer = self._get_whisper_recognizer()
                            user_input = recognizer.recognize(audio_bytes['bytes'])
                            if user_input:
                                st.success(f"인식된 텍스트: {user_input}")
                            else:
                                st.warning("음성을 인식하지 못했습니다.")
                        except Exception as e:
                            st.error(f"음성 인식 오류: {str(e)}")

            # 처리 버튼
            if st.button("전송", disabled=not user_input, key="send_button"):
                if user_input:
                    with st.spinner("처리 중..."):
                        try:
                            # 모드에 따른 처리
                            if mode == "일반 대화":
                                result = self.conversation_chain(user_input)
                                response = result.get("response", "응답을 생성할 수 없습니다.")
                            elif mode == "문서 검색":
                                response = self.rag_chain.invoke(user_input)
                            else:  # 도구 사용
                                response = self.agent.run(user_input)
                            
                            # 응답 표시
                            st.success("✅ 응답:")
                            st.write(response)
                            
                            # TTS 출력
                            if use_tts and response:
                                self.speak_text(response)
                                
                        except Exception as e:
                            st.error(f"처리 중 오류가 발생했습니다: {str(e)}")

        with col2:
            st.subheader("📋 대화 기록")
            
            # 대화 기록 표시
            if 'conversation_history' in st.session_state:
                history = st.session_state.conversation_history
                if history:
                    for i, conv in enumerate(reversed(history[-5:])):  # 최근 5개만 표시
                        with st.expander(f"대화 {len(history)-i}", expanded=(i==0)):
                            st.write(f"**사용자:** {conv['user']}")
                            st.write(f"**비서:** {conv['assistant']}")
                            st.caption(datetime.now().strftime("%Y-%m-%d %H:%M"))
                else:
                    st.info("아직 대화 기록이 없습니다.")
            
            # 시스템 정보
            st.subheader("ℹ️ 시스템 정보")
            st.info(f"""
            **현재 모드:** {mode}
            **음성 출력:** {'켜짐' if use_tts else '꺼짐'}
            **Ollama 연결:** {'✅' if True else '❌'}
            """)
