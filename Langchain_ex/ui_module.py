# ui_module.py (st.audio_input 사용 버전)
import streamlit as st
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
        except Exception:
            st.warning("TTS 엔진을 초기화할 수 없습니다. 음성 출력 기능이 비활성화됩니다.")
            return None

    def _get_whisper_recognizer(self):
        """지연 로딩으로 Whisper 인식기 초기화"""
        if self.whisper_recognizer is None:
            from whisper_module import WhisperRecognizer
            self.whisper_recognizer = WhisperRecognizer()
        return self.whisper_recognizer

    def speak_text(self, text: str):
        """텍스트를 음성으로 출력 (별도 스레드)"""
        if self.tts_engine and text:
            def _speak():
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except Exception:
                    pass # TTS 오류는 치명적이지 않으므로 조용히 실패
            
            thread = threading.Thread(target=_speak)
            thread.daemon = True
            thread.start()

    def _process_input(self, user_input, mode, use_tts):
        """사용자 입력을 처리하고 AI 응답을 생성 및 저장합니다."""
        with st.spinner("🤖 AI가 생각 중입니다..."):
            try:
                if mode == "일반 대화":
                    result = self.conversation_chain(user_input)
                    response = result.get("response", "응답을 생성할 수 없습니다.")
                elif mode == "문서 검색":
                    response = self.rag_chain.invoke(user_input)
                else: # 도구 사용
                    response = self.agent.run(user_input)
                
                if 'conversation_history' not in st.session_state:
                    st.session_state.conversation_history = []
                st.session_state.conversation_history.append({'user': user_input, 'assistant': response})

                if use_tts:
                    self.speak_text(response)
                
                st.rerun()

            except Exception as e:
                st.error(f"처리 중 오류가 발생했습니다: {e}")

    def run_ui(self):
        st.markdown("---")

        with st.sidebar:
            st.header("⚙️ 설정")
            mode = st.selectbox(
                "응답 모드",
                ["일반 대화", "문서 검색", "도구 사용"],
                help="원하는 응답 방식을 선택하세요"
            )
            use_tts = st.checkbox("음성 출력", value=False)
            if st.button("대화 기록 초기화"):
                if 'conversation_history' in st.session_state:
                    st.session_state.conversation_history = []
                st.rerun()

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("💬 대화")
            
            input_method = st.radio("입력 방식", ["텍스트", "음성"], horizontal=True, label_visibility="collapsed")
            
            if input_method == "텍스트":
                user_input = st.text_input("메시지를 입력하세요:", placeholder="안녕하세요! 무엇을 도와드릴까요?", key="text_input")
                can_send = bool(user_input and user_input.strip())
                if st.button("📤 전송", disabled=not can_send, key="send_button", use_container_width=True):
                    self._process_input(user_input, mode, use_tts)
            
            else:
                # st.audio_input 사용 - 더 안정적이고 사용자 친화적
                st.write("🎤 아래 버튼을 클릭하여 음성을 녹음하세요:")
                audio_data = st.audio_input("음성 메시지 녹음", key="audio_input")
                
                if audio_data is not None:
                    # 오디오 재생 위젯 표시 (사용자가 확인 가능)
                    st.audio(audio_data, format="audio/wav")
                    
                    # 음성 인식 버튼
                    if st.button("🔄 음성을 텍스트로 변환", key="transcribe_button", use_container_width=True):
                        with st.spinner("🔄 음성을 텍스트로 변환 중..."):
                            try:
                                recognizer = self._get_whisper_recognizer()
                                # st.audio_input은 바이트 데이터를 직접 제공
                                audio_bytes = audio_data.getvalue()
                                recognized_text = recognizer.recognize_audio_input(audio_bytes)
                                
                                if recognized_text:
                                    st.success(f"인식된 텍스트: {recognized_text}")
                                    # 인식된 텍스트를 바로 처리
                                    self._process_input(recognized_text, mode, use_tts)
                                else:
                                    st.warning("음성을 인식할 수 없습니다. 다시 시도해주세요.")
                            
                            except RuntimeError as e:
                                st.error("음성 처리 설정 오류가 발생했습니다.")
                                st.code(str(e), language='bash')
                            except Exception as e:
                                st.error(f"음성 인식 중 오류가 발생했습니다: {e}")

        with col2:
            st.subheader("📋 대화 기록")
            if 'conversation_history' not in st.session_state or not st.session_state.conversation_history:
                st.info("아직 대화 기록이 없습니다.")
            else:
                history = st.session_state.conversation_history
                for i, conv in enumerate(reversed(history)):
                    with st.expander(f"대화 {len(history)-i} ({datetime.now().strftime('%H:%M:%S')})", expanded=(i==0)):
                        st.markdown(f"**👤 사용자:**\n> {conv['user']}")
                        st.markdown(f"**🤖 비서:**\n> {conv['assistant']}")