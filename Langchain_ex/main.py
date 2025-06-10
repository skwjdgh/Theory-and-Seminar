# main.py (수정 완료)
import os
import time
import requests
import streamlit as st
from chain_module import ChainBuilder
from tool_module import AssistantTools
from ui_module import AssistantUI

def check_ollama_connection():
    """Ollama 서버 연결 확인"""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def check_ollama_model(model_name):
    """특정 모델이 설치되어 있는지 확인"""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return any(model_name in model.get('name', '') for model in models)
        return False
    except requests.exceptions.RequestException:
        return False

def initialize_components(status_text):
    """컴포넌트 초기화 (Session State 사용)"""
    if 'components_initialized' in st.session_state:
        return st.session_state.agent, st.session_state.rag_chain, st.session_state.conversation_chain

    progress_bar = st.progress(0, "🔍 Ollama 서버 연결 확인 중...")

    if not check_ollama_connection():
        st.error("❌ Ollama 서버에 연결할 수 없습니다! `ollama serve`를 실행했는지 확인해주세요.")
        st.stop()
    progress_bar.progress(20, "✅ Ollama 서버 연결됨. 모델 확인 중...")

    model_name = os.getenv("LLM_MODEL", "llama3")
    if not check_ollama_model(model_name):
        st.error(f"❌ 모델 '{model_name}'이 설치되지 않았습니다! `ollama pull {model_name}`으로 설치해주세요.")
        st.stop()
    progress_bar.progress(40, f"✅ 모델 '{model_name}' 확인됨. 체인 빌더 초기화 중...")

    try:
        chain_builder = ChainBuilder(status_text)
        progress_bar.progress(60, "⛓️ 체인 및 도구 구성 중...")

        base_chain = chain_builder.build_base_chain()
        rag_chain = chain_builder.build_rag_chain()
        conversation_chain = chain_builder.build_conversation_chain()
        tools = AssistantTools(base_chain, chain_builder.llm)

        progress_bar.progress(100, "✅ 초기화 완료!")
        time.sleep(1)

        st.session_state['components_initialized'] = True
        st.session_state['agent'] = tools.agent
        st.session_state['rag_chain'] = rag_chain
        st.session_state['conversation_chain'] = conversation_chain

        # 진행 표시 위젯들 제거
        progress_bar.empty()
        status_text.empty()

        return tools.agent, rag_chain, conversation_chain

    except Exception as e:
        st.error(f"초기화 중 심각한 오류 발생: {e}")
        st.info("오류를 해결한 후 '시스템 재시작' 버튼을 눌러주세요.")
        st.stop()

def main():
    st.set_page_config(
        page_title="AI 개인 비서",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 AI 개인 비서")

    with st.sidebar:
        st.header("🔧 시스템 제어")
        if st.button("🔄 시스템 재시작"):
            # session state 초기화
            st.session_state.clear()
            st.rerun()

    # 초기화가 완료되지 않았을 경우에만 초기화 블록 표시
    if 'components_initialized' not in st.session_state:
        st.markdown("---")
        st.subheader("🚀 시스템을 초기화합니다...")
        status_text_area = st.empty()
        initialize_components(status_text_area)
        st.rerun() # 초기화 후 UI를 다시 그리도록 강제 실행
    
    # 초기화 완료 후 메인 UI 로드
    else:
        agent = st.session_state.agent
        rag_chain = st.session_state.rag_chain
        conversation_chain = st.session_state.conversation_chain
        
        ui = AssistantUI(agent, rag_chain, conversation_chain)
        ui.run_ui()

if __name__ == "__main__":
    main()