# main.py
from chain_module import ChainBuilder
from tool_module import AssistantTools
from ui_module import AssistantUI
import streamlit as st
import requests
import time
import os

def check_ollama_connection():
    """Ollama 서버 연결 확인"""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        return response.status_code == 200
    except:
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
    except:
        return False

# 수정: @st.cache_resource 제거하고 session_state 사용
def initialize_components():
    """컴포넌트 초기화 with detailed progress"""
    if 'components_initialized' in st.session_state:
        return st.session_state['agent'], st.session_state['rag_chain'], st.session_state['conversation_chain']
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. Ollama 연결 확인
        status_text.text("🔍 Ollama 서버 연결 확인 중...")
        progress_bar.progress(10)
        
        if not check_ollama_connection():
            st.error("""
            ❌ **Ollama 서버에 연결할 수 없습니다!**
            
            해결 방법:
            1. Ollama가 설치되어 있는지 확인: https://ollama.com
            2. Ollama 서버가 실행 중인지 확인: `ollama serve`
            3. 포트 확인: 기본값은 11434번 포트
            """)
            st.stop()
        
        # 2. 모델 확인
        model_name = os.getenv("LLM_MODEL", "llama3")
        status_text.text(f"🔍 모델 '{model_name}' 확인 중...")
        progress_bar.progress(20)
        
        if not check_ollama_model(model_name):
            st.error(f"""
            ❌ **모델 '{model_name}'이 설치되지 않았습니다!**
            
            해결 방법:
            1. 터미널에서 실행: `ollama pull {model_name}`
            2. 또는 다른 모델 사용: `ollama list`로 설치된 모델 확인
            """)
            st.stop()
        
        # 3. ChainBuilder 초기화
        status_text.text("🔧 ChainBuilder 초기화 중...")
        progress_bar.progress(40)
        chain_builder = ChainBuilder()
        
        # 4. Base chain 구축
        status_text.text("⛓️ Base chain 구축 중...")
        progress_bar.progress(60)
        base_chain = chain_builder.build_base_chain()
        
        # 5. RAG chain 구축
        status_text.text("📚 RAG chain 구축 중...")
        progress_bar.progress(70)
        rag_chain = chain_builder.build_rag_chain()
        
        # 6. Conversation chain 구축
        status_text.text("💬 Conversation chain 구축 중...")
        progress_bar.progress(80)
        conversation_chain = chain_builder.build_conversation_chain()
        
        # 7. Tools 초기화
        status_text.text("🛠️ Tools 초기화 중...")
        progress_bar.progress(90)
        tools = AssistantTools(base_chain, chain_builder.llm)
        
        # 완료
        progress_bar.progress(100)
        status_text.text("✅ 초기화 완료!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Session state에 저장
        st.session_state['components_initialized'] = True
        st.session_state['agent'] = tools.agent
        st.session_state['rag_chain'] = rag_chain
        st.session_state['conversation_chain'] = conversation_chain
        
        return tools.agent, rag_chain, conversation_chain
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"""
        ❌ **초기화 오류 발생:**
        
        ```
        {str(e)}
        ```
        
        **문제 해결 단계:**
        1. Ollama 서버가 실행 중인지 확인
        2. 모델이 올바르게 설치되었는지 확인
        3. 네트워크 연결 상태 확인
        4. 로그를 확인하여 구체적인 오류 파악
        """)
        st.stop()

def show_system_status():
    """시스템 상태 표시"""
    st.sidebar.header("🔧 시스템 상태")
    
    # Ollama 연결 상태
    if check_ollama_connection():
        st.sidebar.success("✅ Ollama 연결됨")
        
        # 설치된 모델 목록
        try:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            response = requests.get(f"{ollama_url}/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model.get('name', '').split(':')[0] for model in models]
                st.sidebar.info(f"📦 설치된 모델: {', '.join(set(model_names))}")
        except:
            pass
    else:
        st.sidebar.error("❌ Ollama 연결 실패")
    
    # 환경 변수 상태
    st.sidebar.info(f"""
    **설정:**
    - Base URL: {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}
    - LLM Model: {os.getenv('LLM_MODEL', 'llama3')}
    - Embedding Model: {os.getenv('EMBEDDING_MODEL', 'llama3')}
    """)

def main():
    st.set_page_config(
        page_title="AI 개인 비서",
        page_icon="🤖",
        layout="wide"
    )
    
    # 시스템 상태 먼저 확인
    show_system_status()
    
    st.title("🤖 AI 개인 비서")
    st.markdown("---")
    
    # 수동 재시작 버튼
    if st.button("🔄 시스템 재시작"):
        # session state 초기화
        for key in ['components_initialized', 'agent', 'rag_chain', 'conversation_chain']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    try:
        # 초기화 (진행 상황 표시)
        with st.container():
            st.subheader("🚀 시스템 초기화")
            agent, rag_chain, conversation_chain = initialize_components()
        
        # 초기화 완료 후 UI 실행
        st.success("🎉 시스템이 준비되었습니다!")
        ui = AssistantUI(agent, rag_chain, conversation_chain)
        ui.run_ui()
        
    except Exception as e:
        st.error(f"애플리케이션 실행 오류: {str(e)}")
        st.info("🔄 '시스템 재시작' 버튼을 눌러 다시 시도해보세요.")

if __name__ == "__main__":
    main()