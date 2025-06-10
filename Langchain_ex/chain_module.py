# chain_module.py (수정 완료)
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

class ChainBuilder:
    def __init__(self, status_text_area=None):
        self.status_text = status_text_area
        self._update_status("📊 임베딩 모델 로딩 중...")
        self.embeddings = self.load_embeddings()
        
        self._update_status("📚 벡터 저장소 구축 중...")
        self.vectorstore = self.load_vectorstore(self.embeddings)
        self.retriever = self.vectorstore.as_retriever()
        
        self._update_status("🤖 LLM 연결 중...")
        self.llm = self._create_llm()

    def _update_status(self, text):
        if self.status_text:
            self.status_text.text(text)

    def load_embeddings(self):
        """임베딩 모델 로딩"""
        try:
            embeddings = OllamaEmbeddings(
                model=os.getenv("EMBEDDING_MODEL", "llama3"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
            embeddings.embed_query("연결 테스트")
            return embeddings
        except Exception as e:
            raise RuntimeError(f"임베딩 모델 로딩 실패: {e}")

    def load_vectorstore(self, _embeddings):
        """벡터 저장소 구축"""
        docs = [
            Document(page_content="업데이트된 회의 정보: 매주 화요일 오전 10시"),
            Document(page_content="새로운 정책: 모든 요청은 24시간 내 처리"),
            Document(page_content="회사 연락처: 본사 02-1234-5678, 지원팀 02-8765-4321"),
            Document(page_content="업무 시간: 평일 오전 9시 - 오후 6시"),
        ]
        return FAISS.from_documents(docs, _embeddings)

    def _create_llm(self):
        """LLM 생성 및 연결 테스트"""
        try:
            llm = ChatOllama(
                model=os.getenv("LLM_MODEL", "llama3"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0.7,
            )
            llm.invoke("연결 테스트")
            return llm
        except Exception as e:
            raise RuntimeError(f"LLM 연결 실패: {e}")

    def build_base_chain(self):
        prompt = ChatPromptTemplate.from_template(
            """당신은 전문 개인 비서입니다. 다음 규칙을 준수하세요:
            1. 사용자 요청을 정확히 이해하고 도움이 되는 답변 제공
            2. 친절하고 정중한 말투 사용 (존댓말)
            3. 답변은 간결하되 필요한 정보는 충분히 포함
            4. 모르는 것은 솔직히 모른다고 답변

            대화 이력: {history}
            현재 요청: {message}
            
            답변:"""
        )
        return prompt | self.llm | StrOutputParser()

    def build_rag_chain(self):
        rag_prompt = ChatPromptTemplate.from_template(
            """다음 문서를 참고하여 질문에 답변해주세요:

            참고 문서:
            {context}

            질문: {question}
            
            답변: 참고 문서의 정보를 바탕으로 정확하고 도움이 되는 답변을 제공해드리겠습니다."""
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        return (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )

    def build_conversation_chain(self):
        """대화 체인 생성 (기록 관리는 UI에서 담당)"""
        base_chain = self.build_base_chain()
        
        def conversation_handler(message: str):
            if 'conversation_history' not in st.session_state:
                st.session_state.conversation_history = []
            
            history = st.session_state.conversation_history
            formatted_history = "\n".join([
                f"사용자: {conv['user']}\n비서: {conv['assistant']}"
                for conv in history[-3:]  # 최근 3개의 대화만 컨텍스트로 활용
            ])
            
            try:
                response = base_chain.invoke({
                    "message": message,
                    "history": formatted_history
                })
                return {"response": response, "success": True}
            except Exception as e:
                error_msg = f"죄송합니다. 처리 중 오류가 발생했습니다: {e}"
                return {"response": error_msg, "success": False}
        
        return conversation_handler