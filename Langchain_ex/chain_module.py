# chain_module.py
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, List, TypedDict
import streamlit as st
from dotenv import load_dotenv
import time

load_dotenv()

# 수정: TypedDict로 State 정의
class ConversationState(TypedDict):
    message: str
    response: str
    success: bool

class ChainBuilder:
    def __init__(self):
        st.write("📊 임베딩 모델 로딩 중...")
        self.embeddings = self.load_embeddings()
        
        st.write("📚 벡터 저장소 구축 중...")
        self.vectorstore = self.load_vectorstore(self.embeddings)
        self.retriever = self.vectorstore.as_retriever()
        
        st.write("🤖 LLM 연결 중...")
        self.llm = self._create_llm()

    @staticmethod
    def load_embeddings():
        """임베딩 모델 로딩"""
        try:
            st.write("🔄 임베딩 모델 초기화 중...")
            embeddings = OllamaEmbeddings(
                model=os.getenv("EMBEDDING_MODEL", "llama3"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
            
            # 간단한 연결 테스트
            st.write("🔄 임베딩 테스트 중...")
            test_embed = embeddings.embed_query("안녕하세요")
            st.write(f"✅ 임베딩 모델 로딩 완료 (차원수: {len(test_embed)})")
            return embeddings
            
        except Exception as e:
            error_msg = str(e)
            if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                st.error("❌ Ollama 서버 연결 실패. 서버가 실행 중인지 확인하세요.")
            elif "model" in error_msg.lower():
                st.error(f"❌ 모델 '{os.getenv('EMBEDDING_MODEL', 'llama3')}'를 찾을 수 없습니다. 모델을 설치하세요.")
            else:
                st.error(f"❌ 임베딩 모델 로딩 실패: {error_msg}")
            raise e

    @staticmethod
    def load_vectorstore(_embeddings):
        """벡터 저장소 구축"""
        try:
            docs = [
                Document(page_content="업데이트된 회의 정보: 매주 화요일 오전 10시"),
                Document(page_content="새로운 정책: 모든 요청은 24시간 내 처리"),
                Document(page_content="회사 연락처: 본사 02-1234-5678, 지원팀 02-8765-4321"),
                Document(page_content="업무 시간: 평일 오전 9시 - 오후 6시"),
            ]
            vectorstore = FAISS.from_documents(docs, _embeddings)
            st.write(f"✅ 벡터 저장소 구축 완료 ({len(docs)}개 문서)")
            return vectorstore
        except Exception as e:
            st.error(f"❌ 벡터 저장소 구축 실패: {str(e)}")
            raise e

    def _create_llm(self):
        """LLM 생성 및 연결 테스트"""
        try:
            st.write("🔄 LLM 초기화 중...")
            llm = ChatOllama(
                model=os.getenv("LLM_MODEL", "llama3"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0.7,
                num_predict=50  # 빠른 테스트를 위해 응답 길이 제한
            )
            
            # 간단한 연결 테스트
            st.write("🔄 LLM 연결 테스트 중...")
            test_response = llm.invoke("Hi")
            st.write(f"✅ LLM 연결 성공 (응답: {test_response.content[:30]}...)")
            return llm
            
        except Exception as e:
            error_msg = str(e)
            if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                st.error("❌ Ollama 서버 연결 실패. 서버가 실행 중인지 확인하세요.")
            elif "model" in error_msg.lower():
                st.error(f"❌ 모델 '{os.getenv('LLM_MODEL', 'llama3')}'를 찾을 수 없습니다. 모델을 설치하세요.")
            else:
                st.error(f"❌ LLM 연결 실패: {error_msg}")
            raise e

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
        # 수정: 대화 체인을 단순화
        base_chain = self.build_base_chain()
        
        def conversation_handler(message: str) -> Dict[str, Any]:
            """대화 처리 함수"""
            # 대화 기록 관리
            if 'conversation_history' not in st.session_state:
                st.session_state.conversation_history = []
            
            history = st.session_state.conversation_history
            
            # 최근 3개 대화만 컨텍스트로 사용
            formatted_history = "\n".join([
                f"사용자: {conv['user']}\n비서: {conv['assistant']}"
                for conv in history[-3:]
            ])
            
            try:
                response = base_chain.invoke({
                    "message": message,
                    "history": formatted_history
                })
                
                # 대화 기록에 추가
                history.append({"user": message, "assistant": response})
                
                # 최대 10개 대화만 유지
                if len(history) > 10:
                    history.pop(0)
                
                return {"response": response, "success": True}
                
            except Exception as e:
                error_msg = f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"
                return {"response": error_msg, "success": False}
        
        return conversation_handler