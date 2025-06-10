# tool_module.py
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferMemory
from typing import Any

class AssistantTools:
    def __init__(self, base_chain, llm):
        self.base_chain = base_chain
        self.llm = llm
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.agent = self.build_agent()

    def draft_email(self, input_text: str) -> str:
        """이메일 초안을 작성합니다."""
        prompt = f"""다음 내용으로 정중하고 전문적인 이메일을 작성해주세요:

내용: {input_text}

이메일 형식:
- 제목
- 인사말
- 본문
- 마무리 인사"""
        
        try:
            return self.base_chain.invoke({
                "message": prompt,
                "history": ""
            })
        except Exception as e:
            return f"이메일 작성 중 오류가 발생했습니다: {str(e)}"

    def schedule_meeting(self, input_text: str) -> str:
        """회의 일정을 관리합니다."""
        prompt = f"""다음 회의 일정 요청을 처리해주세요:

요청 내용: {input_text}

다음 정보를 포함하여 회의 일정을 정리해주세요:
- 회의 목적
- 제안 일시
- 참석자
- 준비사항
- 확인 필요사항"""
        
        try:
            return self.base_chain.invoke({
                "message": prompt,
                "history": ""
            })
        except Exception as e:
            return f"일정 처리 중 오류가 발생했습니다: {str(e)}"

    def general_qa(self, input_text: str) -> str:
        """일반적인 질문에 답변합니다."""
        try:
            return self.base_chain.invoke({
                "message": input_text,
                "history": ""
            })
        except Exception as e:
            return f"답변 처리 중 오류가 발생했습니다: {str(e)}"

    def build_agent(self):
        tools = [
            Tool(
                name="DraftEmail",
                func=self.draft_email,
                description="이메일 초안을 작성할 때 사용합니다. 이메일 내용을 입력으로 받습니다."
            ),
            Tool(
                name="ScheduleMeeting",
                func=self.schedule_meeting,
                description="회의 일정을 잡거나 관리할 때 사용합니다. 회의 관련 정보를 입력으로 받습니다."
            ),
            Tool(
                name="GeneralQA",
                func=self.general_qa,
                description="일반적인 질문이나 요청에 답변할 때 사용합니다."
            )
        ]

        return initialize_agent(
            tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=3
        )