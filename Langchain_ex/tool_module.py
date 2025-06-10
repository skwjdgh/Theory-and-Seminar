# tool_module.py (수정 완료)
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

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
        """이메일 초안을 작성합니다. 이메일에 포함될 내용을 입력으로 받습니다."""
        prompt = f"""다음 내용으로 정중하고 전문적인 이메일 초안을 작성해주세요:

        내용: {input_text}
        
        이메일 형식:
        - 제목
        - 인사말
        - 본문
        - 마무리 인사"""
        
        try:
            # 참고: 이 구조에서 base_chain은 대화의 전체 맥락을 알지 못합니다.
            # Agent가 tool에 넘겨주는 input_text에 맥락을 잘 요약해서 전달해야 합니다.
            return self.base_chain.invoke({
                "message": prompt,
                "history": "" # 이 부분이 맥락을 단절시키는 원인
            })
        except Exception as e:
            return f"이메일 작성 중 오류가 발생했습니다: {str(e)}"

    def schedule_meeting(self, input_text: str) -> str:
        """회의 일정을 정리하거나 요약합니다. 회의 관련 정보를 입력으로 받습니다."""
        prompt = f"""다음 회의 일정 요청을 처리하고 정리해주세요:

        요청 내용: {input_text}

        다음 정보를 포함하여 회의 일정을 정리해주세요:
        - 회의 목적
        - 제안 일시
        - 참석자
        - 준비사항"""
        
        try:
            return self.base_chain.invoke({
                "message": prompt,
                "history": ""
            })
        except Exception as e:
            return f"일정 처리 중 오류가 발생했습니다: {str(e)}"

    def build_agent(self):
        tools = [
            Tool(
                name="DraftEmail",
                func=self.draft_email,
                description="새로운 이메일 초안을 작성해야 할 때 사용합니다. '김과장에게 보낼 주간 보고서 이메일 써줘'와 같이 이메일의 핵심 내용을 입력으로 받습니다."
            ),
            Tool(
                name="ScheduleMeeting",
                func=self.schedule_meeting,
                description="회의 일정을 잡거나 정리해야 할 때 사용합니다. '내일 3시 팀 회의 잡아줘'와 같이 회의 관련 정보를 입력으로 받습니다."
            ),
            # [수정] 불필요한 GeneralQA 도구 제거
        ]

        return initialize_agent(
            tools,
            self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, # 대화에 더 적합한 Agent Type
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=3
        )