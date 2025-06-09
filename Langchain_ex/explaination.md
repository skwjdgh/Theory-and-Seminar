
# 랭체인을 활용한 개인용 비서 서비스

이 프로그램은 랭체인을 활용해 로컬로 동작하는 개인용 비서의 프로토타입 모델입니다.
이 마크다운 문서는 이 프로그램의 설치 및 구동을 다루고 있습니다.

## 설치 방법

```
# 파이썬이 설치되어 있다는 가정하에 작성되어 있습니다.
# 필요한 주요 라이브러리르 한번에 받는 코드입니다.

pip install streamlit streamlit-mic-recorder openai-whisper langchain langchain-community langchain-ollama faiss-cpu edge-tts

```

| 라이브러리 이름              | 설명 및 주요 용도                                    |
|---------------------------|----------------------------------------------------|
| streamlit                 | 웹 기반 대화형 앱 개발 프레임워크                       |
| streamlit-mic-recorder    | 웹/모바일에서 마이크 입력 및 녹음 지원                   |
| openai-whisper            | 오픈소스 음성 인식(STT) 엔진 (로컬/웹 기반)             |
| langchain                 | 대형 언어모델(LLM) 기반 애플리케이션 개발 프레임워크      |
| langchain-community       | LangChain의 커뮤니티 확장 모듈                        |
| langchain-ollama          | LangChain과 Ollama 모델 연동 지원                     |
| faiss-cpu                 | 벡터 검색 및 유사도 계산용 라이브러리                   |
| edge-tts                  | Microsoft Edge 기반 텍스트-음성 변환(TTS) 엔진         |


이를 통해 설치하는 과정에서 환경변수 문제로 인한 error이 발생합니다. 이를 해결하기 위해 환경변수에 값을 추가해줘야 합니다.


#### 추가할 경로 - 사용자 이름의 변경이 필요함함

C:\Users\{사용자이름}\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts


- 위의 폴더 경로를 시스템 환경 변수 Path에 추가합니다.
- 명령 프롬프트(cmd) 실행
- pip list를 통해 확인 혹은 명령어 재실행을 통해 에러 발생 확인인


### whisper 사용을 위한 ffmpeg 설치

#### 추천 빌드 링크
https://www.gyan.dev/ffmpeg/builds/

- "Release" 버전 중 "ffmpeg-release-essentials.zip" 또는 "full" 버전 다운로드 (보통 essentials로 충분함)
- 다운받은 zip 파일을 원하는 폴더(예: C:\ffmpeg)에 풀어줍니다.
- C:\ffmpeg\bin 폴더 경로를 시스템 환경 변수 Path에 추가합니다.
- 명령 프롬프트(cmd) 실행
- ffmpeg -version 입력


# **환경변수 적용을 위해선 반드시 새 창을 열어야 적용됩니다! **
```
streamlit run 경로\assistant.py

# streamlit run C:\p_assistant\assistant.py
# 경로에 한글이 들어있을 경우, 에러가 발생합니다.
```

이를 실행하여 서버 주소를 확인하고 접속합니다.



이 코드는 랭체인의 기능을 설명하기 위해 AI의 도움을 받았습니다.

랭체인에는 이 코드의 기능을 제외하고 다양한 기능을 제공하고 있습니다. 
외부 데이터 연동, 다양한 툴킷(자연어로 SQL 질의 및 응답, 사용자 입력을 실시간 파이썬 코드로 실행, 실시간 웹 검색 기반 답변 생성), LangSmith 통합기능, 멀티모달 처리(이미지 입력(이미지->텍스트 변환), 혼합 입력 인식 등) 을 제공합니다.
