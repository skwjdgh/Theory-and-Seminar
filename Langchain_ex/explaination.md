# 🧠 LangChain 기반 개인용 비서 서비스

이 프로젝트는 **LangChain**을 활용하여 로컬에서 동작하는 **개인 비서** 프로토타입입니다.

## 📁 파일 구성

```
├── chain_module.py
├── tool_module.py
├── ui_module.py
├── whisper_module.py
├── main.py
└── requirements.txt
```

---

## 🛠️ 설치 방법

Python이 설치되어 있다는 전제하에 아래 명령어를 실행하세요:

```bash
pip install -r {경로}\requirements.txt
```

> ❗ `requirements.txt`의 경로에 한글이 포함되어 있으면 에러가 발생할 수 있습니다.

pip install --upgrade streamlit
# 업데이트 필수
---

## 📦 주요 라이브러리 설명

| 라이브러리             | 설명 |
|----------------------|------|
| `langchain-core`     | LangChain의 핵심 컴포넌트 제공 |
| `langchain-community`| 다양한 오픈소스 통합 커넥터 모음 |
| `langchain-ollama`   | LangChain과 Ollama(로컬 LLM) 연동 |
| `langgraph`          | 대화 흐름 및 상태 그래프 구성용 프레임워크 |
| `langsmith`          | 평가, 로깅, 모니터링 등 앱 성능 분석 도구 |
| `faiss-cpu`          | 고속 벡터 유사도 검색 (CPU 전용) |
| `torch`              | PyTorch 딥러닝 프레임워크 |
| `torchvision`        | 컴퓨터 비전용 유틸리티 |
| `torchaudio`         | 오디오 처리 라이브러리 |
| `openai-whisper`     | 오픈소스 음성 인식(STT) 모델 |
| `streamlit`          | 웹 기반 인터페이스 구축 프레임워크 |
| `pyttsx3`            | 오프라인 텍스트 음성 변환 (TTS) |
| `python-dotenv`      | .env 파일을 통해 환경변수 로드 |

---

## ⚠️ Whisper 설치 문제 해결

- `openai-whisper`는 Python **3.11 이하**만 공식 지원합니다.
- **Python 3.9** 사용을 권장합니다.

### ✅ 환경변수 설정

다음 경로를 시스템 환경 변수 `Path`에 추가하세요:

```
C:\Users\{사용자이름}\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\Scripts
```

- 추가 후 **새 cmd 창**을 열고 아래 명령어로 정상 설치 여부를 확인하세요:

```bash
pip list
```

---

## 🎙️ ffmpeg 설치 (선택 사항)

> Whisper를 독립적으로 사용할 경우 필요합니다.  
> Streamlit 마이크 입력만 사용할 경우 생략해도 무방합니다.

### 설치 링크

[🔗 FFmpeg Download - Gyan.dev](https://www.gyan.dev/ffmpeg/builds/)

### 설치 방법

1. `ffmpeg-release-essentials.zip` 다운로드
2. 예: `C:\ffmpeg` 경로에 압축 해제
3. `C:\ffmpeg\bin`을 환경 변수 `Path`에 추가
4. cmd에서 아래 명령어로 설치 확인

```bash
ffmpeg -version
```

---

## 🚀 실행 방법

### ✅ 로컬 실행

```bash
streamlit run main.py
```

> 예시:
> ```
> streamlit run C:\p_assistant\main.py
> ```

💡 **경로에 한글이 포함되어 있으면 실행이 실패할 수 있습니다.**

---

### 🌐 핫스팟(다른 기기 접속 허용)

```bash
streamlit run main.py --server.address=0.0.0.0 --server.port=8501
```
> 예시:
> ```
> streamlit run C:\p_assistant\main.py --server.address=0.0.0.0 --server.port=8501
> ```

| 옵션 | 설명 |
|------|------|
| `--server.address=0.0.0.0` | 모든 네트워크 인터페이스에서 접속 허용 |
| `--server.port=8501`       | 접속 포트 설정 (필요 시 변경 가능) |

```bash
# IP 주소 확인
ipconfig
```
