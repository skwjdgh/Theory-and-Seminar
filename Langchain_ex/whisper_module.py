# whisper_module.py (st.audio_input 호환 버전 - 수정됨)
import io
import os
import tempfile
import subprocess
import whisper
import torch
import streamlit as st
from pathlib import Path

@st.cache_resource(show_spinner="🔊 음성 모델 초기화 중...")
def load_whisper_model():
    """Whisper 모델을 로드합니다. GPU가 있으면 CUDA를 사용합니다."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("base", device=device)

class WhisperRecognizer:
    def __init__(self):
        self.model = load_whisper_model()

    def check_ffmpeg(self) -> bool:
        """FFmpeg 설치 여부를 확인합니다."""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def transcode_to_wav(self, audio_bytes: bytes) -> io.BytesIO:
        """
        오디오를 Whisper가 처리할 수 있는 16kHz 모노 WAV 형식으로 변환합니다.
        """
        if not self.check_ffmpeg():
            error_message = """
            [오류] FFmpeg가 설치되어 있지 않거나 시스템 경로에 없습니다.
            음성 인식을 사용하려면 FFmpeg를 반드시 설치해야 합니다.

            - **macOS (Homebrew 사용 시):** `brew install ffmpeg`
            - **Ubuntu/Debian (APT 사용 시):** `sudo apt-get install ffmpeg`
            - **Windows (Chocolatey 사용 시):** `choco install ffmpeg`
            - 또는 공식 사이트(ffmpeg.org)에서 다운로드하여 시스템 PATH에 추가하세요.

            FFmpeg 설치 후, 현재 실행 중인 터미널과 애플리케이션을 완전히 종료하고 재시작해주세요.
            """
            raise RuntimeError(error_message)
        
        try:
            ffmpeg_cmd = [
                'ffmpeg', '-i', 'pipe:0',
                '-f', 'wav',
                '-ac', '1',
                '-ar', '16000',
                '-y', 'pipe:1'
            ]
            process = subprocess.run(
                ffmpeg_cmd,
                input=audio_bytes,
                capture_output=True,
                check=True
            )
            return io.BytesIO(process.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"오디오 변환에 실패했습니다. FFmpeg 오류: {e.stderr.decode()}")

    def recognize_audio_input(self, audio_bytes: bytes) -> str:
        """
        st.audio_input에서 받은 오디오 데이터를 텍스트로 변환합니다.
        """
        try:
            if not audio_bytes or len(audio_bytes) < 1000:
                st.warning("녹음된 오디오 데이터가 너무 짧거나 없습니다.")
                return ""
            
            # FFmpeg로 변환된 오디오를 임시 파일로 저장 후 처리
            wav_audio = self.transcode_to_wav(audio_bytes)
            
            # 임시 파일에 저장
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(wav_audio.getvalue())
                temp_path = temp_file.name
            
            try:
                # 임시 파일 경로로 Whisper 실행
                result = self.model.transcribe(temp_path, language="ko", fp16=False)
                text = result.get("text", "").strip()
                
                if len(text) < 2:
                    return ""
                    
                return text
                
            finally:
                # 임시 파일 삭제
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            raise e

    def recognize(self, audio_bytes: bytes) -> str:
        """기존 호환성을 위한 메서드 (streamlit_mic_recorder용)"""
        try:
            if not audio_bytes or len(audio_bytes) < 1000:
                st.warning("녹음된 오디오 데이터가 너무 짧거나 없습니다.")
                return ""
            
            wav_audio = self.transcode_to_wav(audio_bytes)
            
            # 임시 파일에 저장
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(wav_audio.getvalue())
                temp_path = temp_file.name
            
            try:
                # 임시 파일 경로로 Whisper 실행
                result = self.model.transcribe(temp_path, language="ko", fp16=False)
                text = result.get("text", "").strip()
                
                if len(text) < 2:
                    return ""
                    
                return text
                
            finally:
                # 임시 파일 삭제
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            raise e