# whisper_module.py
import io
import subprocess
import whisper
import torch
import streamlit as st
from pathlib import Path

@st.cache_resource(show_spinner="🔊 음성 모델 초기화 중...")
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("base", device=device)

class WhisperRecognizer:
    def __init__(self):
        self.model = load_whisper_model()

    def check_ffmpeg(self) -> bool:
        """FFmpeg 설치 여부 확인"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def transcode_to_wav(self, audio_bytes: bytes) -> io.BytesIO:
        """오디오를 WAV 형식으로 변환"""
        if not self.check_ffmpeg():
            raise RuntimeError("FFmpeg가 설치되지 않았습니다. 설치 후 다시 시도해주세요.")
        
        try:
            ffmpeg_cmd = [
                'ffmpeg', '-i', 'pipe:0',
                '-f', 'wav', '-ac', '1', '-ar', '16000',
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
            raise RuntimeError(f"오디오 변환 실패: {e.stderr.decode()}")

    def recognize(self, audio_bytes: bytes) -> str:
        """음성을 텍스트로 변환"""
        try:
            wav_bio = self.transcode_to_wav(audio_bytes)
            result = self.model.transcribe(
                wav_bio, 
                language="ko",
                temperature=0.0,
                fp16=False,
                word_timestamps=False
            )
            return result.get("text", "").strip()
        except Exception as e:
            st.error(f"음성 인식 오류: {str(e)}")
            return ""
