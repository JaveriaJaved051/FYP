from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Union
import os
import uuid
import time
from pathlib import Path
from gtts import gTTS
from pydub import AudioSegment
import numpy as np
import librosa
import soundfile as sf
import logging
import shutil
import tempfile
import re
import random
from pydub.effects import normalize
from pydub.silence import split_on_silence

# Disable GPU for all libraries
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_CPU_ONLY"] = "1"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="UltraRealistic Voice API (CPU Optimized)",
    description="Next-gen multilingual TTS optimized for CPU-only environments",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

# Constants
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)
SUPPORTED_LANGUAGES = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ar', 'zh', 'ja', 'ko', 'hi']
EMOTIONS = ["Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust", "Neutral"]
SUPPORTED_AUDIO_TYPES = [
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3",
    "audio/aac", "audio/flac", "audio/ogg", "audio/x-flac",
    "audio/x-m4a", "audio/webm", "audio/mp4"
]
MAX_TEXT_LENGTH = 500  # Characters
MAX_CLONE_DURATION = 30  # Seconds

# Mount static files
app.mount("/temp_audio", StaticFiles(directory="temp_audio"), name="temp_audio")

# Models
class VoiceSettings(BaseModel):
    language: str = 'en'
    emotion: str = "Neutral"
    speed: float = 1.0
    vocal_enhance: bool = True
    breath_effects: float = 0.3
    intonation: float = 0.7
    articulation: float = 0.8

class TextInput(BaseModel):
    text: Union[str, List[str]]
    settings: Optional[VoiceSettings] = None

class CloneResponse(BaseModel):
    file_url: str
    message: str

class SupportedFormatsResponse(BaseModel):
    supported_audio_types: List[str]

# Helper functions
def cleanup_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted temp file: {path}")
    except Exception as e:
        logger.warning(f"Failed to delete {path}: {str(e)}")

def cleanup_old_files(directory: Path, max_age_minutes: int = 60):
    try:
        now = time.time()
        for f in directory.iterdir():
            if f.is_file():
                file_age = now - f.stat().st_mtime
                if file_age > max_age_minutes * 60:
                    cleanup_file(str(f))
    except Exception as e:
        logger.warning(f"Failed to clean old files: {str(e)}")

def validate_audio_length(file_path: str, max_seconds: int = MAX_CLONE_DURATION):
    try:
        duration = librosa.get_duration(filename=file_path)
        if duration > max_seconds:
            raise ValueError(f"Audio too long (>{max_seconds}s). Please provide a shorter sample.")
    except Exception as e:
        raise ValueError(f"Invalid audio file: {str(e)}")

# Background cleanup
@app.on_event("startup")
async def startup_event():
    cleanup_old_files(TEMP_DIR)

# Core Voice Engine (CPU Optimized)
class UltraRealisticVoice:
    def __init__(self, language: str = 'en', clone_voice: bool = False, voice_model: str = None):
        self.text_segments = []
        self.language = self._validate_language(language)
        self.clone_voice = clone_voice
        self.voice_model = voice_model
        self.advanced_settings = {
            'vocal_enhance': True,
            'breath_effects': 0.3,
            'intonation': 0.7,
            'articulation': 0.8
        }
        
        if clone_voice and voice_model:
            try:
                from TTS.api import TTS
                self.tts_engine = TTS(
                    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                    progress_bar=False,
                    gpu=False
                )
                logger.info("XTTS v2 model loaded in CPU mode")
                self._preprocess_voice_model()
            except ImportError:
                logger.warning("Coqui TTS not available - voice cloning disabled")
                self.clone_voice = False
            except Exception as e:
                logger.error(f"Failed to initialize TTS: {str(e)}")
                self.clone_voice = False

    def _validate_language(self, lang_code: str) -> str:
        lang_code = lang_code.lower().split('-')[0]
        if lang_code not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language. Available: {SUPPORTED_LANGUAGES}")
        return lang_code

    def _preprocess_voice_model(self):
        try:
            logger.info("Optimizing voice model for CPU...")
            y, sr = librosa.load(self.voice_model, sr=16000)  # Lower sample rate for CPU
            
            # Lightweight processing
            y = librosa.util.normalize(y)
            y = librosa.effects.preemphasis(y, coef=0.97)
            
            processed_path = f"processed_{os.path.basename(self.voice_model)}"
            sf.write(processed_path, y, sr)
            self.voice_model = processed_path
        except Exception as e:
            logger.error(f"Voice model processing failed: {str(e)}")
            raise RuntimeError(f"Could not process voice model: {str(e)}")

    def _lightweight_compress(self, audio: AudioSegment) -> AudioSegment:
        """Simplified compression for CPU"""
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        compressed = np.tanh(samples * 2) * 32767
        return audio._spawn(compressed.astype(np.int16))

    def add_text(self, text: Union[str, List[str]]) -> 'UltraRealisticVoice':
        if isinstance(text, str):
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(f"Text too long (max {MAX_TEXT_LENGTH} chars)")
            text = re.sub(r'\s+', ' ', text).strip()
            self.text_segments.append(text)
        else:
            texts = [t.strip() for t in text if t.strip()]
            if sum(len(t) for t in texts) > MAX_TEXT_LENGTH:
                raise ValueError(f"Combined text too long (max {MAX_TEXT_LENGTH} chars)")
            self.text_segments.extend(texts)
        return self

    def generate(self, output_file: str) -> AudioSegment:
        if not self.text_segments:
            raise ValueError("No text to convert")
            
        temp_files = []
        try:
            text = " ".join(self.text_segments)
            
            # Generate base voice
            temp_tts = f"temp_base_{uuid.uuid4()}.mp3"
            gTTS(text=text, lang=self.language, slow=False).save(temp_tts)
            temp_files.append(temp_tts)
            
            audio = AudioSegment.from_mp3(temp_tts)
            audio = self._lightweight_mastering(audio)
            
            # Voice cloning if enabled
            if self.clone_voice and self.voice_model:
                try:
                    temp_clone = f"temp_cloned_{uuid.uuid4()}.wav"
                    self.tts_engine.tts_to_file(
                        text=text,
                        speaker_wav=self.voice_model,
                        language=self.language,
                        file_path=temp_clone,
                        emotion="Neutral",  # Simpler emotion for CPU
                        speed=1.0
                    )
                    temp_files.append(temp_clone)
                    
                    cloned_audio = AudioSegment.from_wav(temp_clone)
                    if len(cloned_audio) > len(audio):
                        cloned_audio = cloned_audio[:len(audio)]
                    audio = audio.overlay(cloned_audio - 6)
                except Exception as e:
                    logger.warning(f"Voice cloning skipped: {str(e)}")
            
            audio.export(output_file, format='wav', bitrate='192k')  # Lower bitrate for CPU
            return audio
        finally:
            for f in temp_files:
                cleanup_file(f)

    def _lightweight_mastering(self, audio: AudioSegment) -> AudioSegment:
        """Simplified mastering pipeline for CPU"""
        # Basic normalization
        audio = normalize(audio, headroom=1.0)
        
        # Lightweight effects
        if self.advanced_settings['intonation'] > 0:
            chunks = split_on_silence(
                audio,
                min_silence_len=100,  # Longer chunks for CPU
                silence_thresh=-30,
                keep_silence=100
            )
            processed = AudioSegment.silent(duration=0)
            
            for chunk in chunks:
                processed += chunk
                if random.random() < self.advanced_settings['breath_effects']:
                    processed += AudioSegment.silent(duration=100)
            
            audio = processed
        
        return self._lightweight_compress(audio)

# API Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "mode": "cpu"}

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "CPU-Optimized Voice API",
        "version": "2.1.0",
        "status": "running",
        "docs": "/docs",
        "limits": {
            "max_text_length": MAX_TEXT_LENGTH,
            "max_clone_duration": MAX_CLONE_DURATION
        }
    }

@app.post("/generate-standard", response_model=CloneResponse, tags=["TTS"])
async def generate_standard_voice(
    input: TextInput, 
    background_tasks: BackgroundTasks
):
    try:
        if not input.text:
            raise HTTPException(status_code=400, detail="No text provided")

        settings = input.settings or VoiceSettings()
        tts = UltraRealisticVoice(language=settings.language)
        
        if settings:
            tts.advanced_settings = {
                'vocal_enhance': settings.vocal_enhance,
                'breath_effects': settings.breath_effects,
                'intonation': settings.intonation,
                'articulation': settings.articulation
            }

        tts.add_text(input.text)
        output_file = TEMP_DIR / f"standard_{uuid.uuid4()}.wav"
        tts.generate(output_file=str(output_file))
        background_tasks.add_task(cleanup_file, str(output_file))

        return {
            "file_url": f"/temp_audio/{output_file.name}",
            "message": "Voice generated successfully"
        }
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/clone-voice", response_model=CloneResponse, tags=["Voice Cloning"])
async def clone_voice(
    background_tasks: BackgroundTasks,
    voice_file: UploadFile = File(...),
    text: str = Form(...),
    language: str = Form('en')
):
    try:
        if not text.strip() or len(text) > MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Text must be 1-{MAX_TEXT_LENGTH} characters"
            )

        # Save uploaded file
        temp_dir = tempfile.mkdtemp()
        original_path = os.path.join(temp_dir, f"original_{uuid.uuid4()}.wav")
        with open(original_path, "wb") as buffer:
            shutil.copyfileobj(voice_file.file, buffer)
        
        validate_audio_length(original_path)

        # Initialize TTS
        from TTS.api import TTS
        tts_engine = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
            gpu=False
        )
        
        output_file = TEMP_DIR / f"cloned_{uuid.uuid4()}.wav"
        tts_engine.tts_to_file(
            text=text,
            speaker_wav=original_path,
            language=language,
            file_path=str(output_file),
            emotion="Neutral",
            speed=1.0
        )

        # Cleanup
        background_tasks.add_task(cleanup_file, original_path)
        background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)

        return {
            "file_url": f"/temp_audio/{output_file.name}",
            "message": "Voice cloned successfully"
        }
    except Exception as e:
        logger.error(f"Cloning error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/temp_audio/{filename}", tags=["Files"])
async def get_temp_file(filename: str):
    file_path = TEMP_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for CPU
        log_level="info"
    )