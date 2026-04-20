import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
import soundfile as sf
from tts.base import BaseTTS, TTSResult


class KokoroTTS(BaseTTS):
    """
    Best open source TTS right now in terms of quality/latency tradeoff.
    Runs locally on GPU, no API calls, and sounds noticeably better than
    older options like Coqui or Piper.
    """

    VOICES = ["af_heart", "af_bella", "am_adam", "bf_emma", "bm_george"]

    def __init__(self, voice: str = "af_heart", device: str = "cuda"):
        super().__init__(model_name=f"kokoro-{voice}")
        self.voice = voice
        self.device = device
        self.pipeline = None

    def load(self):
        from kokoro import KPipeline
        self.pipeline = KPipeline(lang_code="a")  # "a" = American English

    def synthesize(self, text: str, output_path: str) -> TTSResult:
        if self.pipeline is None:
            raise RuntimeError("Call load() before synthesize()")

        start = time.perf_counter()

        first_chunk_time = None
        all_audio = []

        for _, _, audio in self.pipeline(text, voice=self.voice, speed=1.0):
            if first_chunk_time is None:
                first_chunk_time = (time.perf_counter() - start) * 1000
            all_audio.append(audio)

        full_duration_ms = (time.perf_counter() - start) * 1000

        full_audio = np.concatenate(all_audio)
        sf.write(output_path, full_audio, samplerate=24000)

        audio_length_sec = len(full_audio) / 24000
        rtf = (full_duration_ms / 1000) / audio_length_sec if audio_length_sec > 0 else 0

        return TTSResult(
            audio_path=output_path,
            latency_ms=first_chunk_time or full_duration_ms,
            full_duration_ms=full_duration_ms,
            audio_length_sec=round(audio_length_sec, 2),
            rtf=round(rtf, 3),
            model_name=self.model_name,
            text=text,
        )

    def unload(self):
        del self.pipeline
        self.pipeline = None
        import torch
        torch.cuda.empty_cache()