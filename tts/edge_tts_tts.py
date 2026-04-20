import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import asyncio
import soundfile as sf
import numpy as np
import edge_tts
from tts.base import BaseTTS, TTSResult


class EdgeTTS(BaseTTS):
    """
    Microsoft's cloud TTS — no GPU, no model loading, just an API call.
    Including this as a baseline because:
    1. It's what a lot of people default to when starting a voice project
    2. Interesting to see how a cloud API compares to local models on latency
    3. Zero setup cost — good reference point

    Downside: needs internet, not suitable for offline/on-prem deployments.
    """

    def __init__(self, voice: str = "en-US-AriaNeural"):
        super().__init__(model_name=f"edge-tts-{voice.split('-')[-1]}")
        self.voice = voice

    def load(self):
        # nothing to load — it's a cloud API
        pass

    def synthesize(self, text: str, output_path: str) -> TTSResult:
        return asyncio.run(self._synthesize_async(text, output_path))

    async def _synthesize_async(self, text: str, output_path: str) -> TTSResult:
        start = time.perf_counter()

        communicate = edge_tts.Communicate(text, self.voice)

        # stream chunks and write to file
        # first chunk time = when audio could start playing in a streaming setup
        first_chunk_time = None
        audio_data = b""

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                if first_chunk_time is None:
                    first_chunk_time = (time.perf_counter() - start) * 1000
                audio_data += chunk["data"]

        full_duration_ms = (time.perf_counter() - start) * 1000

        # save to disk
        with open(output_path, "wb") as f:
            f.write(audio_data)

        # measure how long the generated audio actually is
        audio_length_sec = self._get_audio_length(output_path)
        rtf = (full_duration_ms / 1000) / audio_length_sec if audio_length_sec > 0 else 0

        return TTSResult(
            audio_path=output_path,
            latency_ms=first_chunk_time or full_duration_ms,
            full_duration_ms=full_duration_ms,
            audio_length_sec=audio_length_sec,
            rtf=round(rtf, 3),
            model_name=self.model_name,
            text=text,
        )

    def _get_audio_length(self, path: str) -> float:
        try:
            data, sr = sf.read(path)
            return len(data) / sr
        except Exception:
            return 0.0

    def unload(self):
        pass  # nothing to unload