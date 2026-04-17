import time
from faster_whisper import WhisperModel
from .base import BaseASR, ASRResult


class FasterWhisperASR(BaseASR):
    """
    faster-whisper is CTranslate2-based — it runs significantly faster than
    the original OpenAI whisper with the same accuracy. Good starting point
    for the benchmark since it's the most commonly used in production pipelines.
    """

    def __init__(self, model_size: str = "medium", device: str = "cuda", compute_type: str = "float16"):
        super().__init__(model_name=f"faster-whisper-{model_size}")
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type  # float16 on GPU, int8 on CPU
        self.model = None

    def load(self):
        # Downloads model on first run, cached locally after that
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )

    def transcribe(self, audio_path: str) -> ASRResult:
        if self.model is None:
            raise RuntimeError("Call load() before transcribe()")

        start = time.perf_counter()

        # beam_size=5 is the default — gives good accuracy without being too slow
        # vad_filter removes silence at start/end, helps with latency
        segments, info = self.model.transcribe(audio_path, beam_size=5, vad_filter=True)

        # segments is a generator — need to consume it to get the full transcript
        transcript = " ".join(segment.text.strip() for segment in segments)

        latency_ms = (time.perf_counter() - start) * 1000

        return ASRResult(
            transcript=transcript,
            latency_ms=latency_ms,
            model_name=self.model_name,
        )