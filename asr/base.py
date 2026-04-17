from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ASRResult:
    transcript: str
    latency_ms: float   # wall-clock time from audio start → transcript ready
    model_name: str     # keeping track so results don't get mixed up in bench runs


class BaseASR(ABC):
    """
    Every ASR model I add needs to follow this interface.
    That way the pipeline doesn't care which model is running underneath —
    faster-whisper, wav2vec2, whatever — it just calls .transcribe() and gets back
    an ASRResult. Swapping models becomes a one-line config change.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def load(self):
        """
        Load the model weights into memory (CPU or GPU).
        Keeping this separate from __init__ so I can control exactly when
        the GPU memory gets allocated — important when benchmarking multiple
        models back to back without OOM-ing.
        """
        pass

    @abstractmethod
    def transcribe(self, audio_path: str) -> ASRResult:
        """
        Takes a path to an audio file, returns the transcript + how long it took.
        All implementations should measure latency internally so the profiler
        doesn't have to do any guesswork.
        """
        pass
