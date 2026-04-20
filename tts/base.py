import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TTSResult:
    audio_path: str
    latency_ms: float       # time to first chunk — when audio can start playing
    full_duration_ms: float # total generation time
    audio_length_sec: float # how long the generated audio is
    rtf: float              # real time factor = generation_time / audio_length
                            # RTF < 1 means faster than realtime — what you want in production
    model_name: str
    text: str               # keeping input text so results are self-explanatory


class BaseTTS(ABC):
    """
    Same pluggable pattern as ASR and LLM.
    RTF is the metric that actually matters here — latency numbers alone
    are misleading because a model that takes 500ms to generate 10 seconds
    of audio is way better than one that takes 500ms for 2 seconds.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def synthesize(self, text: str, output_path: str) -> TTSResult:
        pass

    def unload(self):
        """Free memory between models. Override if model needs special cleanup."""
        import torch
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()