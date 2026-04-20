import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResult:
    response: str
    ttft_ms: float          # time to first token — the number that matters most in voice
    total_latency_ms: float
    tokens_per_sec: float
    token_count: int
    model_name: str
    prompt: str             # keeping the prompt so results make sense when reading JSON later


class BaseLLM(ABC):
    """
    Same pattern as BaseASR — every LLM implements this so the
    benchmark runner doesn't care which model it's talking to.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def generate(self, prompt: str) -> LLMResult:
        """
        Generate a response and return LLMResult.
        TTFT must be captured via streaming — don't just wait for
        full generation and then measure, that defeats the point.
        """
        pass

    def unload(self):
        """
        Free GPU memory after benchmarking this model.
        With 80GB we probably don't need this, but good habit —
        especially when running many models back to back.
        """
        import torch
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()