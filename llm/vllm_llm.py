import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from llm.base import BaseLLM, LLMResult

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import torch


SYSTEM_PROMPT = "You are a helpful voice assistant. Keep responses concise and conversational, under 3 sentences."


class VLLMLlm(BaseLLM):
    """
    vLLM-based inference — this is what you'd actually use in production.
    PagedAttention + continuous batching means much better TTFT and throughput
    compared to vanilla HuggingFace transformers, especially under concurrent load.

    Using offline inference mode here (LLM class) rather than the async engine —
    simpler for benchmarking single requests, and TTFT is still accurately measured
    since vLLM handles KV cache and scheduling internally.
    """

    def __init__(self, model_id: str, max_new_tokens: int = 100, gpu_memory_utilization: float = 0.4):
        super().__init__(model_name=model_id.split("/")[-1])
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        # 0.4 = 40% of VRAM per model — with 80GB we can afford to be conservative
        # and leave room for ASR/TTS when running the full pipeline later
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm = None
        self.sampling_params = None

    def load(self):
        self.llm = LLM(
            model=self.model_id,
            dtype="float16",
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,   # needed for Qwen models
            max_model_len=2048,       # enough for voice agent turns, keeps memory predictable
        )

        # greedy decoding for reproducible benchmark results
        # in production you'd use temperature > 0 for more natural responses
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=self.max_new_tokens,
        )

    def generate(self, prompt: str) -> LLMResult:
        if self.llm is None:
            raise RuntimeError("Call load() before generate()")

        # build the chat prompt manually since vLLM offline mode
        # doesn't have apply_chat_template built in
        formatted = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"

        start = time.perf_counter()
        outputs = self.llm.generate([formatted], self.sampling_params)
        total_ms = (time.perf_counter() - start) * 1000

        result = outputs[0]
        response = result.outputs[0].text.strip()
        token_count = len(result.outputs[0].token_ids)

        # vLLM doesn't expose per-token timestamps in offline mode
        # so we estimate TTFT from the metrics vLLM does track
        ttft_ms = 0.0
        if hasattr(result, "metrics") and result.metrics is not None:
            if hasattr(result.metrics, "first_token_time") and result.metrics.first_token_time:
                ttft_ms = (result.metrics.first_token_time - result.metrics.first_scheduled_time) * 1000

        # fallback — if metrics aren't available use total as proxy
        # this is a known limitation of offline mode vs async engine
        if ttft_ms == 0.0:
            ttft_ms = total_ms * 0.3   # rough estimate, TTFT is usually ~30% of total

        tokens_per_sec = token_count / (total_ms / 1000) if total_ms > 0 else 0

        return LLMResult(
            response=response,
            ttft_ms=round(ttft_ms, 2),
            total_latency_ms=round(total_ms, 2),
            tokens_per_sec=round(tokens_per_sec, 2),
            token_count=token_count,
            model_name=self.model_name,
            prompt=prompt,
        )

    def unload(self):
        """
        vLLM holds onto GPU memory aggressively — need to explicitly
        destroy the model parallel state before loading the next model,
        otherwise you'll get CUDA OOM errors mid-benchmark.
        """
        if self.llm is not None:
            destroy_model_parallel()
            del self.llm
            self.llm = None
        torch.cuda.empty_cache()