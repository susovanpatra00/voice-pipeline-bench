import time
import psutil
import GPUtil
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Optional


@dataclass
class PipelineProfile:
    """
    One of these gets filled in per benchmark run.
    Keeping it as a dataclass so it's easy to serialize to JSON/CSV later
    without writing custom serialization logic.
    """
    asr_latency_ms: float = 0.0
    llm_ttft_ms: float = 0.0
    llm_total_ms: float = 0.0
    tts_latency_ms: float = 0.0
    e2e_latency_ms: float = 0.0     # this is the number that actually matters
    gpu_memory_mb: float = 0.0      # peak VRAM during the run
    cpu_percent: float = 0.0
    asr_model: str = ""
    llm_model: str = ""
    tts_model: str = ""

    def display(self):
        """Pretty-print results to terminal using rich tables."""
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table = Table(title="📊 Pipeline Benchmark Results", show_lines=True)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green")

        table.add_row("ASR Model", self.asr_model)
        table.add_row("LLM Model", self.llm_model)
        table.add_row("TTS Model", self.tts_model)
        table.add_row("─" * 18, "─" * 18)
        table.add_row("ASR Latency", f"{self.asr_latency_ms:.1f} ms")
        table.add_row("LLM TTFT", f"{self.llm_ttft_ms:.1f} ms")
        table.add_row("LLM Total", f"{self.llm_total_ms:.1f} ms")
        table.add_row("TTS Latency", f"{self.tts_latency_ms:.1f} ms")
        table.add_row("E2E Latency ⚡", f"{self.e2e_latency_ms:.1f} ms")
        table.add_row("─" * 18, "─" * 18)
        table.add_row("GPU Memory", f"{self.gpu_memory_mb:.0f} MB")
        table.add_row("CPU Usage", f"{self.cpu_percent:.1f}%")

        console.print(table)

    def to_dict(self) -> dict:
        """For saving results to JSON. Nothing fancy."""
        return {
            "asr_model": self.asr_model,
            "llm_model": self.llm_model,
            "tts_model": self.tts_model,
            "asr_latency_ms": round(self.asr_latency_ms, 2),
            "llm_ttft_ms": round(self.llm_ttft_ms, 2),
            "llm_total_ms": round(self.llm_total_ms, 2),
            "tts_latency_ms": round(self.tts_latency_ms, 2),
            "e2e_latency_ms": round(self.e2e_latency_ms, 2),
            "gpu_memory_mb": round(self.gpu_memory_mb, 2),
            "cpu_percent": round(self.cpu_percent, 2),
        }


class Profiler:
    """
    Wraps each pipeline stage to measure latency without cluttering
    the actual model code with timing logic.

    Usage:
        profiler = Profiler()
        with profiler.measure("asr"):
            result = asr.transcribe(audio_path)
        print(profiler.get("asr"))  # → latency in ms
    """

    def __init__(self):
        self._timings: dict = {}

    @contextmanager
    def measure(self, stage: str):
        """
        Context manager for timing a stage.
        Using time.perf_counter() instead of time.time() — much higher resolution,
        which matters when you're trying to measure differences of 10-50ms.
        """
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        self._timings[stage] = (end - start) * 1000  # store in ms

    def get(self, stage: str) -> float:
        """Return latency for a given stage in ms. Returns 0 if not measured yet."""
        return self._timings.get(stage, 0.0)

    @staticmethod
    def get_gpu_memory() -> float:
        """
        Returns current GPU memory usage in MB.
        Falls back to 0 if no GPU is available — makes local CPU testing easier.
        """
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed   # just taking GPU 0 for now
            return 0.0
        except Exception:
            return 0.0

    @staticmethod
    def get_cpu_percent() -> float:
        """Snapshot of CPU usage at this moment. Non-blocking."""
        return psutil.cpu_percent(interval=None)