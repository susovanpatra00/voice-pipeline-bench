import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

"""
TTS Benchmark — Phase 3
Tests edge-tts, Kokoro, and Qwen3-TTS across short/medium/long texts.
Key metric is RTF (Real Time Factor) — generation time / audio length.
RTF < 1 means faster than realtime, which is the minimum bar for voice pipelines.

Usage:
    python benchmark/run_tts_benchmark.py
    python benchmark/run_tts_benchmark.py --config configs/tts_benchmark_config.yaml
"""

import json
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table

from tts.edge_tts_tts import EdgeTTS
from tts.kokoro_tts import KokoroTTS
from tts.qwen3_tts import Qwen3TTS

console = Console()

TTS_MAP = {
    "edge": lambda cfg: EdgeTTS(voice=cfg.get("voice", "en-US-AriaNeural")),
    "kokoro": lambda cfg: KokoroTTS(voice=cfg.get("voice", "af_heart"), device=cfg.get("device", "cuda")),
    "qwen3": lambda cfg: Qwen3TTS(device=cfg.get("device", "cuda")),
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def run_tts_benchmark(config: dict):
    models_cfg = config["tts"]["models"]
    texts = config["benchmark"]["texts"]
    n_runs = config["benchmark"].get("runs", 3)
    warmup = config["benchmark"].get("warmup_runs", 1)
    output_dir = Path(config["benchmark"].get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # wav files go here, JSON results go in output_dir root
    audio_dir = output_dir / "tts_audio"
    audio_dir.mkdir(exist_ok=True)

    all_results = []

    for model_cfg in models_cfg:
        model_type = model_cfg["type"]
        model_name = model_cfg["name"]
        console.print(f"\n[bold cyan]Loading: {model_name}[/bold cyan]")

        tts = TTS_MAP[model_type](model_cfg)
        tts.load()

        for text_cfg in texts:
            label = text_cfg["label"]
            text = text_cfg["text"]

            console.print(f"\n  [yellow]Text: {label}[/yellow]")
            console.print(f"  [dim]\"{text[:60]}...\"[/dim]" if len(text) > 60 else f"  [dim]\"{text}\"[/dim]")

            if warmup > 0:
                console.print(f"  [dim]Warming up...[/dim]")
                warmup_path = str(audio_dir / f"warmup_{model_name}.wav")
                for _ in range(warmup):
                    tts.synthesize(text, warmup_path)

            latencies = []
            full_durations = []
            rtfs = []
            audio_lengths = []

            for i in range(n_runs):
                out_path = str(audio_dir / f"{model_name}_{label}_run{i}.wav")  # ← audio_dir, one loop
                result = tts.synthesize(text, out_path)
                latencies.append(result.latency_ms)
                full_durations.append(result.full_duration_ms)
                rtfs.append(result.rtf)
                audio_lengths.append(result.audio_length_sec)

                console.print(
                    f"    Run {i+1}: "
                    f"first chunk [green]{result.latency_ms:.0f}ms[/green] | "
                    f"total [green]{result.full_duration_ms:.0f}ms[/green] | "
                    f"RTF [yellow]{result.rtf:.3f}[/yellow] | "
                    f"audio [dim]{result.audio_length_sec:.1f}s[/dim]"
                )

            all_results.append({
                "model": model_name,
                "text_label": label,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
                "avg_full_duration_ms": round(sum(full_durations) / len(full_durations), 2),
                "avg_rtf": round(sum(rtfs) / len(rtfs), 3),
                "avg_audio_length_sec": round(sum(audio_lengths) / len(audio_lengths), 2),
            })

        tts.unload()
        console.print(f"\n[dim]Unloaded {model_name}[/dim]")

    _print_results_table(all_results)
    _save_results(all_results, output_dir)


def _print_results_table(results: list):
    console.print("\n")
    table = Table(title="TTS Benchmark Results", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("Text", style="white")
    table.add_column("First Chunk", style="green")
    table.add_column("Full Gen", style="green")
    table.add_column("RTF", style="yellow")
    table.add_column("Audio Len", style="dim")

    for r in results:
        # color RTF — green if < 0.5, yellow if < 1.0, red if >= 1.0
        rtf = r["avg_rtf"]
        rtf_color = "green" if rtf < 0.5 else "yellow" if rtf < 1.0 else "red"
        rtf_str = f"[{rtf_color}]{rtf:.3f}[/{rtf_color}]"

        table.add_row(
            r["model"],
            r["text_label"],
            f"{r['avg_latency_ms']:.0f} ms",
            f"{r['avg_full_duration_ms']:.0f} ms",
            rtf_str,
            f"{r['avg_audio_length_sec']:.1f}s",
        )

    console.print(table)


def _save_results(results: list, output_dir: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = output_dir / f"tts_benchmark_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump({"timestamp": timestamp, "results": results}, f, indent=2)
    console.print(f"\n[dim]Results saved to {fname}[/dim]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tts_benchmark_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_tts_benchmark(config)


if __name__ == "__main__":
    main()