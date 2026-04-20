"""
ASR Benchmark — Phase 1
Tests faster-whisper at different sizes and compares latency + accuracy.

Usage:
    python benchmark/run_benchmark.py
    python benchmark/run_benchmark.py --config configs/benchmark_config.yaml
    python benchmark/run_benchmark.py --audio samples/test.wav
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table

from asr.faster_whisper_asr import FasterWhisperASR

console = Console()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_asr_benchmark(config: dict, audio_files: list):
    asr_configs = config["asr"]["models"]
    bench_cfg = config["benchmark"]
    n_runs = bench_cfg.get("runs", 3)
    warmup = bench_cfg.get("warmup_runs", 1)
    output_dir = Path(bench_cfg.get("output_dir", "results"))
    output_dir.mkdir(exist_ok=True)

    all_results = []

    for asr_cfg in asr_configs:
        model_name = asr_cfg["name"]
        console.print(f"\n[bold cyan]Testing: {model_name}[/bold cyan]")

        asr = FasterWhisperASR(
            model_size=asr_cfg["size"],
            device=asr_cfg.get("device", "cuda"),
            compute_type=asr_cfg.get("compute_type", "float16"),
        )
        asr.load()

        for audio in audio_files:
            audio_path = audio["path"]
            label = audio["label"]

            if not Path(audio_path).exists():
                console.print(f"  [red]Skipping {label} — file not found: {audio_path}[/red]")
                continue

            console.print(f"  [yellow]Audio: {label}[/yellow]")

            # warmup
            if warmup > 0:
                for _ in range(warmup):
                    asr.transcribe(audio_path)

            latencies = []
            transcript = ""
            for i in range(n_runs):
                result = asr.transcribe(audio_path)
                latencies.append(result.latency_ms)
                transcript = result.transcript
                console.print(f"    Run {i+1}: [green]{result.latency_ms:.1f}ms[/green]")

            avg_latency = sum(latencies) / len(latencies)
            all_results.append({
                "model": model_name,
                "audio": label,
                "avg_latency_ms": round(avg_latency, 2),
                "min_latency_ms": round(min(latencies), 2),
                "max_latency_ms": round(max(latencies), 2),
                "transcript": transcript,
            })

        del asr

    _print_results_table(all_results)
    _save_results(all_results, output_dir)


def _print_results_table(results: list[dict]):
    console.print("\n")
    table = Table(title="ASR Benchmark Results", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("Avg Latency", style="green")
    table.add_column("Min", style="green")
    table.add_column("Max", style="yellow")
    table.add_column("Transcript (preview)", style="white", max_width=40)

    for r in results:
        table.add_row(
            r["model"],
            f"{r['avg_latency_ms']:.1f} ms",
            f"{r['min_latency_ms']:.1f} ms",
            f"{r['max_latency_ms']:.1f} ms",
            r["transcript"][:80] + "..." if len(r["transcript"]) > 80 else r["transcript"],
        )

    console.print(table)


def _save_results(results: list[dict], output_dir: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = output_dir / f"asr_benchmark_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump({"timestamp": timestamp, "results": results}, f, indent=2)
    console.print(f"\n[dim]Results saved to {fname}[/dim]")

def main():
    parser = argparse.ArgumentParser(description="ASR benchmark runner")
    parser.add_argument("--config", default="configs/asr_benchmark_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    bench_cfg = config["benchmark"]

    # support both old single-file format and new multi-file format
    if "test_audios" in bench_cfg:
        audio_files = bench_cfg["test_audios"]
    else:
        audio_files = [{"path": bench_cfg["test_audio"], "label": "default"}]

    # check at least one file exists before starting
    found = [a for a in audio_files if Path(a["path"]).exists()]
    missing = [a for a in audio_files if not Path(a["path"]).exists()]

    if missing:
        for a in missing:
            console.print(f"[yellow]Warning: audio file not found, skipping — {a['path']}[/yellow]")

    if not found:
        console.print("[red]No audio files found. Add .wav files to samples/ and update the config.[/red]")
        return

    run_asr_benchmark(config, found)


if __name__ == "__main__":
    main()