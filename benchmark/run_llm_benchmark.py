import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

"""
LLM Benchmark — Phase 2
Tests multiple HuggingFace models on realistic voice agent prompts.
Measures TTFT, total latency, and tokens/sec.

Usage:
    python benchmark/run_llm_benchmark.py
    python benchmark/run_llm_benchmark.py --config configs/llm_benchmark_config.yaml
"""

import json
import argparse
import yaml
import torch
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table

from llm.vllm_llm import VLLMLlm

console = Console()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_llm_benchmark(config: dict):
    models = config["llm"]["models"]
    prompts = config["benchmark"]["prompts"]
    n_runs = config["benchmark"].get("runs", 3)
    warmup = config["benchmark"].get("warmup_runs", 1)
    output_dir = Path(config["benchmark"].get("output_dir", "results"))
    output_dir.mkdir(exist_ok=True)

    all_results = []

    for model_cfg in models:
        model_id = model_cfg["id"]
        console.print(f"\n[bold cyan]Loading: {model_id}[/bold cyan]")

        llm = VLLMLlm(
            model_id=model_cfg["id"],
            max_new_tokens=config["llm"].get("max_new_tokens", 100),
            gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.4),
        )
        llm.load()

        for prompt_cfg in prompts:
            label = prompt_cfg["label"]
            prompt = prompt_cfg["text"]

            console.print(f"\n  [yellow]Prompt: {label}[/yellow]")
            console.print(f"  [dim]\"{prompt}\"[/dim]")

            # warmup — first run is always slower, don't count it
            if warmup > 0:
                console.print(f"  [dim]Warming up...[/dim]")
                for _ in range(warmup):
                    llm.generate(prompt)

            ttfts = []
            totals = []
            tps_list = []
            response = ""

            for i in range(n_runs):
                result = llm.generate(prompt)
                ttfts.append(result.ttft_ms)
                totals.append(result.total_latency_ms)
                tps_list.append(result.tokens_per_sec)
                response = result.response
                console.print(
                    f"    Run {i+1}: TTFT [green]{result.ttft_ms:.1f}ms[/green] | "
                    f"Total [green]{result.total_latency_ms:.1f}ms[/green] | "
                    f"[dim]{result.tokens_per_sec:.1f} tok/s[/dim]"
                )

            all_results.append({
                "model": llm.model_name,
                "prompt_label": label,
                "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 2),
                "avg_total_ms": round(sum(totals) / len(totals), 2),
                "avg_tokens_per_sec": round(sum(tps_list) / len(tps_list), 2),
                "sample_response": response,
            })

        # free GPU memory before loading next model
        llm.unload()
        console.print(f"\n[dim]Unloaded {llm.model_name}[/dim]")

    _print_results_table(all_results)
    _save_results(all_results, output_dir)


def _print_results_table(results: list):
    console.print("\n")
    table = Table(title="LLM Benchmark Results — TTFT", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("Prompt", style="white")
    table.add_column("Avg TTFT", style="green")
    table.add_column("Avg Total", style="green")
    table.add_column("Tok/s", style="yellow")

    for r in results:
        table.add_row(
            r["model"],
            r["prompt_label"],
            f"{r['avg_ttft_ms']:.1f} ms",
            f"{r['avg_total_ms']:.1f} ms",
            f"{r['avg_tokens_per_sec']:.1f}",
        )

    console.print(table)


def _save_results(results: list, output_dir: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = output_dir / f"llm_benchmark_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump({"timestamp": timestamp, "results": results}, f, indent=2)
    console.print(f"\n[dim]Results saved to {fname}[/dim]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llm_benchmark_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_llm_benchmark(config)


if __name__ == "__main__":
    main()