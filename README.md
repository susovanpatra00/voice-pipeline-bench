# voice-pipeline-bench

Benchmarking suite for real-time ASR → LLM → TTS voice pipelines.

I built this because when I was working on voice agents, there was no good way to compare models across the full pipeline with actual latency numbers. Most benchmarks test models in isolation — this one tests them the way they actually run in production: chained together, under load, with real audio.

---

## What it does

- Runs any ASR / LLM / TTS model through a standardized pipeline
- Measures latency at each stage separately (ASR time, LLM TTFT, TTS time, E2E)
- Tests across multiple audio types — short, long, fast speech, numbers, noisy
- Averages over N runs to smooth out noise
- Saves results as JSON so you can track changes over time

---

## ASR Results

Tested on 5 audio samples: short sentence, long sentence, flight booking query, fast speech, numbers.
All runs on a single GPU, `float16`, 3 runs per model with 1 warmup.

| Model | short | long | booking | fast | numbers |
|---|---|---|---|---|---|
| faster-whisper-tiny | 121ms | 247ms | 186ms | 125ms | 186ms |
| faster-whisper-small | 167ms | 336ms | 261ms | 186ms | 224ms |
| faster-whisper-medium | 235ms | 465ms | 362ms | 269ms | 325ms |
| faster-whisper-large-v3-turbo | 332ms | 626ms | 452ms | 348ms | 428ms |

**Takeaway:** `small` hits the best accuracy/latency tradeoff for voice pipelines.
`tiny` is faster but makes transcription errors on numbers and longer sentences.
`large-v3-turbo` at 626ms on a long sentence alone blows a typical 600ms E2E budget.

---

## Project Structure

```
voice-pipeline-bench/
├── asr/              # ASR implementations (faster-whisper, wav2vec2, mms)
├── llm/              # LLM implementations (Qwen, LLaMA, Mistral, Phi3) — WIP
├── tts/              # TTS implementations (Kokoro, Piper, Edge TTS) — WIP
├── pipeline/         # Orchestrator + profiler
├── benchmark/        # CLI runner
├── configs/          # YAML configs for each benchmark run
├── samples/          # Test audio files
└── results/          # Saved JSON results
```

---

## Run it yourself

```bash
git clone https://github.com/susovanpatra00/voice-pipeline-bench
cd voice-pipeline-bench
uv sync
python benchmark/run_benchmark.py
```

To change which models run, edit `configs/benchmark_config.yaml`.

---
