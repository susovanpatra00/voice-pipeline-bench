# voice-pipeline-bench

A benchmarking suite for real-time ASR → LLM → TTS voice pipelines.

I built this while working on voice agents because I kept running into the same problem — there's no straightforward way to compare models at each stage with real latency numbers. Most benchmarks test models in isolation. This one is designed around how voice pipelines actually work: you need ASR, LLM, and TTS to all be fast, and the numbers at each stage directly affect what's possible end-to-end.

If you're building a voice agent and trying to pick models, this should give you a useful starting point. You can also plug in your own models pretty easily — each stage has a base class and adding a new model is just implementing two methods.

---

## What's benchmarked

**ASR** — how fast can it transcribe, and does it get it right?
- Metrics: latency (ms) per audio type
- Models tested: faster-whisper tiny / small / medium / large-v3-turbo

**LLM** — time to first token is what matters for voice, not total generation time
- Metrics: TTFT (ms), total latency, tokens/sec
- Models tested: Qwen2.5-0.5B, Qwen2.5-1.5B, Llama-3.2-3B, Llama-3.1-8B
- Backend: vLLM (not HuggingFace transformers — this is what you'd actually use in production)

**TTS** — RTF (Real Time Factor) tells you more than raw latency here
- Metrics: first chunk latency, full generation time, RTF
- RTF < 1.0 = faster than realtime, which is the minimum bar for a usable voice pipeline
- Models tested: edge-tts, Kokoro

---

## Results

### ASR
Tested on 5 audio types. GPU, float16, 3 runs per model with 1 warmup.

| Model | short | long | booking | fast speech | numbers |
|---|---|---|---|---|---|
| faster-whisper-tiny | 121ms | 247ms | 186ms | 125ms | 186ms |
| faster-whisper-small | 167ms | 336ms | 261ms | 186ms | 224ms |
| faster-whisper-medium | 235ms | 465ms | 362ms | 269ms | 325ms |
| faster-whisper-large-v3-turbo | 332ms | 626ms | 452ms | 348ms | 428ms |

`small` is the sweet spot — accurate on numbers and longer sentences, fast enough to leave room for LLM + TTS. `tiny` makes transcription errors. `large-v3-turbo` at 626ms on a single sentence already blows a 600ms total budget.

### LLM (vLLM backend)
| Model | Avg TTFT | Avg Total | Tok/s |
|---|---|---|---|
| Qwen2.5-0.5B | ~13ms | ~247ms | ~348 |
| Qwen2.5-1.5B | ~16ms | ~446ms | ~196 |
| Llama-3.2-3B | ~16ms | ~363ms | ~144 |
| Llama-3.1-8B | ~26ms | ~621ms | ~78 |

All TTFTs under 40ms — vLLM's PagedAttention makes a real difference here. `Qwen2.5-1.5B` is probably the best pick for voice: barely slower than 0.5B on TTFT but noticeably better response quality.

### TTS
| Model | Text | First Chunk | RTF |
|---|---|---|---|
| edge-tts | short | ~280ms | cloud |
| edge-tts | long | ~900ms | cloud |
| kokoro | short | ~35ms | 0.007 |
| kokoro | long | ~79ms | 0.007 |

Kokoro's RTF of 0.007 means it generates 12 seconds of audio in ~79ms — roughly 150x faster than realtime. edge-tts is a useful baseline since it requires zero GPU, but the latency is higher and it needs internet.

---

## Project structure

```
voice-pipeline-bench/
├── asr/
│   ├── base.py                  # BaseASR class — implement this to add a new ASR model
│   └── faster_whisper_asr.py
├── llm/
│   ├── base.py                  # BaseLLM class — implement this to add a new LLM
│   └── vllm_llm.py
├── tts/
│   ├── base.py                  # BaseTTS class — implement this to add a new TTS model
│   ├── kokoro_tts.py
│   └── edge_tts_tts.py
├── benchmark/
│   ├── run_asr_benchmark.py
│   ├── run_llm_benchmark.py
│   └── run_tts_benchmark.py
├── configs/
│   ├── asr_benchmark_config.yaml
│   ├── llm_benchmark_config.yaml
│   └── tts_benchmark_config.yaml
├── samples/                     # test audio files
└── results/                     # benchmark JSON outputs + generated TTS audio
```

---

## Run it

```bash
git clone https://github.com/susovanpatra00/voice-pipeline-bench
cd voice-pipeline-bench
uv sync

# run each benchmark separately
python benchmark/run_asr_benchmark.py
python benchmark/run_llm_benchmark.py
python benchmark/run_tts_benchmark.py
```

To change which models run or add new test inputs, edit the corresponding config file in `configs/`.

---

## Adding your own model

Each stage has a base class with two methods to implement: `load()` and the inference method (`transcribe` / `generate` / `synthesize`). The benchmark runner picks it up automatically once you add it to the config.

For example, to add a new ASR model:

```python
from asr.base import BaseASR, ASRResult

class MyASR(BaseASR):
    def load(self):
        # load your model here
        pass

    def transcribe(self, audio_path: str) -> ASRResult:
        # run inference, return ASRResult with transcript + latency
        pass
```

Then add it to `configs/asr_benchmark_config.yaml` and run.

---