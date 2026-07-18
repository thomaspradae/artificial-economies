# Local LLM Benchmark: ofi1

Date: 2026-07-18

Host: `uace-ofi-01`

- CPU: Intel Core i7-4790, 4 cores / 8 threads
- RAM after removing `poormans-adopt1`: about 6.8 GiB available
- Ollama: user-local install at `~/.local/bin/ollama`, version `0.32.1`
- Runtime mode: CPU-only, `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_KEEP_ALIVE=0`

## Models Tested

| Model | Size | Prompt mode | Load seconds | Output tok/s | Output quality |
| --- | ---: | --- | ---: | ---: | --- |
| `qwen3:1.7b` | 1.4 GB | `/api/chat`, `think=false` | 16.4 | 16.6 | Fast, visible structured output, but weaker extraction depth. |
| `qwen3:4b` | 2.5 GB | `/api/chat`, `think=false` | 32.5 | 8.0 | Viable speed, but still emits reasoning/prose before JSON under strict prompts. |
| `llama3.2:3b` | 2.0 GB | `/api/chat` | 29.3 | 9.9 | Best bulk extractor: returns compact JSON reliably under strict prompt. |
| `qwen3:8b` | 5.2 GB | `/api/chat`, `think=false` | 46.9 | 4.1 | Cleanest output, but too slow for bulk extraction on this CPU box. |

## Recommendation

Use `llama3.2:3b` as the default local paper-card extractor for bulk runs.

Use `qwen3:8b` only for slower audit passes on high-value papers where cleaner extraction matters more than throughput.

Do not use `qwen3:*` through `/api/generate` for this pipeline unless the extractor explicitly disables thinking with a supported chat call. In `/api/generate`, Qwen3 consumed the output budget in hidden/thinking tokens and returned empty visible text.

## Operational Notes

- Keep `OLLAMA_MAX_LOADED_MODELS=1`; loading multiple models at once pushed `ofi1` into swap.
- Use `keep_alive=0` per extraction request when sweeping many papers.
- Keep output caps tight for first-pass paper cards, then rerun selected cards with a larger model if needed.
