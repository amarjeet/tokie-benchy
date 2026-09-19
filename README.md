# tokie-benchy

Token throughput / latency benchmarking CLI with a live TUI for any
OpenAI-compatible chat endpoint (vLLM, llama.cpp, LiteLLM, Ollama, OpenAI, ...).

Measures per request:

| key          | meaning                                                   |
|--------------|-----------------------------------------------------------|
| `ttft`       | time to first token (ms)                                  |
| `tps`        | output tokens/s during decode (after first token)         |
| `itl`        | mean inter-token latency (ms)                             |
| `e2e`        | end-to-end request latency (ms)                           |
| `prefill`    | prompt tokens/s (prompt tokens / TTFT, approx prefill)    |
| `throughput` | aggregate output tokens/s across all concurrent requests  |

## Profiles

| profile | prompt tokens | max tokens | requests | concurrency | measurements                        |
|---------|---------------|------------|----------|-------------|-------------------------------------|
| `L`     | 128           | 64         | 6        | 1           | ttft, tps, e2e                      |
| `M`     | 512           | 256        | 12       | 4           | ttft, tps, itl, e2e, throughput     |
| `H`     | 2048          | 512        | 24       | 8           | all                                 |

Every profile value can be overridden from the CLI. Prompt sizes are approximate;
the server-reported `prompt_tokens` is what gets recorded in results.

## Configuration

Environment variables (a `.env` in the current directory is loaded automatically,
but variables already exported in your shell always win):

```
OPENAI_API_URL=http://host:port/v1     # also accepts OPENAI_BASE_URL / OPENAI_API_BASE
OPENAI_API_KEY=sk-...
OPENAI_MODEL=my-model
```

## Usage

```bash
uv sync

# medium profile, live TUI
uv run tokie-benchy run

# heavy profile, 16 parallel requests, save results
uv run tokie-benchy run -p H -c 16 -o results/h16.json

# all three profiles back-to-back in one TUI session
uv run tokie-benchy run -p L -p M -p H

# custom sizes, only TTFT + TPS, headless (CI-friendly)
uv run tokie-benchy run -p M --prompt-tokens 1024 --max-tokens 128 -n 20 -c 8 -m ttft,tps --no-tui

# override endpoint from the shell
OPENAI_MODEL=other-model uv run tokie-benchy run -p L

uv run tokie-benchy profiles      # list profiles and measurement keys
```

TUI keys: `n` configure a new run, `s` save JSON results, `d` toggle dark mode, `q` quit.
`ctrl+p` opens the command palette, which also lists "New run" and "Save results".

### Configuring runs inside the TUI

Press `n` (or pick *New run* from the `ctrl+p` palette) to open a settings dialog
covering everything the CLI flags cover: profiles, the four size/parallelism
overrides (concurrency accepts a sweep list), request rate and arrival pattern,
exact output, measurements, endpoint (URL, model, key), thinking effort and
style, extra body JSON, timeout, warmup, seed, temperature and output path. `ctrl+s` or *start run* validates the form and
starts immediately; if a run is still in progress it is cancelled and the stats
table, charts and log are cleared first. `esc` cancels the dialog and leaves the
current run untouched.

To start the TUI directly in that dialog and enter every parameter there:

```bash
uv run tokie-benchy run --configure        # or -C
```

Fields are pre-filled from the environment / `.env` and from any flags you passed,
so `run -C -p H -c 16` opens the dialog with H checked and concurrency 16 ready to edit.

## CLI flags (`run`)

| flag | default | what it controls |
|------|---------|------------------|
| `-p, --profile L\|M\|H` | `M` | Profile to run. Repeat (`-p L -p M -p H`) to run several stages back-to-back in one session; charts and stats are colored per profile. |
| `--prompt-tokens N` | profile | Approximate size of each generated prompt in tokens. Exercises prefill: TTFT and `prefill` tok/s scale with this. The server-reported `prompt_tokens` is what gets recorded. |
| `--max-tokens N` | profile | Output cap per request. Exercises decode: `tps`, `itl` and aggregate `throughput` need enough output tokens to stabilise (>=128 is a good floor). |
| `-n, --requests N` | profile | Total requests per profile. More requests give tighter p95/p99. |
| `-c, --concurrency N[,N...]` | profile | Closed loop: requests held in flight at once. A comma list such as `1,2,4,8` runs a **sweep**, one stage per level, and adds a load-sweep chart/table. |
| `--request-rate R[,R...]` | none | Open loop: launch R requests/s regardless of completions, with no concurrency cap. A comma list sweeps rates. Reports peak in-flight so you can see queue build-up. |
| `--arrival constant\|poisson` | `constant` | Inter-arrival distribution for `--request-rate`. Poisson gives realistic bursty traffic, reproducible via `--seed`. |
| `--exact-output` | off | Adds `ignore_eos: true` and `min_tokens = max_tokens` so every reply is exactly `--max-tokens` long, making decode numbers comparable. The stage log reports how many replies actually hit the target. |
| `-m, --measure a,b,c` | profile | Which measurements to compute/show: `ttft,tps,itl,e2e,prefill,throughput`. |
| `--model NAME` | `$OPENAI_MODEL` | Model id sent in the request body. |
| `--base-url URL` | `$OPENAI_API_URL` | OpenAI-compatible base URL ending in `/v1`. |
| `--api-key KEY` | `$OPENAI_API_KEY` | Bearer token; omit for unauthenticated local servers. |
| `--timeout S` | `120` | Per-request timeout in seconds (read timeout between stream chunks and total connect). Raise for long prompts or long outputs. |
| `--warmup / --no-warmup` | on | One small request before measuring, so model load / cold caches do not land in request #0. |
| `--tui / --no-tui` | on | Live TUI, or headless Rich progress + summary table (exit code 1 if any request failed). |
| `-C, --configure` | off | Open the TUI settings dialog first instead of starting right away; all parameters can be entered there. |
| `-o, --output PATH` | none | Write per-request and per-stage results as JSON. Includes a `repro` field with the exact CLI to repeat the run. |
| `-r, --reasoning-effort LEVEL` | `low` | Thinking budget: `off`, `minimal`, `low`, `medium`, `high`, `max`. Translated into whichever parameter the endpoint understands (see below). |
| `--reasoning-style STYLE` | `auto` | Which spelling to send: `auto` (probe the endpoint and keep the one that provably works), one of `openai`, `openrouter`, `anthropic`, `deepseek`, `gemini`, `qwen`, `ollama` (send as-is, no probe), or `none` (send nothing). |
| `--extra-body JSON` | none | Raw JSON object deep-merged into every request body, e.g. `'{"top_p": 0.9}'`. Applied last, so it overrides any detected thinking field. |
| `--seed N` | `42` | Prompt generator seed. Each request uses `seed + index`, so prompts differ per request and server prefix caching does not flatter TTFT. |
| `--temperature T` | `0.7` | Sampling temperature. |
| `--env-file PATH` | `./.env` | Alternate `.env`. Variables already exported in the shell always win over the file. |

## Load patterns and sweeps

Two ways to apply load:

- **Closed loop** (`-c N`): N requests are always in flight; a new one starts when
  one finishes. Measures the server at a fixed parallelism.
- **Open loop** (`--request-rate R`): requests are launched on a clock, R per
  second, whether or not earlier ones finished. This is how a real harness or
  user population behaves, and it exposes queue collapse that a fixed
  concurrency never shows. `--arrival poisson` randomises the gaps. Watch the
  `in flight` counter and the reported peak: if it climbs stage after stage, the
  server cannot keep up with that rate.

A comma list on either flag runs a **sweep**: one stage per level, in order, in
one session. The TUI gains a per-stage chart of aggregate tok/s against TTFT p50,
and headless mode prints a load-sweep table.

```bash
# find the knee: throughput vs. concurrency
uv run tokie-benchy run -p M -n 16 -c 1,2,4,8,16,32

# open loop at realistic bursty traffic
uv run tokie-benchy run -p M -n 40 --request-rate 5 --arrival poisson

# rate sweep with fixed-length replies for comparable decode numbers
uv run tokie-benchy run -p M -n 30 --request-rate 2,4,8 --exact-output

# long-context harness turns at three parallelism levels
uv run tokie-benchy run -p H -n 6 -c 1,2,4 --prompt-tokens 32768 --max-tokens 512 --timeout 1800
```

Every run prints and stores a **repro command** (headless output, TUI config
panel, `repro` in the JSON), so a run configured through the dialog can be
repeated from a script.

`--exact-output` relies on `ignore_eos` and `min_tokens`, which vLLM, SGLang and
llama.cpp accept; hosted APIs typically reject them with HTTP 400.

Percentiles reported per measurement: p50, p90, p95, p99 plus mean, std, min, max
(the TUI table shows a subset; JSON and headless output have all of them).

## Thinking / reasoning budget

Reasoning models spend most of their output on hidden thinking, which dominates
TTFT-to-answer and total tokens. Providers spell the control differently, so
`--reasoning-effort` is translated per flavor:

| style | request field | granularity | used by |
|-------|---------------|-------------|---------|
| `openai` | `reasoning_effort: none/minimal/low/medium/high/xhigh` | levels | OpenAI o-series & GPT-5, gpt-oss on vLLM, Grok, LiteLLM, DeepSeek V3.2 |
| `openrouter` | `reasoning: {effort}` / `{enabled: false}` | levels | OpenRouter |
| `anthropic` | `thinking: {type, budget_tokens}` | budget 512 / 1k / 4k / 16k / 32k | Claude via proxies (needs `--max-tokens` > budget) |
| `deepseek` | `thinking: {type: enabled/disabled}` | toggle | DeepSeek V3.1+ |
| `gemini` | `google.thinking_config.thinking_budget` | budget | Gemini OpenAI-compat |
| `qwen` | `chat_template_kwargs: {enable_thinking}` | toggle | Qwen3 and any thinking chat template on vLLM / SGLang / llama.cpp |
| `ollama` | `think: false / low / medium / high / true` | toggle+levels | Ollama |

With the default `--reasoning-style auto`, the run starts by probing: each flavor
is sent twice with a tiny request (once at the requested effort, once toggled the
other way) and the responses are compared for reasoning tokens. Servers like vLLM
silently accept unknown fields, so acceptance alone proves nothing; a flavor
counts as **verified** only if it actually changed the output. Outcomes, shown in
the TUI config panel, the log, the headless output and the JSON:

- **verified** — the flavor demonstrably controls thinking; it is sent with every request.
- **accepted, effect unverified** — accepted but the opposite setting was rejected, so the effect could not be compared (typical for OpenAI models that refuse `none`). Sent anyway.
- **unsupported** — nothing changed the output. The run continues **without** any thinking parameter, the TUI shows a red status line and a warning toast, and the JSON records every probe result. Nothing fails.
- **explicit** — you picked a style; it is sent as-is with no probe. Rejections show up as failed requests.

```bash
uv run tokie-benchy run -p M                       # low effort, auto-detected (default)
uv run tokie-benchy run -p M -r off                # thinking disabled: measure pure answer speed
uv run tokie-benchy run -p M -r high --reasoning-style openai   # gpt-oss / OpenAI, no probe
uv run tokie-benchy run -p M --reasoning-style none             # do not touch thinking at all
uv run tokie-benchy run -p M --extra-body '{"chat_template_kwargs": {"enable_thinking": false}, "top_p": 0.8}'
```

Toggle-only flavors (`qwen`, `deepseek`) ignore the level: anything except `off`
means "on". The probe costs about a dozen tiny requests; use an explicit style or
`none` to skip it.

## Long-context runs (32k / 64k / 128k prompts)

To see how a model holds up when driven by a heavy agent harness (large system
prompts, long tool transcripts), push `--prompt-tokens` up and keep output short
so the run isolates prefill:

```bash
# single-stream prefill scaling: TTFT vs. prompt size
for n in 8192 32768 65536 131072; do
  uv run tokie-benchy run -p L -n 3 -c 1 --prompt-tokens $n --max-tokens 64 \
    --timeout 900 --no-tui -o results/prefill-$n.json
done

# does it still hold under parallel long-context load?
uv run tokie-benchy run -p H -n 8 -c 4 --prompt-tokens 65536 --max-tokens 256 --timeout 1800

# long context AND long generation (harness-style turn)
uv run tokie-benchy run -p M -n 4 -c 2 --prompt-tokens 32768 --max-tokens 2048 --timeout 1800
```

Things to know at these sizes:

- **Raise `--timeout`.** A 128k prefill alone can take minutes on a busy GPU; the
  default 120s will report the request as failed.
- **Start with `-c 1`.** KV-cache memory is the usual limit. If the server cannot
  fit `concurrency x prompt_tokens` it will queue, evict, or return HTTP 4xx/5xx;
  failures show in the log with the server's message.
- **Check the model's context window.** A prompt larger than the window returns
  an HTTP 400 from most servers, which the tool reports as a failed request.
- **Prompts are synthetic filler, not real transcripts.** They measure serving
  capacity (prefill/decode speed, KV pressure, scheduling), not answer quality
  at long context.
- **Body size is fine.** 131072 tokens is roughly 700 KB of JSON; generation
  takes a few ms.
- **Use `--seed`** to change the filler text between runs if the server has
  prefix caching and you want cold prefill every time.

Example from this repo's test endpoint: a single 32k-token prompt reported
32,805 prompt tokens server-side and a TTFT of about 16s at concurrency 1.
