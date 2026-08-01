# Caching — design notes & a temperature experiment

Exploratory notes on where caching could live in this project, why, and how
`temperature` decides whether a cache is *transparent* or *behaviour-changing*.
Nothing here is implemented yet — it's a design map plus a runnable experiment.

> See also [`concepts.md`](concepts.md) (embeddings/cosine, training vs
> inference, agents) — semantic caching reuses the cosine machinery from there.

## Why cache at all

Three drivers, and one enabler that makes it *safe*:

- **Cost.** Locally, model calls cost compute + latency. Against a hosted API
  (e.g. OpenAI) they cost **money per token**, so removing identical calls is
  real savings — this is why hosted providers ship *prompt caching* as a feature.
- **Latency / UX.** Embedding and generation are the slow steps; a cache hit
  returns in ~ms. Compounds for the agent, where one question = several
  sequential LLM calls.
- **Rate limits / throughput.** Fewer backend calls = more headroom under load.

**The enabler:** much of the pipeline is *deterministic*. Embeddings are a pure
function of `(text, model)`. Generation is deterministic **at `temperature=0`** —
which the judge and the agent already use. Deterministic ⇒ exact-key cacheable
with no correctness cost.

## The call graph, annotated

```
/chat   → chat(messages, model, temp)                    # cacheable @ temp 0
/rag    → embed(question) → store.query() → chat(...)     # embed pure; query corpus-dependent; chat temp-dependent
/agent  → loop: chat(reason) → tool.run() → chat(...)     # reason temp 0; tools vary
eval    → answer_fn(q) + judge(temp 0)                    # judge very cacheable; re-runs repeat identical calls
ingest  → embed(chunk) per chunk                          # pure; re-ingest re-embeds unchanged docs
```

## Caching opportunities, layer by layer

1. **Embedding cache — highest value, lowest risk.** `embed(text, model)` is
   pure. Key = `hash(model + text)` → vector. Wins for repeated queries,
   re-ingesting an unchanged corpus (you otherwise re-embed thousands of chunks
   for nothing), and eval runs. Only a **model change** invalidates it (so the
   model id goes in the key). *Build this first.*

2. **Generation (chat) cache — big win at temp 0.** Key =
   `hash(model + temperature + seed + messages)` → response. Helps the judge
   (temp 0, hammered by evals), agent reasoning steps, and repeated identical
   `/chat`/`/rag` questions. Sound only at `temp=0` (see the temperature section).

3. **Retrieval cache — modest.** Key = `query_embedding → top-k hits`. HNSW is
   already fast, so lower value; and any `ingest` changes the corpus, so the key
   must carry a **corpus-generation stamp** or a short TTL.

4. **Full-RAG / answer cache — convenient but staleness-prone.** Key =
   `question → RagResult`. One hit skips embed+retrieve+generate, but it's
   invalidated by the *union* of everything beneath: corpus change, model change,
   **and a prompt edit** (`SYSTEM_PROMPT`). Key it on a prompt/version hash.

5. **Tool-result cache (agent) — the teaching case.** Cache
   `tool(name, input) → observation`. `rag_search` is cacheable (RAG rules
   apply); `calculator` is pure but cheap (caching is harmless, pointless);
   **`get_date` must never be cached** — it reads the clock. The lesson: cache
   only pure/deterministic operations; anything touching the clock, a DB, or
   external state is a correctness trap.

## Cross-cutting design choices

Every cache above forces the same four decisions:

- **Keying — exact vs semantic.** Exact = hash of inputs (simple, correct, but
  misses "who made Python?" vs "who created Python?"). Semantic — see below.
- **Scope / storage.** *In-process* (`dict`/`functools.lru_cache`: trivial,
  fastest, per-process, lost on restart) → *persistent* (SQLite/disk: survives
  restarts, shareable — natural fit, we already use SQLite for traces) →
  *distributed* (Redis: only if running multiple API instances).
- **Eviction.** Size-bounded LRU/LFU, or **TTL** for anything that can go stale.
- **Invalidation — the hard part.** Rule: **the key must include everything that
  affects the output** — model id, `temperature`, prompt version, corpus stamp.
  Get it wrong and you serve confidently-wrong stale answers.

## Semantic caching (reuses embeddings)

Exact-key caching hits only on *identical* inputs. A **semantic cache** embeds
the incoming query and matches it against cached queries by **cosine similarity**
above a threshold, returning the cached answer for a "close enough" query. Every
piece already exists here (embedder + cosine). The trade-off is a **correctness
knob**: too low a threshold and "how do I *delete* a user?" hits the entry for
"how do I *create* a user?" — a dangerously wrong answer. It trades exactness for
hit-rate, gated by a tunable threshold worth *measuring* rather than guessing.

## Temperature × caching (the crux)

**`temperature` adds variability; caching removes it.** Opposing forces — so
temperature decides whether an exact-key cache is transparent or behaviour-changing.

- **`temperature = 0` (greedy) → cache is transparent.** Same input → same output,
  so a cached response is exactly what the model would have produced anyway. This
  is why the judge and agent-reasoning calls are ideal cache targets.
- **`temperature > 0` (sampling) → cache = "sample once, reuse forever."** The
  same input maps to a *distribution*; caching freezes **one draw** and serves it
  to everyone. Not wrong (it's a valid sample) but no longer sampling. The higher
  the temperature, the more of the distribution you discard. Fine if you only
  wanted mild phrasing variety; **breaks** techniques that rely on variety
  (regenerate, self-consistency / majority-vote over N samples).

**Therefore the cache key must include `temperature` (and `seed`).** Otherwise a
`temp=0` and a `temp=1.0` request with the same messages collide.

**The escape hatch — seed.** Sampling is uncacheable because it's *unseeded*. Pin
a `seed` (Ollama supports one) and `(messages, model, temp>0, seed)` becomes
reproducible — cacheable *and* nonzero temperature.

**Honest caveat.** "temp 0 = perfectly deterministic" holds in principle and is
close enough within a stable deployment, but GPU/float non-determinism, batching,
quantization, and especially a **model version change** can shift outputs — one
more reason the model id belongs in every cache key.

Where each temperature lives today:

| Call | temperature | cache implication |
| --- | --- | --- |
| `judge_score` | 0 | fully cacheable, transparent — best target |
| agent reasoning (`run_agent`) | 0 | cacheable; also aids repro/debugging |
| `/chat` | `default_temperature` (0.7) or request override | freezes a sample; safe only at 0 |
| `/rag` grounding, `rag_search` | 0.7 | same — caching changes behaviour |

## The experiment

Two runnable experiments that make the temperature ↔ caching relationship
visible, using only config / request params (no code changes). Both need Ollama
running and an ingested corpus (`make ingest`).

### A — temperature's effect on determinism

Start the API (`make run` in another terminal), then send the *same* question a
few times at each temperature:

```bash
# temperature 0 → identical output every time (⇒ exact-key cacheable, transparent)
for i in 1 2 3; do
  curl -s http://127.0.0.1:8000/chat \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"In one sentence, who created Python?"}],"temperature":0}' \
    | jq -r .content
done

# temperature 1.0 → varies each time (⇒ caching would freeze one sample)
for i in 1 2 3; do
  curl -s http://127.0.0.1:8000/chat \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"In one sentence, who created Python?"}],"temperature":1.0}' \
    | jq -r .content
done
```

**What to observe:** the temp-0 runs are identical (a cache would return exactly
this); the temp-1.0 runs differ (a cache would pick one and serve it forever).

### B — temperature's effect on eval stability

The judge is already temp 0 (stable grading), but the *answers being judged* are
generated at `default_temperature`. Lowering it makes scores more reproducible
**and** the answers cacheable — at the cost of more repetitive phrasing.

```bash
# deterministic answers: run the golden set twice at temp 0, compare
APP_DEFAULT_TEMPERATURE=0 make eval OUTPUT=eval_t0_run1.json
APP_DEFAULT_TEMPERATURE=0 make eval OUTPUT=eval_t0_run2.json

# sampled answers: run twice at temp 0.7, compare
APP_DEFAULT_TEMPERATURE=0.7 make eval OUTPUT=eval_t07_run1.json
APP_DEFAULT_TEMPERATURE=0.7 make eval OUTPUT=eval_t07_run2.json

# diff the aggregate scores between the two runs at each temperature
jq .aggregates eval_t0_run1.json  eval_t0_run2.json
jq .aggregates eval_t07_run1.json eval_t07_run2.json
```

**What to observe:** run-to-run aggregate scores should be more stable at temp 0
than at 0.7. That stability is exactly what makes temp-0 generations safe to
serve from an exact-key cache — and it's the reproducibility-vs-naturalness
trade-off in a single measurement. (Some residual variance can remain even at
temp 0 — see the honest caveat above.)

## What the project already caches

- `get_settings()` uses `@lru_cache` — the `.env` is read once.
- FastAPI caches each dependency *within* a request.
- The long-lived `httpx.AsyncClient` reuses connections (pooling — not caching,
  but the same "don't redo expensive setup" instinct).

## Tie-ins to what's built

- **Measure with M3 tracing.** Add `cache_hit: bool` to the embed/chat spans and
  read hit-rate + latency delta from `make traces`. Caching without measuring
  hit-rate is flying blind.
- **Verify with M4 evals.** A cache should be *transparent*: same answers, faster.
  Run the golden set cache-warm vs cache-cold; scores must be identical — a real
  correctness test for the cache.

## If building one (later)

**Embedding cache first** — highest value, safest (pure function), improves
ingest + RAG + eval at once. A small `Protocol`-fronted cache (in-memory LRU now,
SQLite-backed later — mirroring the `VectorStore` / `TraceStore` seams) wrapping
`OllamaEmbedder`, with the model id in the key and a `cache_hit` span. Then a
**temp-0 generation cache** for the judge/agent. The **semantic cache** is the
most instructive to build — it forces reasoning about the threshold trade-off.
