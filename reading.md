# Further reading

Books on designing the kinds of systems this project is built from — LLM
applications, RAG, evaluation, observability, and (for the stretch goals)
models from scratch. Roughly ordered by how directly they map to this repo.

> Editions move fast in this space — check for the latest printing.

## Most directly on point

- **Chip Huyen — _AI Engineering: Building Applications with Foundation Models_
  (O'Reilly, 2025).** The closest match to this project: building _on top of_
  foundation models — RAG, evaluation (incl. LLM-as-judge → M4), prompting,
  inference optimization, cost/latency tradeoffs, overall app architecture.
  _Start here._
- **Chip Huyen — _Designing Machine Learning Systems_ (O'Reilly, 2022).** The
  "platform" companion: data, deployment, monitoring, and the feedback loops
  that make tracing (M3) and eval/regression (M4) a discipline rather than
  one-offs.

## For the "build your own models" stretch goal

- **Sebastian Raschka — _Build a Large Language Model (From Scratch)_
  (Manning, 2024).** A nanoGPT-scale decoder built up step by step — almost
  exactly the long-term goal of swapping home-grown models behind the
  `OllamaClient` / `OllamaEmbedder` interfaces.
- **Tunstall, von Werra & Wolf — _Natural Language Processing with Transformers_
  (O'Reilly).** The HuggingFace book; transformer internals and fine-tuning if
  you go past from-scratch.

## Practical — RAG & embeddings

- **Alammar & Grootendorst — _Hands-On Large Language Models_ (O'Reilly,
  2024).** Strong, visual treatment of embeddings, semantic search, and RAG —
  the conceptual underpinnings of M2.

## Systems / platform craft underneath it all

- **Martin Kleppmann — _Designing Data-Intensive Applications_.** Not
  LLM-specific, but the canonical book on storage, consistency, and data
  systems — relevant to the parts that aren't the model: the SQLite trace store
  (M3), the vector store, persistence, reliability. (A 2nd edition has been in
  the works — check status.)

---

**If you read one:** Huyen's _AI Engineering_ (covers RAG + eval + observability,
i.e. M2–M4). Then Raschka when starting the from-scratch models.
