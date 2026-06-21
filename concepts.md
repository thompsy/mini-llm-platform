# Concepts

Running notes on the ideas behind this project, to revisit later.

## Terminology

Brief definitions of recurring terms.

**Token.** The unit a language model operates on — not a word, but a sub-word fragment (e.g. "embedding" might be `["embed", "ding"]`). Models read and write sequences of tokens; all limits (context window, cost, speed) are measured in tokens, not characters or words.

**Context window.** The maximum number of tokens a model can "see" at once — its working memory. Everything outside the window is invisible to it. Modern models range from ~8 k to ~200 k tokens. RAG is partly a workaround for this limit: retrieve the relevant slice of a large corpus rather than fitting it all in.

**LLM (Large Language Model).** A transformer-based neural network trained to predict the next token given a sequence of tokens. "Large" refers to parameter count (billions). Despite the name, the same mechanism handles summarisation, translation, code, and reasoning — next-token prediction is a surprisingly general objective.

**Prompt.** The input text (sequence of tokens) you hand to a model. Everything the model knows about your task comes from the prompt; prompt engineering is the craft of phrasing inputs to get useful outputs.

**Inference.** Running a trained model to produce an output — as opposed to *training*, which updates the weights. This project only does inference (via Ollama); training is the "build your own models" stretch goal.

**Embedding.** A fixed-length vector (list of floats) that represents a piece of text in a high-dimensional space where similar meanings sit close together. Produced by an *encoder* model (e.g. `nomic-embed-text`). See the _Embeddings & cosine similarity_ section below for the full picture.

**Vector store.** A database optimised for similarity search over embeddings. Given a query vector, it returns the *k* nearest stored vectors efficiently (using approximate nearest-neighbour algorithms like HNSW). This project uses Chroma.

**RAG (Retrieval-Augmented Generation).** A pattern that grounds a generative model's answer in retrieved documents: embed the query → find the most similar chunks in the vector store → inject those chunks into the prompt → ask the model to answer *only* from that context. Reduces hallucination and lets the model answer questions about documents it was never trained on.

**Chunking.** Splitting a document into smaller pieces before embedding, because embedding a whole document into one vector loses fine-grained detail. Chunk size and overlap are tunable; too small loses context, too large dilutes the signal.

**Agent.** A model in a loop that can take *actions* (call tools, query APIs, run code) and observe results before deciding what to do next, rather than producing a single response and stopping. The loop runs until the model decides it has enough to answer, or hits a step limit.

**Tool use / function calling.** A protocol where the model can emit a structured request to call a named function (e.g. `rag_search("cosine similarity")`) instead of plain text. The caller executes the function, returns the result, and the model continues. Requires the model to have been trained to produce and interpret these structured calls.

**ReAct.** A specific agentic prompting pattern (_Reason + Act_): the model alternates between a `Thought` (reasoning about what to do), an `Action` (tool call), and an `Observation` (tool result), cycling until it can produce a final answer. Simple to implement; surprisingly capable.

## Embeddings & cosine similarity

**The core bet (distributional hypothesis).** "You shall know a word by the
company it keeps" (Firth, 1957; Harris, 1954). Meaning is relational and
statistical: a word or passage means what it tends to co-occur with. Embeddings
make that bet numerical — text becomes a vector, and _similar contexts produce
similar vectors_. It is not the characters of the text that are encoded, but the
contexts the text appears in.

**Comparing by angle (cosine similarity).** We compare two vectors by the angle
between them, not their length:

    cos(A, B) = (A · B) / (|A| · |B|)

where `A · B` is the dot product (multiply matching components, then sum) and
`|A|` is the vector's length (`sqrt` of the sum of its squares). Cosine ignores
magnitude and measures _direction_, so a one-sentence chunk and a paragraph on
the same topic still score as similar — length does not pollute relevance. That
is why the vector store uses cosine space (`hnsw:space: cosine`).

**Tiny worked example.** Suppose our corpus has three short documents, and we
chunk each into one chunk:

    C1: "the cat sat"
    C2: "the dog sat"
    C3: "the stock rose"

We want embeddings for the words **cat**, **dog**, and **stock**. We build each
word's vector from the company it keeps, using two context features:

- feature 1: does the word share a chunk with "sat"?
- feature 2: does the word share a chunk with "rose"?

Reading those off the chunks above:

- "cat" is in C1 (with "sat") -> `[1, 0]`
- "dog" is in C2 (with "sat") -> `[1, 0]`
- "stock" is in C3 (with "rose") -> `[0, 1]`

| word  | shares chunk with "sat" | shares chunk with "rose" | vector   |
| ----- | ----------------------- | ------------------------ | -------- |
| cat   | yes (C1)                | no                       | `[1, 0]` |
| dog   | yes (C2)                | no                       | `[1, 0]` |
| stock | no                      | yes (C3)                 | `[0, 1]` |

Now compare by cosine (the angle between vectors):

- **cat vs dog:**

      dot   = (1×1) + (0×0) = 1
      |cat| = sqrt(1² + 0²) = 1 ;  |dog| = 1
      cos   = 1 / (1 × 1) = 1.0      -> 0°,  "very similar"

- **cat vs stock:**

      dot   = (1×0) + (0×1) = 0
      cos   = 0 / (1 × 1) = 0.0      -> 90°, "unrelated"

So cat ≈ dog falls out _purely_ because they appeared in similar chunks (both
with "sat"), even though the words share no letters and we never defined
"animal". cat vs stock is unrelated because they never shared a context. That is
the entire trick in miniature.

**In this project.** `ChromaStore.query` does exactly this arithmetic, but with
768-dimensional vectors _learned_ by `nomic-embed-text` (not hand-counted), and
it reports `score = 1 - cosine_distance` so that higher means more similar.

**Modern refinements over the toy.**

1. _Contextual, not static._ word2vec (2013) gave one fixed vector per word;
   transformers (BERT, 2018+) compute a vector per word _in context_, so "river
   bank" and "savings bank" get different vectors.
2. _Passage-level._ For RAG we embed whole chunks into one vector, not
   individual words — so the chunk is the unit of meaning.

**Brief history.** Distributional hypothesis (Harris/Firth, 1950s) -> Vector
Space Model + cosine for search (Salton, 1960s–70s) -> dense semantic vectors via
LSA (Deerwester/Dumais, 1990) -> word2vec with vector arithmetic
(`king − man + woman ≈ queen`; Mikolov, 2013) -> GloVe (2014) -> contextual
embeddings, BERT (2018+). A ~60-year relay; what improved was our ability to
_learn_ good vectors, not the core idea.

Follow-up questions:
- how do we decide what the features should be?
