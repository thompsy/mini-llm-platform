# Concepts

Running notes on the ideas behind this project, to revisit later.

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
