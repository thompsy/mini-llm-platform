# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) is a technique that lets a language model
answer questions using an external document corpus rather than only its training
data. At query time, the system retrieves the most relevant chunks of text and
includes them in the prompt, grounding the model's answer in real sources.

RAG reduces hallucination because the model is instructed to answer only from
the retrieved passages. It also makes answers auditable: the model can cite the
specific sources it used, so a reader can verify each claim.
