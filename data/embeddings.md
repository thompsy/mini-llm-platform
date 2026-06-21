# Embeddings

An embedding is a fixed-length vector of numbers that represents the meaning of
a piece of text. Text that appears in similar contexts ends up with similar
vectors, following the idea that you can know a word by the company it keeps.

Similarity between two embeddings is usually measured with cosine similarity,
which compares the angle between the vectors rather than their length. A small
angle means the texts are semantically similar; a right angle means they are
unrelated. This is what makes search by meaning possible.
