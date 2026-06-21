# Mini LLM Platform

The Mini LLM Platform is a small, self-hosted learning project for building
hands-on intuition about large language models. It exposes a typed FastAPI
service that wraps a local model served by Ollama.

The project is organised into milestones: M1 is a typed inference API, M2 adds
retrieval-augmented generation, M3 adds tracing of LLM and retrieval calls, and
M4 adds an evaluation harness. The tech stack includes Python, FastAPI, Ollama,
and Chroma for vector storage.
