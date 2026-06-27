# Learning roadmap: roles, skills, and what to study

A map from **development roles** (who you might want to be) to the **skills**
each needs and **what to study** for them. Pairs with [`reading.md`](reading.md)
(the books) and [`concepts.md`](concepts.md) (the ideas). This project itself is
a training ground — see "Where this project fits" at the end.

## The skill axes

Roles aren't a single ladder; they're a **mix of these independent areas**:

1. **Prompting / context engineering** — phrasing, structured output, context windows.
2. **LLM app patterns** — RAG, agents, tool use, evals, guardrails.
3. **Software & systems engineering** — APIs, services, observability, scaling, reliability.
4. **LLMOps / MLOps** — deployment, monitoring, cost/latency, CI for models.
5. **ML fundamentals** — neural nets, deep learning, training dynamics, fine-tuning.
6. **Research depth** — architectures, the math, reading/writing papers, novel methods.
7. **Product / domain / communication** — translating business problems, working with stakeholders/customers.

Going from a *user* toward a *researcher* adds **ML depth**; moving toward
enterprise/ops/product roles adds **engineering & communication breadth**.

## Roles → skills

Intensity: `.` low · `o` some · `+` solid · `#` deep

| Skill area                 | User | AI eng (startup) | AI eng (enterprise) | FDE (frontier lab) | ML/Applied eng | Researcher |
| -------------------------- | ---- | ---------------- | ------------------- | ------------------ | -------------- | ---------- |
| Prompting / context        | +    | #                | #                   | #                  | +              | +          |
| LLM app patterns           | o    | #                | +                   | #                  | +              | o          |
| Software & systems eng     | .    | +                | #                   | #                  | +              | o          |
| LLMOps / MLOps             | .    | o                | #                   | +                  | #              | o          |
| ML fundamentals            | .    | o                | o                   | +                  | #              | #          |
| Research depth             | .    | .                | .                   | o                  | +              | #          |
| Product / domain / comms   | o    | +                | +                   | #                  | o              | o          |

## Role by role — what to study

**Power user.** Prompting, model capabilities & limits, context windows, when
RAG/tools help (conceptually). No code or ML theory. Highest leverage-to-effort
ratio; most people stop too early.
→ _Study:_ Hands-On LLMs (concepts); play with the models directly.

**AI engineer — startup.** Full-stack on top of foundation-model APIs: ship
RAG/agents/evals fast, own the whole app, prompt well, glue services. Breadth and
speed over depth; you rarely train models.
→ _Study:_ Huyen, _AI Engineering_. **This is what M1–M4 of this project build.**

**AI engineer — enterprise.** Same app core, weight shifts to systems, LLMOps,
governance, integration, reliability, cost, compliance. Less greenfield, more
"robust and auditable."
→ _Study:_ Huyen, _AI Engineering_ + _Designing ML Systems_; Kleppmann, _DDIA_.
Maps onto this project's M3 (observability) and platform mindset.

**Forward Deployed Engineer (FDE), frontier lab.** A hybrid: take the lab's most
capable models and build bespoke production solutions **embedded with a
customer**. Top-tier software eng + deep applied-LLM (prompting, agents, evals,
pushing models to their limits) + strong communication/consulting. Needs deep
intuition for model capabilities and failure modes, but generally does **not
pretrain models** — the bridge between research and the real world. The most
"T-shaped" role (broad + a deep applied spike), which is why it's hard to hire.
→ _Study:_ Huyen, _AI Engineering_ (cover to cover) + ship the FDE capstone
below + practice the communication artifacts. ML fundamentals help you reason
about limits; research depth is optional.

**ML / Applied engineer.** Crosses into *changing* the model: fine-tuning,
distillation, eval infra, data pipelines, training/serving at scale. ML
fundamentals mandatory.
→ _Study:_ NN/DL foundations (Nielsen, Karpathy) then Raschka, _Build an LLM
from Scratch_. This project's stretch goals (own models, distillation) live here.

**Researcher / research engineer.** Architectures, objectives, the math, novel
methods, papers. Deepest ML + research depth; software is a means to an end.
→ _Study:_ Prince, _Understanding Deep Learning_; Goodfellow et al., _Deep
Learning_; primary papers.

## FDE capstone — a toy project covering most FDE skills

The FDE-defining elements aren't more theory; they're **(a)** a bespoke agentic
solution against a customer's messy systems, **(b)** hardening a prototype into
something reliable, and **(c)** the communication artifacts around it. So the
ideal project is a **simulated forward-deployed engagement**:

> _Embed with a fictional customer and ship an agentic AI solution end-to-end._

Pick a vertical with a concrete, slightly-ambiguous problem — e.g. a mid-size
SaaS **customer-support team drowning in tickets** (alternatives: law-firm
contract review, logistics ops copilot, internal data-analyst agent over a fake
warehouse). Build on this platform and deliver:

| FDE skill                       | Deliverable                                                                 |
| ------------------------------- | --------------------------------------------------------------------------- |
| Customer ambiguity → scope      | 1-page **problem framing**: vague ask → measurable goal + success metric    |
| Top-tier software eng           | Production-shaped service (the FastAPI bones already exist here)             |
| Deep applied LLM                | An **agent** with tool use: RAG over docs + structured tools (`lookup_order`, `search_kb`, `escalate`) |
| Integration with *their* systems| Stand up a **mock customer API** (fake CRM/ticketing/DB) the agent must call — the FDE-defining bit |
| Evals                           | Domain golden set + LLM-judge (M4) scoring resolution quality, not string match |
| Observability + cost            | Tracing/cost view (M3): latency and `$` per resolved ticket                  |
| Guardrails / failure modes      | PII handling, refusal on out-of-scope, hallucination checks, human fallback |
| Hardening                       | From "demo works" to handling the weird 20%: retries, timeouts, bad inputs  |
| Communication / handoff         | **Demo script**, **capability & limitations memo**, "how to operate this" doc |

The mock customer API, the written consulting artifacts, and the hardening pass
are what elevate this from a generic AI-engineering project into an FDE
simulation.

## Where this project fits

`mini-llm-platform` is a near-perfect **AI-engineer (startup/enterprise)**
trainer: M1 (API), M2 (RAG), M3 (tracing/observability), M4 (eval) cover the app
+ ops + eval core. The **stretch goals** (build-your-own models, distillation)
are the on-ramp to the **ML-engineer** column. To target **FDE**, layer the
capstone above on top — the remaining gap is mostly **prototyping speed +
communication/consulting**, not more theory.
