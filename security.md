# Security — access control design notes

Exploratory notes on what authentication, authorization (RBAC/ABAC), and the
adjacent security concerns could look like for this system. Nothing here is
implemented yet — the API is currently open — it's a design map.

> See also [`concepts.md`](concepts.md) and [`caching.md`](caching.md); several
> concerns overlap (multi-tenancy must scope the vector store, trace store, *and*
> any cache).

## Starting point: there is no auth today

Every endpoint (`/chat`, `/rag`, `/agent`, `/traces`) is wide open. So this is
greenfield. Two things worth separating up front:

- **Authentication (authn)** — *who are you?* API key, JWT, or SSO/OIDC. RBAC
  assumes this already happened. Enterprise deployments push toward SSO and
  phishing-resistant auth.
- **Authorization (authz)** — *what may you do?* RBAC answers this via
  `subject → role → permissions → (action, resource)`.

You need a thin authn layer first (even just API key → identity), then authz on
top.

## What is actually worth protecting here (the LLM twist)

Generic web RBAC protects rows in a database. Here the resources are stranger,
which is where it gets interesting:

- **Capabilities / endpoints** — `/agent` (tool use) is more powerful/risky than
  `/chat`; gate it to a higher role.
- **Models** — *which* model a role may call (cheap vs expensive; general vs a
  specialized/permissive model). This is exactly the "trusted access" tier idea:
  verified users reach a more capable model, others don't.
- **Tools** (agent) — which tools a role's agent may invoke.
- **Documents / corpora** — what the RAG step may *retrieve and surface*.
- **Traces** — they hold full prompts + answers + token counts; reading them is a
  privileged, log-like action.
- **Admin actions** — `ingest`, config changes, running evals.
- **Budget / quota** — per-role token/request budgets (cost control as authz).

## A plausible role set

| Role | Can do |
| --- | --- |
| `user` | `/chat`, `/rag` over the public corpus, see **own** traces, small quota |
| `analyst` | + `/agent`, more tools, larger quota, more corpora |
| `auditor` | read **all** traces (observability) — but not run queries |
| `admin` | `ingest`, config, evals, all models/tools/corpora/traces |

## The three genuinely hard, LLM-specific problems

### 1. Document-level authorization in RAG (the subtle one)

Retrieval must filter to what *this subject* may see **before** chunks enter the
prompt — otherwise the model faithfully summarizes, and leaks, a document the
user can't access. `ChromaStore.query` is the enforcement point: attach an ACL to
each chunk's metadata at ingest (owner / allowed-roles / classification / tenant)
and pass a metadata filter on query, so vector search only ever returns permitted
chunks. Citations then inherit the filter for free. This is "RAG + permissions,"
and it's where authz meets the vector store.

### 2. Tool authorization for the agent (the confused-deputy risk)

The agent acts *on behalf of* a user, so a low-privilege user could try to talk it
into using a privileged tool. Authorization must be checked **at tool-execution
time in the runner**, against the *caller's* role — not merely by leaving the tool
out of the prompt (the model can be prompt-injected into "calling" a tool it was
never offered). `run_agent`'s `_run_tool` is exactly where an
`is_authorized(caller, tool)` allow-list check belongs.

### 3. Model-tier access

A permission of the form "may call model X" — the trusted-access pattern: role
determines which model (and effectively which safety posture) you reach. In code
it's a check before the chat call picks the model.

## The load-bearing principle: enforce in code, not in the prompt

You **cannot** implement access control by telling the model *"you're a basic
user, don't reveal restricted docs."* A prompt injection or a clever question
overrides that instantly — **the model is untrusted**. Authorization must be
**deterministic code wrapped around** the model:

- filter retrieval in the store (data authz),
- gate tools in the runner (capability authz),
- choose the model in the route (model authz),

all keyed on the authenticated identity, so the model only ever sees what the
subject is already allowed to see. This is why authz for LLM systems lives at the
*seams that already exist* (Protocol boundaries + FastAPI dependencies), not in
prompts. Related: treat retrieved documents and tool outputs as **untrusted input
too** — a poisoned document can carry an injection ("ignore your instructions and
…"); the same "model is untrusted, enforce around it" stance applies.

## Where it slots into the architecture

All additive, on existing seams:

- A `get_current_user` dependency (authn) + a `require(permission)` dependency
  (authz) on each route — mirrors the existing `Depends(get_client)` style.
- A metadata/ACL filter threaded into `VectorStore.query` (retrieval authz).
- A per-tool authorization check in the agent runner (tool authz).
- A model-selection check before the chat call (model authz).

## RBAC vs ABAC, and multi-tenancy

Pure RBAC (role → permissions) is coarse. Document access often really wants
**ABAC** (attribute-based): "user's clearance ≥ document classification," or
"user.tenant == document.tenant" — a role alone can't express "this user, these
documents." Realistic systems end up **RBAC for capabilities + ABAC for data**.
**Multi-tenancy** is just an attribute (`tenant`) that must scope the vector store,
the trace store, **and any cache**, so tenant A never retrieves, sees the traces
of, or gets a cached answer belonging to tenant B.

## Auditability — a tie-in to M3 tracing

Add `subject` + `role` to each `Trace`, and the trace store becomes an **audit
log**: who asked what, which tools ran, which model, which documents were cited.
Access control + observability together answer "who did what, and were they
allowed to?" — the accountability story an enterprise or government customer asks
for.

## Adjacent concerns (brief)

- **Secrets** — API keys / model credentials via env + a secrets manager, never in
  code or the trace store.
- **PII / data stewardship** — prompts, answers, and traces may contain sensitive
  data; mind retention, redaction, and per-tenant isolation of anything persisted
  (traces, caches).
- **Rate limiting / quotas** — per-subject budgets bound both cost and abuse.
- **Input validation** — already partly present via pydantic (`min_length`,
  bounds); size limits also blunt cost-amplification attacks.
- **Denial-of-wallet** — an open, model-backed endpoint is a cost-amplification
  target; authn + quotas are the first defense.

## If building one (later)

Smallest instructive slice, in order:

1. **Authn**: an API-key → identity dependency (`get_current_user`).
2. **Capability authz**: a `require(permission)` dependency gating `/agent`,
   `/traces`, and admin actions by role.
3. **Data authz**: chunk ACLs at ingest + a metadata filter in `VectorStore.query`
   — the RAG document-filtering problem, the subtlest and most valuable to build.
4. **Tool authz**: a per-role allow-list checked in `run_agent`.
5. **Audit**: `subject`/`role` on each trace.
