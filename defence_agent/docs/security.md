# Security

## Controls

- Demo users are fixed personas with clearance and tenant metadata.
- The policy engine checks tool access and page access.
- Chroma applies `access_level`, `status`, and `language` filters before returning evidence.
- Retrieval runs a second policy check before evidence reaches the model.
- Retrieval also performs an answerability check before generation. This is not
  an access-control boundary, but it prevents weak authorized evidence from
  becoming an unsupported answer.
- Excluded source text is never returned from `search_documents`.
- `answer_audit` records excluded source IDs and denial reasons, not excluded
  source text.
- Tool-output caching is session scoped and keyed by persona, allowed access,
  filters, corpus version, and Cohere model settings.
- Cached replay strips excluded-source text before storage and never shares
  results across sessions.
- Synthetic `secret` and `top_secret` labels are illustrative demo tiers only.
- `.env` is ignored. Store local secrets there and do not commit them.

## Real Cohere API Key

- This is still a prototype, not a production-ready system.
- Store the real Cohere key only in the repo-root `.env` as `COHERE_API_KEY`.
- Prefer a Cohere trial key for rehearsal when its limits are sufficient. The
  point is real model behavior, not production account status.
- Do not commit, paste, screenshot, or copy the key into docs, prompts, traces,
  logs, tickets, or chat.
- The active prototype uses real Cohere model calls for embeddings, reranking,
  and final cited answers.
- Use `COHERE_REQUESTS_PER_MINUTE=20` or lower to pace live calls unless the
  account limit is explicitly higher.
- Keep `COHERE_EMBED_PAGE_BATCH_SIZE=1` for normal builds. It avoids large
  multimodal requests and checkpoints each embedded page immediately.
- Treat every run as billable external API usage. Keep prompts scoped to the
  demo corpus and avoid large ad hoc batch runs.
- Rotate the key immediately if it appears in git history, shared logs, or any
  external system.

## Safety Boundary

The main safety and security boundary is authorization before generation.

- The model only receives authorized source pages.
- The model receives no source pages when retrieval marks the turn as
  insufficiently supported.
- Cohere safety settings are not a substitute for retrieval authorization.
- If asked about safety modes, state that this prototype relies on the default
  Cohere behavior for the RAG call and treats policy filtering as the control
  that prevents restricted evidence from entering model context.

## Demo Personas

| Persona id | Clearance | Can retrieve |
|---|---|---|
| `clearance_unclassified` | `unclassified` | Public normalized sources only |
| `clearance_secret` | `secret` | Public normalized sources and synthetic secret PDFs |
| `clearance_top_secret` | `top_secret` | Public normalized sources, synthetic secret PDFs, and synthetic top-secret PDFs |

## Future Production Mapping

- Replace fixed personas with identity provider claims from OIDC, SAML, or another customer identity system.
- Map groups and clearances into the same retrieval filter contract.
- Keep access control outside the prompt.
- Export ADK traces and application logs to the customer observability stack.
