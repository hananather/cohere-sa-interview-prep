# Live Readiness Summary

Last verified: 2026-05-11

Local transcript directory:

```text
defence_agent/data/transcripts/live_readiness_final_20260511_013937
```

The raw transcript directory is intentionally git-ignored.

## Index

- Collection: `defence_agent_pdf_pages_1536`
- Pages indexed: 207
- Embedding model: `embed-v4.0`
- Embedding dimension: 1536
- Index version: `cohere_embed_v4_page_index_v1`
- Second build behavior: skipped unchanged pages with `embedded_pages: 0`
- DOCX-origin proof: one official public source is recorded as
  `source_format: docx`, `normalized_format: pdf`, and
  `normalization_method: official_pdf_pair`

## Demo Cases

| Case | Result | Key Signal |
| --- | --- | --- |
| `flagship_planning_brief_modernization` | Pass | Retrieved `CA-AI-STRAT-2024-EN` and `CA-DEF-POL-2024-EN` with Cohere native citations. |
| `natural_multilingual_nato_core_tasks` | Pass | Retrieved English and French NATO Strategic Concept pages. |
| `acl_unclassified_sensor_fusion_release_rule` | Pass | Refused the unauthorized secret source before generation. |
| `acl_secret_sensor_fusion_release_rule` | Pass | Retrieved `SYN-FUSION-S-RELEASE-001` for the authorized secret persona. |
| `trace_follow_up_source_access` | Pass | Answered source/access follow-up from prior audit metadata. |

## DOCX-Origin Proof

- Source: `UK-MOD-ASOEM-2023-EN`, Aviation Safe Operating Environment Manual.
- Provenance: official MS Word source represented through the publisher's
  official PDF pair.
- Audit metadata confirmed: `source_format: docx`, `normalized_format: pdf`,
  and `normalization_method: official_pdf_pair`.

## Verification

```bash
python defence_agent/scripts/build_chroma_index.py --json
python -m pytest defence_agent/tests/
python defence_agent/scripts/run_demo_query_registry.py \
  --case-id flagship_planning_brief_modernization \
  --case-id acl_unclassified_sensor_fusion_release_rule \
  --case-id acl_secret_sensor_fusion_release_rule \
  --case-id trace_follow_up_source_access \
  --case-id natural_multilingual_nato_core_tasks \
  --output-dir defence_agent/data/transcripts/live_readiness_final_20260511_013937 \
  --delay-seconds 2
```
