# Mixed Corpus Sources

Retrieved: 2026-05-10

These files are vendored for a local retrieval demo. Public documents remain owned by their source organizations. Inclusion does not imply endorsement by NATO, the Government of Canada, the Department of National Defence, the Canadian Armed Forces, the UK Ministry of Defence, or the Military Aviation Authority.

## Public Official PDFs

| doc_id | Local file | Language | Source | Canonical URL | Direct PDF URL |
|---|---|---|---|---|---|
| `CA-DEF-POL-2024-EN` | `public/our_north_strong_free_2024_en.pdf` | EN | Government of Canada, Department of National Defence | https://www.canada.ca/en/department-national-defence/corporate/reports-publications/north-strong-free-2024.html | https://www.canada.ca/content/dam/dnd-mdn/documents/corporate/reports-publications/2024/north-strong-free-2024-v2.pdf |
| `CA-DEF-POL-2024-FR` | `public/notre_nord_fort_libre_2024_fr.pdf` | FR | Government of Canada, Department of National Defence | https://www.canada.ca/fr/ministere-defense-nationale/organisation/rapports-publications/nord-fort-libre-2024.html | https://www.canada.ca/content/dam/dnd-mdn/documents/corporate/reports-publications/2024/nord-fort-libre-2024-v2.pdf |
| `CA-AI-STRAT-2024-EN` | `public/dnd_caf_ai_strategy_2024_en.pdf` | EN | Government of Canada, Department of National Defence | https://publications.gc.ca/site/eng/9.914595/publication.html | https://www.canada.ca/content/dam/dnd-mdn/documents/reports/ai-ia/dndcaf-ai-strategy.pdf |
| `CA-AI-STRAT-2024-FR` | `public/dnd_caf_ai_strategy_2024_fr.pdf` | FR | Government of Canada, Department of National Defence | https://publications.gc.ca/site/eng/9.914595/publication.html | https://www.canada.ca/content/dam/dnd-mdn/documents/reports/ai-ia/mdnfac-strategie-ia.pdf |
| `NATO-STRAT-CONCEPT-2022-EN` | `public/nato_strategic_concept_2022_en.pdf` | EN | NATO | https://www.nato.int/fr/about-us/official-texts-and-resources/official-texts/2023/03/03/nato-2022-strategic-concept | https://www.nato.int/content/dam/nato/webready/documents/publications-and-reports/strategic-concepts/2022/290622-strategic-concept.pdf |
| `NATO-STRAT-CONCEPT-2022-FR` | `public/nato_strategic_concept_2022_fr.pdf` | FR | NATO | https://www.nato.int/fr/about-us/official-texts-and-resources/official-texts/2023/03/03/nato-2022-strategic-concept | https://www.nato.int/content/dam/nato/webready/documents/publications-and-reports/strategic-concepts/2022/290622-strategic-concept-fr.pdf |

## Official DOCX-Origin Normalized PDF

The prototype does not parse `.docx` files natively. This entry proves the
normalization boundary: an official MS Word source is represented through the
publisher's official open-format PDF pair, then indexed through the same
rendered-page Embed v4 pipeline as PDF-origin documents.

| doc_id | Local file | Source format | Indexed format | Source | Canonical URL | Word URL | PDF URL |
|---|---|---|---|---|---|---|---|
| `UK-MOD-ASOEM-2023-EN` | `public/uk_mod_asoem_issue_2.pdf` | DOCX | PDF | UK Ministry of Defence and Military Aviation Authority | https://www.gov.uk/government/publications/aviation-safe-operating-environment-manual-asoem | https://assets.publishing.service.gov.uk/media/656da8951104cf0013fa740b/ASOEM_Issue_2.docx | https://assets.publishing.service.gov.uk/media/656da8b20f12ef070e3e0144/ASOEM_Issue_2.pdf |

## Synthetic Restricted PDFs

The synthetic files are fictional demo data created in this repository. They use `secret` and `top_secret` access labels only to exercise permission-aware retrieval. The disclaimer lives here and in `synthetic/README.md` so indexed PDF pages stay focused on retrieval evidence.

| doc_id | Local file | Language | Access level | Demo purpose |
|---|---|---|---|---|
| `SYN-FUSION-S-RELEASE-001` | `synthetic/fusion_model_release_control.pdf` | EN | `secret` | ACL contrast and authorized citation demo |
| `SYN-FUSION-TS-ANNEX-002` | `synthetic/fusion_model_restricted_routing_annex.pdf` | EN | `top_secret` | Higher-tier ACL contrast demo |
