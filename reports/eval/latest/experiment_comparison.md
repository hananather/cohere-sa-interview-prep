# Experiment Comparison

This report compares retrieval variants on expected-source hit metrics. Fixture mode is a component comparison, not a live Cohere quality claim.

- keyword_only: recall@k=0.755, precision@k=0.280, MRR=0.667, hit_rate=68.2%
- embedding_only: recall@k=0.550, precision@k=0.248, MRR=0.430, hit_rate=50.0%
- hybrid_no_rerank: recall@k=0.709, precision@k=0.267, MRR=0.648, hit_rate=63.6%
- hybrid_plus_rerank_fast: recall@k=0.709, precision@k=0.274, MRR=0.573, hit_rate=63.6%
- hybrid_plus_rerank_pro: recall@k=0.709, precision@k=0.274, MRR=0.573, hit_rate=63.6%
- query_expansion_plus_hybrid_rerank: recall@k=0.727, precision@k=0.268, MRR=0.573, hit_rate=63.6%
- agentic_rag_tools: recall@k=0.709, precision@k=0.274, MRR=0.573, hit_rate=63.6%
- structured_analysis_tools: recall@k=0.709, precision@k=0.274, MRR=0.573, hit_rate=63.6%
