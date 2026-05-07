# Experiment Comparison

This report compares retrieval variants on expected-source hit metrics. Fixture mode is a component comparison, not a live Cohere quality claim.

- keyword_only: recall@k=1.000, precision@k=0.406, MRR=0.895, hit_rate=100.0%
- embedding_only: recall@k=0.886, precision@k=0.382, MRR=0.648, hit_rate=86.4%
- hybrid_no_rerank: recall@k=1.000, precision@k=0.433, MRR=0.865, hit_rate=100.0%
- hybrid_plus_rerank_fast: recall@k=0.955, precision@k=0.407, MRR=0.814, hit_rate=90.9%
- hybrid_plus_rerank_pro: recall@k=0.955, precision@k=0.407, MRR=0.814, hit_rate=90.9%
- query_expansion_plus_hybrid_rerank: recall@k=0.955, precision@k=0.398, MRR=0.814, hit_rate=90.9%
- agentic_rag_tools: recall@k=0.955, precision@k=0.407, MRR=0.814, hit_rate=90.9%
- structured_analysis_tools: recall@k=0.955, precision@k=0.407, MRR=0.814, hit_rate=90.9%
