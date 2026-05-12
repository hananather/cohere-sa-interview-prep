.PHONY: install test test-offline-full test-live index index-force registry registry-live run run-adk cli smoke-live verify verify-live

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest defence_agent/tests/

test-offline-full:
	python -m pytest defence_agent/tests/ --run-slow-offline --durations=10

test-live:
	python -m pytest defence_agent/tests/ --run-live --run-slow-offline -m live --durations=10

index:
	python defence_agent/scripts/build_chroma_index.py

index-force:
	python defence_agent/scripts/build_chroma_index.py --force

registry-live:
	python defence_agent/scripts/run_demo_query_registry.py --no-fail

registry: registry-live

run:
	streamlit run streamlit_app.py --server.port 8501

run-adk:
	./defence_agent/scripts/run_adk_agent.sh

cli:
	python defence_agent/scripts/run_agent_session.py \
		"What are the DND CAF AI Strategy lines of effort?" \
		--persona clearance_unclassified

smoke-live:
	python defence_agent/scripts/smoke_cohere_chroma_live.py

verify: test-offline-full

verify-live: index test-live registry-live
