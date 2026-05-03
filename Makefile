.PHONY: install test eval run smoke compose

install:
	python -m pip install -e ".[dev]"

test:
	USE_MOCK_COHERE=true python -m pytest

eval:
	USE_MOCK_COHERE=true python -c "from defence_agent.evals.runner import eval_runner; print(eval_runner.run().metrics)"

run:
	defence_agent/scripts/run_demo.sh

smoke:
	python defence_agent/scripts/smoke_demo.py

compose:
	docker compose up --build
