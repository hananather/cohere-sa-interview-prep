.PHONY: install test eval eval-validate eval-advanced eval-compare eval-demo run smoke verify compose

install:
	python -m pip install -e ".[dev]"

test:
	USE_MOCK_COHERE=true python -m pytest

eval:
	USE_MOCK_COHERE=true python -c "from defence_agent.evals.runner import eval_runner; print(eval_runner.run().metrics)"

eval-validate:
	PYTHONPATH=defence_agent/backend USE_MOCK_COHERE=true python defence_agent/scripts/run_eval_harness.py validate

eval-advanced:
	PYTHONPATH=defence_agent/backend USE_MOCK_COHERE=true python defence_agent/scripts/run_eval_harness.py run --suite canonical --mode fixture

eval-compare:
	PYTHONPATH=defence_agent/backend USE_MOCK_COHERE=true python defence_agent/scripts/run_eval_harness.py compare --suite canonical --limit 24

eval-demo:
	PYTHONPATH=defence_agent/backend USE_MOCK_COHERE=true python defence_agent/scripts/run_eval_harness.py select-demo --mode fixture --runs 3

run:
	defence_agent/scripts/run_demo.sh

smoke:
	python defence_agent/scripts/smoke_demo.py

verify: test eval eval-validate eval-advanced eval-compare eval-demo
	@echo "Run 'defence_agent/scripts/run_demo.sh' in another terminal, then 'make smoke' for live HTTP checks."

compose:
	docker compose up --build
