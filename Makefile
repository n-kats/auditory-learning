.PHONY: lint format test v1-lint v1-format v1-test quick-test v2-test

v1-lint:
	uv run --project v1 ruff check v1/auditory_learning
	uv run --project v1 mypy v1/auditory_learning

v1-format:
	uv run --project v1 ruff format v1/auditory_learning
	uv run --project v1 ruff check --fix v1/auditory_learning

v1-test:
	uv run --project v1 pytest v1/tests

quick-test:
	uv run --project quick-auditory-learning/backend pytest tests/test_quick_auditory_learning.py

v2-test:
	uv run --project v2/backend pytest v2/backend/tests

lint: v1-lint
format: v1-format
test: v1-test quick-test v2-test
