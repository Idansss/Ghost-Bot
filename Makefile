.PHONY: run test lint format typecheck ci migrate makemigration

PY ?= python

run:
	$(PY) -m app.main

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

typecheck:
	$(PY) -m mypy

ci: lint typecheck test

migrate:
	$(PY) -m alembic upgrade head

makemigration:
	$(PY) -m alembic revision --autogenerate -m "migration"
