.PHONY: run test migrate

run:
	python -m app.main

test:
	pytest -q

migrate:
	alembic upgrade head
