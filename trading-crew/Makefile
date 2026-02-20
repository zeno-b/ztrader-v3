.PHONY: test lint typecheck migrate up down

test:
	python3 -m pytest -q

lint:
	python3 -m ruff check .

typecheck:
	python3 -m mypy --strict agents training models config tools db backtesting

migrate:
	@echo "Run Alembic migrations from trading container."
	@echo "docker compose run --rm trading alembic upgrade head"

up:
	docker compose up -d

down:
	docker compose down
