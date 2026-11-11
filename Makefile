PORT ?= 8884

.PHONY: install test lint format train evaluate gate run docker-build docker-run screenshot clean

install:
	uv sync --python 3.12

test:
	uv run pytest -q

lint:
	uvx ruff check .
	uvx ruff format --check .

format:
	uvx ruff format .
	uvx ruff check . --fix

train:  ## 400 vans, three years, every policy replayed: about 20 seconds, writes data/bundle.pkl
	uv run fleetmaint train

evaluate:
	uv run fleetmaint evaluate

gate:  ## what CI runs: exits 1 unless the model-based policy costs less than fixed intervals
	uv run fleetmaint evaluate --gate

run:
	uv run fleetmaint serve --port $(PORT)

docker-build:
	docker build -t fleet-maint .

docker-run:
	docker run --rm -p $(PORT):8884 fleet-maint

screenshot:  ## needs the server running on $(PORT) and Google Chrome
	uv run python scripts/screenshot.py --port $(PORT)

clean:
	rm -rf .pytest_cache .ruff_cache data
