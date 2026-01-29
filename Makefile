.PHONY: install install-dev lint format test

install:
	python -m pip install -U pip
	pip install -e .

install-dev:
	python -m pip install -U pip
	pip install -e ".[dev,api,explain]"

lint:
	ruff check .

format:
	black .
	ruff check . --fix

test:
	pytest --maxfail=1