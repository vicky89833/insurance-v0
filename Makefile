.PHONY: help setup eda train test serve all clean
.DEFAULT_GOAL := help

PYTHON := .venv/bin/python
PIP    := .venv/bin/pip

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the virtualenv and install dependencies
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

eda:  ## Run exploratory analysis (figures + artifacts/metrics/eda_summary.json)
	$(PYTHON) -m src.eda

train:  ## Train, tune, evaluate and persist the model
	$(PYTHON) -m src.train

test:  ## Run the test suite
	$(PYTHON) -m pytest tests/ -q

serve:  ## Start the prediction API on http://127.0.0.1:8000 (docs at /docs)
	.venv/bin/uvicorn src.api:app --reload --port 8000

all: eda train test  ## Full pipeline: analysis, training, tests

clean:  ## Remove generated artifacts and caches
	rm -rf artifacts/models/*.joblib artifacts/figures/*.png artifacts/metrics/*
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
