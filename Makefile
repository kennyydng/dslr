VENV_DIR ?= venv
PY := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

TRAIN_DATA ?= datasets/dataset_train.csv
TEST_DATA ?= datasets/dataset_test.csv
WEIGHTS ?= weights.json
BATCH ?= 64

.PHONY: help venv install describe histogram scatter pairplot train predict \
	bonus-sgd bonus-minibatch bonus-compare bonus-run clean clean-venv

help:
	@echo "Usage: make <target>"
	@echo "Main:    venv install describe histogram scatter pairplot train predict"
	@echo "Bonus:   bonus-sgd bonus-minibatch bonus-compare bonus-run"
	@echo "Cleanup: clean clean-venv"
	@echo "Vars (optional overrides):"
	@echo "  TRAIN_DATA=... TEST_DATA=... WEIGHTS=... BATCH=... VENV_DIR=..."

venv:
	@test -d "$(VENV_DIR)" || python3 -m venv "$(VENV_DIR)"
	$(PIP) install --upgrade pip setuptools wheel

install: venv
	$(PIP) install -r requirements.txt

describe: install
	$(PY) src/describe.py $(TRAIN_DATA)

histogram: install
	$(PY) src/histogram.py $(TRAIN_DATA)

scatter: install
	$(PY) src/scatter_plot.py $(TRAIN_DATA)

pairplot: install
	$(PY) src/pair_plot.py $(TRAIN_DATA)

train: install
	$(PY) src/logreg_train.py $(TRAIN_DATA)

predict: install
	$(PY) src/logreg_predict.py $(TEST_DATA) $(WEIGHTS)

bonus-sgd: install
	$(PY) bonus/logreg_train_sgd.py $(TRAIN_DATA)

bonus-minibatch: install
	$(PY) bonus/logreg_train_minibatch.py $(TRAIN_DATA) $(BATCH)

bonus-compare: install
	$(PY) bonus/compare_methods.py

bonus-run: install
	bash bonus/run_all_bonus.sh

clean:
	rm -f houses.csv $(WEIGHTS) weights_sgd.json weights_minibatch.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-venv:
	rm -rf "$(VENV_DIR)"
