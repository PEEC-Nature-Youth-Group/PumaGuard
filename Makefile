.PHONY: venv
venv:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip

.PHONY: docs
docs: venv
	. venv/bin/activate && pip install --requirement docs/source/requirements.txt
	. venv/bin/activate && make -C docs html

.PHONY: tests
tests:
	poetry run pytest

.PHONY: install
install:
	poetry install

.PHONY: lint
lint: install
	poetry run pylint --verbose --rcfile=pylintrc pumaguard tests scripts
