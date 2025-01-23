.PHONY: venv
venv:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	@echo "venv is initialized"

.PHONY: apidoc
apidoc: venv
	. venv/bin/activate && pip install --requirement docs/source/requirements.txt
	. venv/bin/activate && cd docs && sphinx-apidoc -o source --force ../pumaguard

.PHONY: docs
docs: venv
	@echo "building documentation webpage"
	. venv/bin/activate && pip install --requirement docs/source/requirements.txt
	. venv/bin/activate && cd docs && sphinx-apidoc --output-dir source --force ../pumaguard
	git ls-files --exclude-standard --others
	git ls-files --exclude-standard --others | wc -l | grep "^0" --quiet
	git diff
	git diff --shortstat | wc -l | grep "^0" --quiet
	. venv/bin/activate && sphinx-build --builder html --fail-on-warning docs/source docs/build
	. venv/bin/activate && sphinx-build --builder linkcheck --fail-on-warning docs/source docs/build

.PHONY: test
test:
	poetry run pytest --verbose --cov=pumaguard --cov-report=term-missing

.PHONY: install
install:
	poetry install

.PHONY: build
build:
	poetry build

.PHONY: lint
lint: install pylint isort mypy bashate ansible-lint

.PHONY: pylint
pylint: install
	poetry run pylint --verbose --recursive=true --rcfile=pylintrc pumaguard tests scripts

.PHONY: isort
isort:
	poetry run isort pumaguard tests scripts

.PHONY: mypy
mypy: install
	poetry run mypy --install-types --non-interactive --check-untyped-defs pumaguard

.PHONY: bashate
bashate: install
	poetry run bashate -v scripts/*sh pumaguard/completions/*sh

.PHONY: lint-notebooks
lint-notebooks: install
	poetry run pynblint notebooks

.PHONY: ansible-lint
ansible-lint: install
	poetry run ansible-lint -v scripts

.PHONY: snap
snap:
	snapcraft

FUNCTIONAL_FILES = \
    "data/stable/angle 1/Lion/SYFW2061.JPG" \
    "data/stable/angle 2/Lion/SYFW0270.JPG" \
    "data/stable/angle 2/Lion/SYFW0270_bright.JPG"

.PHONY: run-functional
run-functional:
	$(EXE) classify --settings models/model_settings_6_pre-trained_512_512.yaml $(FUNCTIONAL_FILES) | tee functional-test.output

.PHONY: check-functional
check-functional:
	if [ "$$(sed --quiet --regexp-extended '/^Predicted.*2061/s/^.*:\s*([0-9.%]+).*$$/\1/p' functional-test.output)" != '66.17%' ]; then \
		cat functional-test.output; \
		false; \
	fi
	if [ "$$(sed --quiet --regexp-extended '/^Predicted.*270.JPG/s/^.*:\s*([0-9.%]+).*$$/\1/p' functional-test.output)" != '32.22%' ]; then \
		cat functional-test.output; \
		false; \
	fi
	if [ "$$(sed --quiet --regexp-extended '/^Predicted.*270_bright.JPG/s/^.*:\s*([0-9.%]+).*$$/\1/p' functional-test.output)" != '91.83%' ]; then \
		cat functional-test.output; \
		false; \
	fi

.PHONY: functional-poetry
functional-poetry: install
	$(MAKE) EXE="poetry run pumaguard" run-functional
	$(MAKE) check-functional

.PHONY: functional-snap
functional-snap:
	$(MAKE) EXE="pumaguard" run-functional
	$(MAKE) check-functional

.PHONY: prepare-trailcam prepare-output prepare-central
prepare-central prepare-trailcam prepare-output: prepare-%:
	scripts/launch-pi-zero.sh --name $* --force
	multipass transfer pumaguard_$(shell git describe --tags)*.snap $*:/home/ubuntu
	multipass exec $* -- sudo snap install --dangerous --devmode $(shell ls pumaguard*snap)

.PHONY: release
release:
	export NEW_RELEASE=$(shell git tag | sed --expression 's/^v//' | \
	    sort --numeric-sort | tail --lines 1 | awk '{print $$1 + 1}') && \
	  git tag -a -m "Release v$${NEW_RELEASE}" v$${NEW_RELEASE}

.PHONY: configure-pi-zero
configure-pi-zero:
	ansible-playbook --inventory pi-zero, --ask-become-pass scripts/configure-pi.yaml

.PHONY: verify
verify:
	pumaguard verify --data-path data --notebook 6 | tee verify.output
	if [ "$$(awk '/accuracy/ {print $$3}' verify.output)" != 96.60% ]; then false; fi

.PHONY: train
train:
	pumaguard train --epochs 1 --model-output . --settings models/model_settings_9_light-3_512_512.yaml --data-path data --lions data/stable/angle\ 1/Lion --no-lions data/stable/angle\ 1/No\ lion/

.PHONY: pre-commit
pre-commit: lint docs test
