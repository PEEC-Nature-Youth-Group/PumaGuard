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
	poetry run pytest --capture=no --verbose --cov=pumaguard --cov-report=term-missing

.PHONY: install
install:
	poetry install

.PHONY: build
build:
	poetry build

.PHONY: lint
lint: install pylint mypy bashate ansible-lint

.PHONY: pylint
pylint: install
	poetry run pylint --verbose --recursive=true --rcfile=pylintrc pumaguard tests scripts

.PHONY: mypy
mypy: install
	poetry run mypy --install-types --non-interactive pumaguard

.PHONY: bashate
bashate: install
	poetry run bashate -v scripts/*sh

.PHONY: lint-notebooks
lint-notebooks: install
	poetry run pynblint notebooks

.PHONY: ansible-lint
ansible-lint: install
	poetry run ansible-lint -v scripts

.PHONY: snap
snap:
	snapcraft

.PHONY: integration
integration: install
	poetry run pumaguard-classify --notebook 1 \
	  "data/stable/angle 1/Lion/SYFW1930.JPG" \
	  2>&1 | tee integration-test.output
	if [ "$$(awk '/^Predicted/ { print $$2 }' integration-test.output)" != '87.73%' ]; then false; fi

.PHONY: prepare-trailcam prepare-output prepare-central
prepare-central prepare-trailcam prepare-output: prepare-%:
	scripts/launch-pi-zero.sh --name $* --force
	multipass transfer pumaguard_$(shell git describe --tags)*.snap $*:/home/ubuntu
	multipass exec $* -- sudo snap install --dangerous --devmode $(shell ls pumaguard*snap)

.PHONY: release
release:
	export NEW_RELEASE=$(shell git tag | sort | tail -n1 | awk -F v '{print $$2 + 1}') && \
	  git tag -a -m "Release v$${NEW_RELEASE}" v$${NEW_RELEASE}

.PHONY: configure-pi-zero
configure-pi-zero:
	ansible-playbook --inventory pi-zero, --ask-become-pass scripts/configure-pi.yaml
