.PHONY: venv
venv:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	@echo "venv is initialized"

.PHONY: docs
docs: venv
	@echo "building documentation webpage"
	. venv/bin/activate && pip install --requirement docs/source/requirements.txt
	. venv/bin/activate && cd docs && sphinx-apidoc -o source ../pumaguard
	git ls-files --exclude-standard --others
	git ls-files --exclude-standard --others | wc -l | grep "^0" --quiet
	git diff
	git diff --shortstat | wc -l | grep "^0" --quiet
	. venv/bin/activate && make -C docs html
	. venv/bin/activate && make -C docs linkcheck

.PHONY: test
test:
	poetry run pytest

.PHONY: install
install:
	poetry install

.PHONY: build
build:
	poetry build

.PHONY: lint
lint: install
	poetry run pylint --verbose --recursive=true --rcfile=pylintrc pumaguard tests scripts
	poetry run mypy pumaguard
	poetry run bashate -v scripts/*sh

.PHONY: lint-notebooks
lint-notebooks: install
	poetry run pynblint notebooks

.PHONY: snap
snap:
	snapcraft

.PHONY: integration
integration: prepare-trailcam prepare-output
	multipass info

.PHONY: prepare-trailcam prepare-output
prepare-trailcam prepare-output: prepare-%:
	scripts/launch-pi-zero.sh --name $*
	multipass transfer pumaguard_$(shell git describe --tags)*.snap $*:/home/ubuntu
	multipass exec $* -- sudo snap install --dangerous --devmode $(shell ls pumaguard*snap)

.PHONY: release
release:
	NEW_RELEASE=$(shell git tag | sort | tail -n1 | awk -F v '{print $$2 + 1}') $(shell git tag -a -m "Release v$${NEW_RELEASE}" v$${NEW_RELEASE})
