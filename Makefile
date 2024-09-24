.PHONY: venv
venv:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip

.PHONY: docs
docs: venv
	. venv/bin/activate && pip install --requirement docs/source/requirements.txt
	. venv/bin/activate && make -C docs html

.PHONY: test
test:
	poetry run pytest

.PHONY: install
install:
	poetry install

.PHONY: lint
lint: install
	poetry run pylint --verbose --recursive=true --rcfile=pylintrc pumaguard tests scripts
	poetry run bashate -v scripts/*sh

.PHONY: snap
snap:
	snapcraft

.PHONY: functional
functional: prepare-trailcam prepare-output
	multipass info

.PHONY: prepare-trailcam prepare-output
prepare-trailcam prepare-output: prepare-%:
	scripts/launch-pi-zero.sh --name $*
	multipass transfer pumaguard_$(shell git describe --tags)*.snap $*:/home/ubuntu
	multipass exec $* -- sudo snap install --dangerous --devmode $(shell ls pumaguard*snap)
