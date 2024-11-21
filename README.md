# PumaGuard

[![Build and Test Webpage](https://github.com/PEEC-Nature-Youth-Group/PumaGuard/actions/workflows/build-webpage.yaml/badge.svg)](https://github.com/PEEC-Nature-Youth-Group/PumaGuard/actions/workflows/build-webpage.yaml)

[![Test and package code](https://github.com/PEEC-Nature-Youth-Group/PumaGuard/actions/workflows/test-and-package.yaml/badge.svg)](https://github.com/PEEC-Nature-Youth-Group/PumaGuard/actions/workflows/test-and-package.yaml)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/PEEC-Nature-Youth-Group/PumaGuard)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

## Introduction

Please visit <http://pumaguard.rtfd.io/> for more information.

## GitHub Codespaces

If you do not want to install any new software on your computer you can use
GitHub Codespaces, which provide a development environment in your browser.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/PEEC-Nature-Youth-Group/PumaGuard/)

## Local Development Environment

A local development environment can be created by using the `poetry` tool,
which can be installed with

```console
sudo apt install python3-poetry
```

Run

```console
poetry install
```

To install all of the necessary Python modules.

## Running the scripts on colab.research.google.com

[Google Colab](https://colab.research.google.com/) offers runtimes with GPUs
and TPUs, which make training a model much faster. In order to run the
[training script](scripts/train.py) in [Google
Colab](https://colab.research.google.com/), do the following from the terminal:

```console
git clone https://github.com/PEEC-Nature-Youth-Group/PumaGuard.git
scripts/train.py
```
