# PumaGuard

[![Build and Test Webpage](https://github.com/PEEC-Nature-Youth-Group/PumaGuard/actions/workflows/build-webpage.yaml/badge.svg)](https://github.com/PEEC-Nature-Youth-Group/PumaGuard/actions/workflows/build-webpage.yaml)

[![Test and package code](https://github.com/PEEC-Nature-Youth-Group/PumaGuard/actions/workflows/test-and-package.yaml/badge.svg)](https://github.com/PEEC-Nature-Youth-Group/PumaGuard/actions/workflows/test-and-package.yaml)

## Introduction

Please visit <http://pumaguard.rtfd.io/> for more information.

## Hacktoberfest 2024

This repository is participating in [Hacktoberfest
2024](https://hacktoberfest.com/). If you are interested in participating,
remember to sign up first!

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

## Using GitHub Codespaces

An alternative for developing is to use GitHub Codespaces.

## Running the scripts on colab.research.google.com

Colab offers runtimes with GPUs and TPUs, which make training a model much
faster. In order to run the [training script](scripts/train.py) on colab, do
the following from the terminal:

```console
git clone https://github.com/PEEC-Nature-Youth-Group/PumaGuard.git
python3 scripts/train.py
```
