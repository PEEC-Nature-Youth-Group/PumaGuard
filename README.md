# PumaGuard

[![Build and Test Webpage](https://github.com/PEEC-Nature-Youth-Group/PumaGuard/actions/workflows/build-webpage.yaml/badge.svg)](https://github.com/PEEC-Nature-Youth-Group/PumaGuard/actions/workflows/build-webpage.yaml)

[![Test and package code](https://github.com/PEEC-Nature-Youth-Group/PumaGuard/actions/workflows/test-and-package.yaml/badge.svg)](https://github.com/PEEC-Nature-Youth-Group/PumaGuard/actions/workflows/test-and-package.yaml)

###  ☁ Open in the Cloud
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
[![Open in VS Code](https://img.shields.io/badge/Open%20in-VS%20Code-blue?logo=visualstudiocode)](https://vscode.dev/github/PEEC-Nature-Youth-Group/PumaGuard)
[![Open in Glitch](https://img.shields.io/badge/Open%20in-Glitch-blue?logo=glitch)](https://glitch.com/edit/#!/import/github/PEEC-Nature-Youth-Group/PumaGuard)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/PEEC-Nature-Youth-Group/PumaGuard)
[![Open in CodeSandbox](https://assets.codesandbox.io/github/button-edit-lime.svg)](https://codesandbox.io/s/github/PEEC-Nature-Youth-Group/PumaGuard)
[![Open in StackBlitz](https://developer.stackblitz.com/img/open_in_stackblitz.svg)](https://stackblitz.com/github/PEEC-Nature-Youth-Group/PumaGuard)
[![Open in Repl.it](https://replit.com/badge/github/withastro/astro)](https://replit.com/github/PEEC-Nature-Youth-Group/PumaGuard)

[![Open in Codeanywhere](https://codeanywhere.com/img/open-in-codeanywhere-btn.svg)](https://app.codeanywhere.com/#https://github.com/PEEC-Nature-Youth-Group/PumaGuard)
[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/PEEC-Nature-Youth-Group/PumaGuard)


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
cd PumaGuard
scripts/train.py --help
```

For example, if you want to train the model from row 1 in the notebook,

```console
scripts/train.py --notebook 1
```

## Running the server

The `pumaguard-server` watches a folder and classifies new files as they are
added to that folder. Run with

```console
poetry run pumaguard-server FOLDER
```

Where `FOLDER` is the folder to watch.

![Server Demo Session](docs/source/_static/server-demo.gif)

## Training new models

For reproducibility, training new models should be done via the train script
and all necessary data, i.e. images, and the resulting weights and history
should be committed to the repository.

1. Get a TPU instance on Colab
2. Open a terminal and run
    ```console
    $ git clone https://github.com/PEEC-Nature-Youth-Group/PumaGuard.git
    $ cd PumaGuard
    ```
3. Get help on how to use the script
    ```console
    $ ./scripts/pumaguard --help
    $ ./scripts/pumaguard train --help
    ```
4. Train the model from scratch
    ```console
    ./scripts/pumaguard train --no-load --settings models/model_settings_6_pre-trained_512_512.yaml
    ```
