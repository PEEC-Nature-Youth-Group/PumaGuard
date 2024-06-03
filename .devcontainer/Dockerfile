FROM ubuntu:jammy

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

WORKDIR /app

RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --yes apt-utils
RUN DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --yes \
    curl \
    git \
    pip \
    pipx \
    sudo

RUN groupadd --gid ${USER_GID} ${USERNAME}
RUN useradd --uid ${USER_UID} --gid ${USER_GID} --groups sudo --create-home ${USERNAME} --shell /usr/bin/bash
RUN echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME}
RUN chmod 0440 /etc/sudoers.d/${USERNAME}

RUN pip install ipykernel tensorflow

USER ${USERNAME}
