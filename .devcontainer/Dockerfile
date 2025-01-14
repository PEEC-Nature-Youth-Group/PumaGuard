FROM ubuntu:jammy

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
ARG http_proxy=""
ARG https_proxy=""
ARG no_proxy=""

RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --yes \
    apt-utils \
    ca-certificates

RUN echo 'Acquire::http::Proxy "'"${http_proxy}"'";' | tee --append /etc/apt/apt.conf.d/00proxy
RUN echo 'Acquire::https::Proxy "'"${https_proxy}"'";' | tee --append /etc/apt/apt.conf.d/00proxy

RUN DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --yes \
    bash-completion \
    command-not-found \
    curl \
    gh \
    git \
    inotify-tools \
    iproute2 \
    less \
    locales \
    make \
    openssh-client \
    python3-pip \
    python3-venv \
    sudo \
    vim

RUN DEBIAN_FRONTEND=noninteractive apt-get update

RUN groupadd --gid ${USER_GID} ${USERNAME}
RUN useradd --uid ${USER_UID} --gid ${USER_GID} --groups sudo \
    --create-home ${USERNAME} --shell /usr/bin/bash
RUN echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME}
RUN chmod 0440 /etc/sudoers.d/${USERNAME}

RUN sed --in-place --expression 's/^#.*\(en_US.UTF.*\)/\1/' /etc/locale.gen
RUN dpkg-reconfigure --frontend=noninteractive locales
RUN update-locale LANG=en_US.UTF-8

RUN pip install poetry

USER ${USERNAME}

RUN echo "source <(gh completion --shell bash)" | tee --append ~/.bashrc
