FROM python:3.11-slim-bookworm as build-base

# set timezone to berlin
ENV TZ=Europe/Berlin
RUN apt-get update && apt-get install -y tzdata gcc curl unzip git bash-completion \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && apt-get upgrade -y --autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install & source poetry; validating that poetry installation / fail early (e.g. SSL issues)
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.6.1 \
    && /root/.local/bin/poetry --version
ENV PATH="${PATH}:/root/.local/bin"
ENV POETRY_VIRTUALENVS_CREATE=False
ENV POETRY_INSTALLER_MAX_WORKERS=10


# create app folder & add to pythonpath for direct python shell execution
WORKDIR /home/root/app
ENV PYTHONPATH=/home/root/app

FROM build-base as ide
# temporary fix for not correct path in vscode terminals (https://github.com/coder/code-server/issues/4699)
RUN echo "export PATH=$PATH" >> /root/.bashrc

# git bash completion
RUN echo 'source /usr/share/bash-completion/completions/git' >> ~/.bashrc

# pyproject.toml and optional lock-file
COPY pyproject.toml poetry.lock* ./
RUN poetry install \
    && rm -rf ~/.cache/pypoetry/{cache,artifacts} \
    && rm pyproject.toml poetry.lock
