FROM --platform=linux/amd64 python:3.10-buster AS builder

RUN apt-get update && apt-get install -y curl gcc g++
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN apt-get update && /usr/local/bin/python -m pip install --upgrade pip && apt-get install -y git

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_INSTALLER_MAX_WORKERS=10 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    PATH="/root/.local/bin:$PATH" \
    PYTHONPATH=/app:$PYTHONPATH

WORKDIR app/

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root && rm -rf $POETRY_CACHE_DIR

FROM --platform=linux/amd64 python:3.10-slim-buster AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils && apt-get install libgomp1

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=TRUE \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}