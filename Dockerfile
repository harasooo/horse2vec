FROM python:3.9.1-slim

RUN apt-get update -y --fix-missing && \
    apt-get -y install \
    make 

RUN pip install --upgrade pip && pip --version
RUN pip install poetry

WORKDIR /root
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction
