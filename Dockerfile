# syntax=docker/dockerfile:1.4
ARG PYTHON_VERSION=3.11.14
ARG PORT=8080

FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Copy source code
COPY . .

# Copy and permission the entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE ${PORT}
EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]