FROM nvcr.io/nvidia/tensorrt:25.06-py3

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y libxcb1 libgl1 libx11-6 libxcb-cursor0

COPY pyproject.toml .
COPY uv.lock .
RUN uv sync
