# syntax=docker/dockerfile:1
# Two stages on the same base image, so the virtualenv's interpreter path is valid in both.
ARG PYTHON_IMAGE=python:3.12-slim

# Build: resolve the locked runtime dependencies into /venv with uv. Every dependency ships a
# manylinux wheel, so nothing is compiled and no toolchain is installed in either stage.
FROM ${PYTHON_IMAGE} AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12
WORKDIR /src
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# Runtime: the virtualenv on a clean slim image, unprivileged (uid 10001). Datasets, models and
# traces are not baked in: mount a volume at /data and run `ircot fetch` to fill it.
FROM ${PYTHON_IMAGE}
RUN groupadd --gid 10001 ircot \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/ircot \
        --shell /usr/sbin/nologin ircot \
    && install -d -o 10001 -g 10001 /data
COPY --from=build /venv /venv
COPY LICENSE /usr/share/doc/ircot-multihop-retrieval/LICENSE
ENV PATH=/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    IRCOT_DATA_DIR=/data \
    HF_HOME=/data/hf
USER 10001:10001
WORKDIR /home/ircot
VOLUME ["/data"]
# A batch command-line tool, not a long-running service: there is nothing to probe.
HEALTHCHECK NONE
ENTRYPOINT ["ircot"]
CMD ["demo"]
