FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN adduser agent
WORKDIR /home/agent

COPY pyproject.toml uv.lock README.md ./
COPY src src
COPY example /home/agent/example

# Install deps as root (more robust), then fix ownership
RUN \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

RUN chown -R agent:agent /home/agent
USER agent

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
