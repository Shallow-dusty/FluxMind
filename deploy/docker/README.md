# FluxMind Docker execution images

This directory records the reproducible production image strategy for FluxMind
Docker code execution.

Trace-Twin runs in Hangzhou. Direct Docker Hub or GHCR pulls can time out or
stall on that host, especially for large runtime images. Production should not
depend on direct `docker.io` pulls during service startup or smoke checks.

## Current production images

```text
Python   m.daocloud.io/docker.io/library/python:3.11-slim
Octave   fluxmind/octave:trixie-slim
```

`fluxmind/octave:trixie-slim` is built locally on Trace-Twin from the mirrored
Python slim base image and installs Debian trixie Octave from the USTC mirror.

## Build Octave runtime on Trace-Twin

```bash
cd /opt/fluxmind
docker build \
  -t fluxmind/octave:trixie-slim \
  -f deploy/docker/octave-trixie-slim.Dockerfile \
  deploy/docker
```

## Runtime configuration

```bash
CODE_EXECUTION_BACKEND=docker
DOCKER_PYTHON_EXECUTION_IMAGE=m.daocloud.io/docker.io/library/python:3.11-slim
DOCKER_OCTAVE_EXECUTION_IMAGE=fluxmind/octave:trixie-slim
```

The API, UI, and worker systemd services must run with Docker daemon access,
for example through `SupplementaryGroups=docker`.

## Smoke test

Run the smoke test as the same runtime boundary used by systemd:

```bash
systemd-run --quiet --wait --pipe --collect \
  -p User=fluxmind \
  -p Group=fluxmind \
  -p SupplementaryGroups=docker \
  -p WorkingDirectory=/opt/fluxmind \
  -p EnvironmentFile=/opt/fluxmind/.env \
  /opt/fluxmind/venv/bin/python /opt/fluxmind/scripts/docker_execution_smoke.py --language all
```
