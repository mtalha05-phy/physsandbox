# Image used for every sandboxed job. No secrets, no network tools.
# Library versions are pinned so the image digest in each receipt maps to a known
# set of code (the same versions the evaluation runs with).
FROM python:3.12-slim

# strace: the runner records the system calls of agent code and attack demos, so
# what the code tried comes from the kernel-level record, not from its own output.
RUN apt-get update \
    && apt-get install -y --no-install-recommends strace \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy==2.5.3 scipy==1.18.1 lmfit==1.3.4 \
    && rm -rf /root/.cache

COPY tools /tools
COPY attacks /attacks

# Runs as nobody; the host also forces --user 65534 and a read-only root.
USER 65534:65534
WORKDIR /tmp
