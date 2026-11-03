# Image used for every sandboxed job. No secrets, no network tools needed.
FROM python:3.12-slim

RUN pip install --no-cache-dir numpy==2.* scipy==1.* lmfit==1.* \
    && rm -rf /root/.cache

COPY tools /tools
COPY attacks /attacks

# Runs as nobody; the host also forces --user 65534 and a read-only root.
USER 65534:65534
WORKDIR /tmp
