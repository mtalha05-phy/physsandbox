#!/usr/bin/env bash
# One-time setup for a fresh Vultr Cloud Compute VM (Ubuntu 24.04).
# Installs Docker + gVisor, builds the sandbox image, and runs the app on port 80.
#
#   git clone https://github.com/<you>/physsandbox.git && cd physsandbox
#   cp .env.example .env && nano .env        # add your Vultr Inference key + model
#   sudo bash deploy/setup_vultr.sh
set -euo pipefail
cd "$(dirname "$0")/.."
APP_DIR=$(pwd)

echo "==> Installing Docker and Python"
apt-get update -y
apt-get install -y docker.io python3-venv python3-pip curl gnupg

echo "==> Installing gVisor (runsc) and registering it with Docker"
curl -fsSL https://gvisor.dev/archive.key | gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" \
  > /etc/apt/sources.list.d/gvisor.list
apt-get update -y && apt-get install -y runsc
runsc install
systemctl restart docker

echo "==> Checking gVisor works"
docker run --rm --runtime=runsc hello-world >/dev/null && echo "gVisor OK"

echo "==> Building the sandbox image"
docker build -t physsandbox-runner:latest sandbox/

echo "==> Python environment"
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python data/generate.py

echo "==> systemd service"
cat > /etc/systemd/system/physsandbox.service <<EOF
[Unit]
Description=PhysSandbox
After=docker.service
Requires=docker.service

[Service]
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 80
Restart=always

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now physsandbox

echo "==> Firewall: allow SSH and HTTP only"
if command -v ufw >/dev/null; then ufw allow OpenSSH; ufw allow 80/tcp; ufw --force enable; fi

echo "==> Done. Open http://$(curl -s ifconfig.me)/"
