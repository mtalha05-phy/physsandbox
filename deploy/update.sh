#!/usr/bin/env bash
# Update a running PhysSandbox VM to the latest main, in place.
# Keeps .env, .venv, the lab notebook, its signing key and the stored results.
#
#   sudo bash deploy/update.sh          update, keep the lab notebook
#   sudo bash deploy/update.sh fresh    update and start a new lab notebook
#                                       (the old one is moved aside, not deleted)
set -euo pipefail
APP_DIR=$(systemctl show -p WorkingDirectory --value physsandbox)
REPO=${REPO:-mukeshkbj/physsandbox}
BRANCH=${BRANCH:-main}
cd "$APP_DIR"

echo "==> Fetching $REPO@$BRANCH into $APP_DIR"
tmp=$(mktemp -d)
curl -fsSL "https://codeload.github.com/$REPO/tar.gz/refs/heads/$BRANCH" | tar -xz -C "$tmp" --strip-components=1
cp -a "$tmp"/. "$APP_DIR"/
rm -rf "$tmp"

echo "==> Python packages"
.venv/bin/pip install -q -r requirements.txt

if [ "${1:-}" = fresh ]; then
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  for f in receipts.jsonl receipts.head.json; do
    if [ -f "$f" ]; then mv "$f" "$f.$stamp"; echo "==> Old notebook kept as $f.$stamp"; fi
  done
fi

echo "==> Sandbox image"
docker build -q -t physsandbox-runner:latest sandbox/ >/dev/null

echo "==> Restarting"
systemctl restart physsandbox
for i in $(seq 1 20); do
  if curl -fsS http://127.0.0.1:8000/api/ledger/public-key >/dev/null 2>&1 \
     || curl -fsS http://127.0.0.1/api/ledger/public-key >/dev/null 2>&1; then
    echo "==> Up. Experiments: $(.venv/bin/python -c 'from app.pipeline import CATALOG; print(len(CATALOG["experiments"]))')"
    curl -fsS http://127.0.0.1:8000/api/ledger/public-key 2>/dev/null || curl -fsS http://127.0.0.1/api/ledger/public-key
    echo
    exit 0
  fi
  sleep 1
done
echo "!! The service did not come up; see: journalctl -u physsandbox -n 50"
exit 1
