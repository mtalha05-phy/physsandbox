#!/usr/bin/env bash
# Serve PhysSandbox over HTTPS with an automatic Let's Encrypt certificate.
# Uses <ip-with-dashes>.sslip.io, a public DNS name that resolves to this IP,
# so no domain purchase is needed. Caddy obtains and renews the certificate.
#   sudo bash deploy/enable_https.sh
set -euo pipefail
IP=$(curl -s -4 ifconfig.me)
HOST="${IP//./-}.sslip.io"

apt-get install -y caddy
# The app moves to localhost:8000; Caddy takes ports 80 and 443.
sed -i -E 's/--host 0\.0\.0\.0 --port [0-9]+/--host 127.0.0.1 --port 8000/' /etc/systemd/system/physsandbox.service
systemctl daemon-reload && systemctl restart physsandbox

cat > /etc/caddy/Caddyfile <<CADDY
$HOST {
    encode gzip
    reverse_proxy 127.0.0.1:8000
}
http://$IP {
    redir https://$HOST{uri}
}
CADDY
systemctl enable caddy && systemctl restart caddy
command -v ufw >/dev/null && ufw allow 443/tcp || true
echo "==> Done. Open https://$HOST/"
