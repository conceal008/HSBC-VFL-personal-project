#!/bin/bash
set -euo pipefail
party="$1"
case "$party" in alice|bob) ;; *) exit 2 ;; esac
export DEBIAN_FRONTEND=noninteractive
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="*" no_proxy="*"
apt-get update
apt-get install --no-install-recommends -y python3.10-venv openssl iptables libgomp1
id "$party" >/dev/null 2>&1 || useradd --create-home --shell /bin/bash "$party"
passwd --lock "$party"
install -d -m 700 -o "$party" -g "$party" "/srv/vfl/$party" "/srv/vfl/$party/tls"
install -d -m 700 /srv/forbidden
printf 'synthetic permission probe\n' >/srv/forbidden/peer_sentinel
chmod 600 /srv/forbidden/peer_sentinel
python3.10 -m venv /opt/secretflow
/opt/secretflow/bin/pip install -r /tmp/hsbc-secretflow.lock
/opt/secretflow/bin/pip freeze >/opt/secretflow/environment.lock
chmod -R go-w /opt/secretflow
sudo -u "$party" openssl req -new -newkey rsa:3072 -nodes \
  -keyout "/srv/vfl/$party/tls/key.pem" -out "/srv/vfl/$party/tls/request.csr" \
  -subj "/CN=lima-hsbc-$party.internal"
chmod 600 "/srv/vfl/$party/tls/key.pem"
/opt/secretflow/bin/python -c 'import secretflow,spu,heu,jax; print("SecretFlow/SPU/HEU/JAX Linux imports passed")'
