# FluxMind Deployment Status

Last live check: 2026-05-12 20:11 CST

This document records the current temporary deployment snapshot. Treat it as a
pointer for re-checking the live host, not as proof that the service is still
healthy at a later time.

## Current Deployment

```
Host          Trace-Twin
Tailscale     root@100.100.233.26
Public IP     223.6.253.9
Deploy root   /opt/fluxmind
Runtime user  fluxmind
UI service    fluxmind-ui.service
API service   fluxmind-api.service
UI port       18501
API port      18502
```

Public endpoints:

- Preferred Web UI: `https://smy.hyper-dusty.cloud/`
- Web UI: `http://223.6.253.9:18501/`
- API health: `http://223.6.253.9:18502/health`
- API query: `POST http://223.6.253.9:18502/query`

API calls to `/query` require `X-API-Key: <token>`. The token is stored only on
the server in `/opt/fluxmind/.env`; do not copy it into this repository.

## Isolation Boundary

FluxMind is deployed separately from the Trace-Twin bot stack:

- no Docker deployment for FluxMind
- no Docker restart during deployment
- no changes to `/opt/trace-twin`
- independent systemd services
- independent ports `18501` and `18502`
- independent Cloudflare Tunnel service `cloudflared-fluxmind-smy.service`

The existing bot containers checked healthy at the last verification:

- `bot-resume`
- `bot-lingju`

## Runtime Configuration

The deployed `.env` currently uses:

```
LLM_MODEL=deepseek-v4-flash
EMBEDDING_MODEL=/opt/fluxmind/models/all-MiniLM-L6-v2
```

The sentence-transformers embedding model was copied to the server under
`/opt/fluxmind/models/all-MiniLM-L6-v2`, so normal service startup should not
depend on downloading from Hugging Face.

## Last Verification

Live checks performed on 2026-05-12 20:11 CST:

```
fluxmind-ui.service     active
fluxmind-api.service    active
cloudflared service     active
docker.service          active
UI listener             0.0.0.0:18501
API listener            0.0.0.0:18502
local API health        {"status":"ok"}
Cloudflare UI HTTP      200 at https://smy.hyper-dusty.cloud/
public UI HTTP          200
public API health HTTP  200
bot-resume              healthy
bot-lingju              healthy
available memory        about 1.8 GiB
root disk free          26G
```

## Cloudflare Tunnel

Cloudflare routes `smy.hyper-dusty.cloud` to the Streamlit UI through a
dedicated named tunnel:

```
Zone       hyper-dusty.cloud
Hostname   smy.hyper-dusty.cloud
Tunnel     fluxmind-smy
Tunnel ID  692b5ddf-2684-4a2f-84d4-30c87bf32dba
Service    cloudflared-fluxmind-smy.service
Ingress    smy.hyper-dusty.cloud -> http://127.0.0.1:18501
```

The tunnel token is stored only on the server in
`/etc/default/cloudflared-fluxmind-smy`; do not copy it into this repository.
The API remains on the original `18502` endpoint with token protection and is
not routed through `smy.hyper-dusty.cloud`.

## Refresh Commands

Use live state before making deployment decisions:

```bash
ssh -o BatchMode=yes root@100.100.233.26 \
  'systemctl is-active cloudflared-fluxmind-smy.service fluxmind-ui.service fluxmind-api.service docker.service;
   ss -ltnp | egrep "18501|18502" || true;
   curl -sS --max-time 10 http://127.0.0.1:18502/health;
   docker ps --format "{{.Names}} {{.Status}}" | egrep "bot-resume|bot-lingju" || true;
   grep -E "^(LLM_MODEL|EMBEDDING_MODEL)=" /opt/fluxmind/.env;
   free -h | sed -n "2p";
   df -h / | sed -n "2p"'

curl -sS --max-time 10 -o /dev/null -w 'public_ui=%{http_code}\n' \
  http://223.6.253.9:18501/

curl -sS --max-time 10 -o /dev/null -w 'smy_https=%{http_code}\n' \
  https://smy.hyper-dusty.cloud/

curl -sS --max-time 10 -o /dev/null -w 'public_api_health=%{http_code}\n' \
  http://223.6.253.9:18502/health
```
