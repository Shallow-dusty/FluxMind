# FluxMind Deployment Status

Last live check: 2026-05-30 01:18 CST

This document records the current deployment snapshot. Treat it as a
pointer for re-checking the live host, not as proof that the service is still
healthy at a later time.

## Current Deployment

Workspace directory: `11.FluxMind/`
Last restarted application-code commit: `5c0440b`

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
- Preferred API:    `https://api-smy.hyper-dusty.cloud/`
- Web UI (raw):     `http://223.6.253.9:18501/`
- API health (raw): `http://223.6.253.9:18502/health`

Both HTTPS hostnames terminate at Cloudflare and tunnel back to the same origin
ports through `fluxmind-smy` (one tunnel, two ingress rules). The raw HTTP IP
endpoints stay reachable for diagnostics, but Coze / third-party agent
integration should use the HTTPS hostname so the OpenAPI schema is fetched over
TLS.

API calls to `/query` require `X-API-Key: <token>` (or the equivalent
`Authorization: Bearer <token>`). The token is stored only on the server in
`/opt/fluxmind/.env`; do not copy it into this repository.

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
LLM_BASE_URL=https://token-plan-sgp.xiaomimimo.com/v1
LLM_MODEL=mimo-v2.5-pro
EMBEDDING_MODEL=/opt/fluxmind/models/all-MiniLM-L6-v2
```

`mimo-v2.5-pro` is a reasoning model: it emits `reasoning_content` first
and the final answer second. `src/chain.py::query_stream` exposes the
reasoning as a `> 💭` blockquote, then a horizontal rule, then the answer,
so the Streamlit UI no longer looks frozen during the thinking phase. The
Streamlit layer uses a stable placeholder plus browser-translation guards,
because Chrome/Google Translate can mutate streamed text DOM nodes while the
frontend is still updating them. The non-streaming `/query` endpoint returns
the final answer only.

Previous pool `api.268526.eu.cc` with `deepseek-v4-flash` was retired on
2026-05-12 after it began returning `upstream_empty_output` (HTTP 429)
on every call. The Xiaomi MiMo pool (`token-plan-sgp.xiaomimimo.com`)
replaced it.

The sentence-transformers embedding model was copied to the server under
`/opt/fluxmind/models/all-MiniLM-L6-v2`, so normal service startup should not
depend on downloading from Hugging Face.

## Last Verification

Live checks refreshed on 2026-05-30 01:18 CST after syncing application commit
`5c0440b` to `/opt/fluxmind` and restarting `fluxmind-ui.service` and
`fluxmind-api.service`. Later documentation-only commits may be synced without
another service restart; use `git log -1` in the source checkout for the latest
repository revision.

```
fluxmind-ui.service     active
fluxmind-api.service    active
cloudflared service     active
docker.service          active
UI listener             0.0.0.0:18501
API listener            0.0.0.0:18502
local API health        {"status":"ok"}
Cloudflare UI HTTP      200 at https://smy.hyper-dusty.cloud/
Cloudflare API HTTP     200 at https://api-smy.hyper-dusty.cloud/health
public UI HTTP          200 at http://223.6.253.9:18501/
public API health HTTP  200 at http://223.6.253.9:18502/health
deployed stream guard   present in /opt/fluxmind/app.py
deployed capabilities   present in /opt/fluxmind/src/capabilities.py
no-key provider docs     present in /opt/fluxmind/docs/BACKLOG.md
bot-resume              healthy
bot-lingju              healthy
available memory        about 2.2 GiB
root disk free          26G
```

During the restart window, Cloudflare briefly returned 502 because the tunnel
reached the origin while the UI/API processes were restarting. Follow-up checks
returned 200 for both HTTPS endpoints and raw diagnostic endpoints.

## Cloudflare Tunnel

Cloudflare routes both `smy.hyper-dusty.cloud` (UI) and
`api-smy.hyper-dusty.cloud` (FastAPI) through a single named tunnel:

```
Zone       hyper-dusty.cloud
Tunnel     fluxmind-smy
Tunnel ID  692b5ddf-2684-4a2f-84d4-30c87bf32dba
Service    cloudflared-fluxmind-smy.service
Ingress    smy.hyper-dusty.cloud      -> http://127.0.0.1:18501   (Streamlit UI)
           api-smy.hyper-dusty.cloud  -> http://127.0.0.1:18502   (FastAPI)
           *                          -> http_status:404
```

Ingress is managed remotely (this is a token-mode tunnel, no local YAML
config). Updates go through the Cloudflare API:
`PUT /accounts/{acct}/cfd_tunnel/{tunnel_id}/configurations`. DNS CNAMEs for
both hostnames point to `<tunnel_id>.cfargotunnel.com` with `proxied: true`.

The tunnel token is stored only on the server in
`/etc/default/cloudflared-fluxmind-smy`; do not copy it into this repository.

## Refresh Commands

Use live state before making deployment decisions:

```bash
python scripts/health_check.py \
  --url https://smy.hyper-dusty.cloud/ \
  --url https://api-smy.hyper-dusty.cloud/health

python scripts/health_check.py --ssh-host root@100.100.233.26

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
