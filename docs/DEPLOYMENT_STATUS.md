# FluxMind Deployment Status

Last live check: 2026-05-30 03:28 CST

This document records the current deployment snapshot. Treat it as a
pointer for re-checking the live host, not as proof that the service is still
healthy at a later time.

## Current Deployment

Workspace directory: `11.FluxMind/`
Last restarted application-code commit: `3ddb936`

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

Local runtime state directories are owned by the `fluxmind` runtime user:
`/opt/fluxmind/metadata`, `/opt/fluxmind/jobs`, and
`/opt/fluxmind/artifacts`.

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

Live checks refreshed on 2026-05-30 03:28 CST after syncing application commit
`3ddb936` to `/opt/fluxmind` and restarting `fluxmind-ui.service` and
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
deployed job panel      present in /opt/fluxmind/app.py
deployed capabilities   present in /opt/fluxmind/src/capabilities.py
deployed runtime layer   present in /opt/fluxmind/src/runtime.py
deployed no-key providers present in /opt/fluxmind/src/providers.py
local Octave provider    present in /opt/fluxmind/src/providers.py; gnu-octave-local metadata installed
deployed job layer       present in /opt/fluxmind/src/jobs.py
SQLite job mirror        present in /opt/fluxmind/src/jobs.py
scheduled retry/backoff  present in /opt/fluxmind/src/jobs.py
queued job recovery      present in /opt/fluxmind/src/jobs.py; API startup calls recover_queued_jobs
admin queue health       present in /opt/fluxmind/src/admin.py; authenticated API returned queue_health
deployed artifact layer  present in /opt/fluxmind/src/artifacts.py
deployed admin layer     present in /opt/fluxmind/src/admin.py
deployed metadata layer  present in /opt/fluxmind/src/metadata.py
deployed eval layer      present in /opt/fluxmind/src/evaluation.py
hybrid retrieval         present in /opt/fluxmind/src/chain.py; query uses hybrid_retrieve
local reranker           present in /opt/fluxmind/src/chain.py; hybrid_retrieve uses rerank_documents
artifact references      present in /opt/fluxmind/src/chain.py; RAG prompt includes Generated Artifact References
artifact formatter       present in /opt/fluxmind/src/artifacts.py; stable artifact IDs can be cited
mock image metadata      present in /opt/fluxmind/src/providers.py; local-mock-svg-v1 metadata installed
execution artifacts      local code job captured result.txt artifact
artifact export route    present; authenticated local API listed result.txt
artifact gallery         present in /opt/fluxmind/app.py; stable IDs and metadata rendered
admin status panel       present in /opt/fluxmind/app.py
admin status route       present; authenticated local API returned runtime state
job SQLite state         /opt/fluxmind/jobs/jobs.sqlite3 exists; 2 current rows
scheduled retry smoke    queued retry executed; parent_job_id/not_before present
job retry/cancel UI      present in /opt/fluxmind/app.py
offline RAG eval         passed in /opt/fluxmind
corpus metadata route    present in /opt/fluxmind/api.py
active corpus route      present in /opt/fluxmind/api.py
corpus metadata papers   6 indexed papers via authenticated local API check
active corpus smoke      PUT /corpus/active preserved 6 active papers; rebuild_required=true
job API routes           present in /opt/fluxmind/api.py
index rebuild job route  present in /opt/fluxmind/api.py
async index job route    present in /opt/fluxmind/api.py
Octave job routes        present in /opt/fluxmind/api.py; immediate and async routes installed
query answer mode        present in /opt/fluxmind/api.py
job retry route          present in /opt/fluxmind/api.py
scheduled retry route    present in /opt/fluxmind/api.py
admin status jobs        5 total, 4 failed historical/smoke local code jobs
admin status job storage jobs.jsonl 6474 bytes; jobs.sqlite3 32768 bytes
admin status queue       queued 0, due 0, scheduled 0, running 0
admin status corpus      6 papers, 6 active, 6 indexed
admin status artifacts   1 artifact, 2 bytes
active paper count      6
FAISS index size        786477 bytes
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
