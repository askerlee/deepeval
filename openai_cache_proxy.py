# Usage:
# screen -dm -L -Logfile openai-proxy.txt uvicorn openai_cache_proxy:app --host 127.0.0.1 --port 8088
# pip install fastapi uvicorn httpx[http2] orjson sqlitedict
import os, time, hashlib, orjson
from typing import Dict, Any
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import httpx
from sqlitedict import SqliteDict

OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com")
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")  # optional; can also forward client's
TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 1 day default
DB_PATH     = os.getenv("CACHE_DB_PATH", "./openai_cache.sqlite")

app = FastAPI()
client = httpx.AsyncClient(base_url=OPENAI_BASE, timeout=None)

cache = SqliteDict(DB_PATH, autocommit=True)

def canonicalize_json(data: Dict[str, Any]) -> bytes:
    # Remove harmless defaults that shouldn't affect results if you want
    # (e.g., "user" or "metadata" fields). Here we take the body as-is.
    return orjson.dumps(data, option=orjson.OPT_SORT_KEYS)

def cache_key(path: str, body: Dict[str, Any], auth_id: str) -> str:
    # Include path + payload + model + auth id (optional) to partition keys
    # (Remove auth_id if you want a shared cache across users)
    blob = path + "\n" + auth_id + "\n" + canonicalize_json(body).decode("utf-8")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def now() -> int:
    return int(time.time())

def get_auth_header(req: Request) -> str:
    # Prefer client's Authorization, else fall back to server OPENAI_KEY
    client_auth = req.headers.get("authorization")
    if client_auth:
        return client_auth
    if OPENAI_KEY:
        return f"Bearer {OPENAI_KEY}"
    return ""  # will 401 upstream

CACHABLE_PATHS = {
    "/v1/chat/completions",
    "/v1/embeddings",
    # add others as needed
}

@app.post("/{full_path:path}")
async def proxy_post(full_path: str, request: Request):
    path = "/" + full_path
    body_bytes = await request.body()
    try:
        body = orjson.loads(body_bytes) if body_bytes else {}
    except Exception:
        # not JSON, just forward
        body = None

    # Only cache non-streaming JSON on known paths
    want_cache = (
        body is not None and
        path in CACHABLE_PATHS and
        not body.get("stream", False)
    )

    auth = get_auth_header(request)
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    if auth:
        headers["authorization"] = auth

    if want_cache:
        key = cache_key(path, body, auth_id="shared")
        entry = cache.get(key)
        if entry and entry["exp"] >= now():
            cached_resp = entry["resp"]
            # Return exact cached JSON
            return JSONResponse(content=cached_resp, status_code=200)

    # Miss or not cachable: forward to OpenAI
    upstream = await client.post(path, content=body_bytes, headers=headers, params=request.query_params)

    # Pass-through non-JSON or error responses without caching
    content_type = upstream.headers.get("content-type", "")
    status = upstream.status_code
    raw = upstream.content

    if want_cache and status == 200 and content_type.startswith("application/json"):
        try:
            json_resp = upstream.json()
            cache[key] = {"exp": now() + TTL_SECONDS, "resp": json_resp}
            return JSONResponse(content=json_resp, status_code=200)
        except Exception:
            # fall back to raw
            pass

    return Response(content=raw, status_code=status, headers=dict(upstream.headers))

@app.get("/_cache/stats")
def cache_stats():
    return {"entries": len(cache)}

@app.delete("/_cache/flush")
def cache_flush():
    cache.clear()
    return {"ok": True}
