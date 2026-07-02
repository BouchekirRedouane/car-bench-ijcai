"""fortiss public-Ollama integration for CAR-bench.

The fortiss Ollama service (https://ollama.fortiss-demo.org) uses a two-step auth:
exchange a long-lived `api-key` for a short-lived (60 min) JWT, then call the
native Ollama API with `Authorization: Bearer <jwt>`. litellm cannot do that
exchange itself, so this module:

  * manages the token (caches it, refreshes ~1 min before expiry, retries on 401);
  * monkeypatches `litellm.completion` so any model named `fortiss/<ollama-model>`
    is routed to the fortiss endpoint via litellm's `ollama_chat` provider with a
    fresh Bearer header injected.

Call `install()` ONCE per process, BEFORE anything does `from litellm import
completion` (so the bound reference picks up the wrapper). Both the agent and the
evaluator import and install it at startup.

Config (env): FORTISS_OLLAMA_API_KEY (required), FORTISS_OLLAMA_BASE_URL (optional,
defaults to the public host).
"""
from __future__ import annotations

import base64
import copy
import json
import logging
import os
import threading
import time
import urllib.request

logger = logging.getLogger("harness.fortiss")

PREFIX = "fortiss/"
_DEFAULT_BASE = "https://ollama.fortiss-demo.org"
_lock = threading.Lock()
_cache = {"token": None, "exp": 0.0}
_orig_completion = None


def base_url() -> str:
    return (os.getenv("FORTISS_OLLAMA_BASE_URL") or _DEFAULT_BASE).rstrip("/")


def _jwt_exp(token: str, fallback: float) -> float:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return float(json.loads(base64.urlsafe_b64decode(payload)).get("exp", fallback))
    except Exception:
        return fallback


def get_token(force: bool = False) -> str:
    """Return a valid bearer token, refreshing if needed (thread-safe)."""
    with _lock:
        now = time.time()
        if not force and _cache["token"] and now < _cache["exp"] - 60:
            return _cache["token"]
        key = os.environ.get("FORTISS_OLLAMA_API_KEY")
        if not key:
            raise RuntimeError("FORTISS_OLLAMA_API_KEY not set")
        req = urllib.request.Request(
            base_url() + "/api/get-token", method="POST", headers={"api-key": key}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            tok = json.load(r).get("access_token")
        if not tok:
            raise RuntimeError("fortiss get-token returned no access_token")
        _cache["token"] = tok
        _cache["exp"] = _jwt_exp(tok, now + 3000)
        logger.info("fortiss token refreshed (valid ~%ds)", int(_cache["exp"] - now))
        return tok


def _sanitize_schema(node):
    """Make a JSON-Schema node safe for Ollama's strict tool parser.

    Ollama's Go structs reject constructs OpenAI/Anthropic/Gemini tolerate:
    boolean sub-schemas (`{"foo": true}`), `additionalProperties: true/false`, and
    non-object schema nodes where an object is expected. We coerce boolean
    sub-schemas to permissive empty objects and drop boolean additionalProperties.
    Recurses through properties / items / anyOf / oneOf / allOf / $defs."""
    if isinstance(node, bool):
        return {}                      # boolean sub-schema -> permissive object
    if isinstance(node, list):
        return [_sanitize_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "additionalProperties":
            if isinstance(v, dict):
                out[k] = _sanitize_schema(v)
            # drop boolean additionalProperties entirely (Ollama can't parse it)
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _sanitize_schema(pv) for pk, pv in v.items()}
        elif k in ("items", "additionalItems"):
            out[k] = _sanitize_schema(v)
        elif k in ("anyOf", "oneOf", "allOf", "prefixItems") and isinstance(v, list):
            out[k] = [_sanitize_schema(x) for x in v]
        elif k in ("$defs", "definitions") and isinstance(v, dict):
            out[k] = {dk: _sanitize_schema(dv) for dk, dv in v.items()}
        elif isinstance(v, (dict, list)):
            out[k] = _sanitize_schema(v)
        else:
            out[k] = v
    return out


def _sanitize_tools(tools):
    """Deep-copy + sanitize each tool's parameter schema for Ollama."""
    cleaned = []
    for t in tools:
        t = copy.deepcopy(t)
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("parameters"), dict):
            fn["parameters"] = _sanitize_schema(fn["parameters"])
        cleaned.append(t)
    return cleaned


def _auth_error(e: Exception) -> bool:
    s = str(e).lower()
    return "401" in s or "unauthorized" in s or "invalid authentication" in s or "expired" in s


def install() -> None:
    """Monkeypatch litellm.completion to route `fortiss/*` models to the fortiss
    Ollama endpoint with a fresh bearer token. Idempotent."""
    global _orig_completion
    import litellm

    litellm.drop_params = True  # ollama models reject unknown params (reasoning_effort, ...)
    if getattr(litellm, "_fortiss_patched", False):
        return
    _orig_completion = litellm.completion

    def wrapper(*args, **kwargs):
        model = kwargs.get("model")
        if isinstance(model, str) and model.startswith(PREFIX):
            kwargs["model"] = model[len(PREFIX):]            # bare ollama model id
            kwargs["custom_llm_provider"] = "ollama_chat"
            kwargs["api_base"] = base_url()
            if kwargs.get("tools"):                          # Ollama-safe tool schemas
                kwargs["tools"] = _sanitize_tools(kwargs["tools"])
            hdr = dict(kwargs.get("extra_headers") or {})
            hdr["Authorization"] = "Bearer " + get_token()
            kwargs["extra_headers"] = hdr
            kwargs["headers"] = dict(hdr)                    # belt and suspenders
            try:
                return _orig_completion(*args, **kwargs)
            except Exception as e:  # refresh token once on auth failure
                if _auth_error(e):
                    tok = get_token(force=True)
                    kwargs["extra_headers"]["Authorization"] = "Bearer " + tok
                    kwargs["headers"]["Authorization"] = "Bearer " + tok
                    return _orig_completion(*args, **kwargs)
                raise
        return _orig_completion(*args, **kwargs)

    litellm.completion = wrapper
    litellm._fortiss_patched = True
    logger.info("fortiss Ollama litellm patch installed (base=%s)", base_url())
