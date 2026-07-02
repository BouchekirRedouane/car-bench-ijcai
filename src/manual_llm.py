"""Manual (human-as-LLM) provider for CAR-bench — run with NO API credit, via a
small local WEB GUI.

When env MANUAL_LLM is truthy, this monkeypatches `litellm.completion` so that
every model call (agent draft, CoVe teacher, policy compiler, simulated user,
policy judge — in BOTH the agent and evaluator processes) is turned into a
browser form instead of an API request:

  1. A tiny web server (started in-process, shared between both processes) holds
     the pending prompt.
  2. You open  http://127.0.0.1:8765  in a browser. The page shows the current
     prompt with a "Copy" button and a box to paste the model's reply.
  3. You copy the prompt into any chat GUI (ChatGPT / Claude / Gemini web),
     paste the reply into the form, click Submit. The run continues.

The benchmark is turn-based, so exactly one call is pending at any moment.

Output-format contract:
  * Tool calls (agent): reply with ONE JSON object
        {"content": "<user-facing text or empty>",
         "tool_calls": [{"name": "<tool>", "arguments": {...}}, ...]}
    Use tool_calls to act; leave it [] and fill content to talk to the user.
  * JSON / TEXT tasks (teacher / policy compiler / user-sim): paste the reply
    verbatim.

Config (env): MANUAL_LLM=1 to enable; MANUAL_LLM_PORT to change the port
(default 8765). Call install() ONCE per process, AFTER any other litellm patch.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_orig_completion = None

# Server-side store (lives in whichever process binds the port first).
_STORE: "OrderedDict[str, dict]" = OrderedDict()
_SLOCK = threading.Lock()
_SEQ = 0
_server_started = False
_server_lock = threading.Lock()

_COPY_START = ">>>>>>>>>>>>>>>>>>>>  COPY FROM HERE  >>>>>>>>>>>>>>>>>>>>"
_COPY_END = "<<<<<<<<<<<<<<<<<<<<  COPY TO HERE  <<<<<<<<<<<<<<<<<<<<"


def enabled() -> bool:
    v = os.getenv("MANUAL_LLM")
    return bool(v) and v.strip().lower() in ("1", "true", "yes", "on")


def _port() -> int:
    try:
        return int(os.getenv("MANUAL_LLM_PORT") or 8765)
    except ValueError:
        return 8765


# --------------------------------------------------------------------------- #
# Render the request into a copy-pasteable prompt
# --------------------------------------------------------------------------- #
def _render_tools(tools) -> str:
    lines = []
    for t in tools or []:
        fn = (t or {}).get("function") or {}
        params = fn.get("parameters") or {}
        lines.append(f"- {fn.get('name', '?')}  params={json.dumps(params, ensure_ascii=False)}")
    return "\n".join(lines) or "(none)"


def _render_messages(messages) -> str:
    out = []
    for m in messages or []:
        role = (m.get("role") or "?").upper()
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        if m.get("tool_calls"):
            tcs = "; ".join(
                f"{(tc.get('function') or {}).get('name')}({(tc.get('function') or {}).get('arguments')})"
                for tc in m["tool_calls"]
            )
            out.append(f"[{role} tool_calls] {tcs}")
        if content:
            out.append(f"[{role}] {content}")
    return "\n\n".join(out)


_TOOL_FORMAT = (
    "OUTPUT FORMAT — reply with ONE JSON object and nothing else:\n"
    '{"content": "<user-facing text, or empty string>", '
    '"tool_calls": [{"name": "<tool name>", "arguments": { ... }}]}\n'
    "Put actions in tool_calls (arguments must match the tool params above). "
    "To speak to the user instead, set tool_calls to [] and fill content."
)
_JSON_FORMAT = (
    "OUTPUT FORMAT — the system message above already states the exact JSON to "
    "return. Paste the model's reply verbatim (JSON or text)."
)


def _schema_text(response_format) -> str:
    """If the call requests structured output (a Pydantic model or a json_schema),
    render its schema so the human produces the exact required fields."""
    rf = response_format
    if not rf:
        return ""
    schema = None
    if isinstance(rf, type):  # a Pydantic model class
        try:
            schema = rf.model_json_schema()
        except Exception:
            schema = None
    elif isinstance(rf, dict):
        if rf.get("type") == "json_schema":
            js = rf.get("json_schema") or {}
            schema = js.get("schema") or js
        # {"type": "json_object"} carries no schema -> nothing to render
    if not schema:
        return ""
    props = schema.get("properties") or {}
    required = schema.get("required") or list(props.keys())
    skeleton = {k: f"<{(props.get(k) or {}).get('type', 'value')}>" for k in props}
    return (
        "OUTPUT FORMAT — reply with ONE JSON object matching THIS schema exactly "
        "(all required fields present, exact field names):\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + "\n\nFill this template:\n"
        + json.dumps(skeleton, ensure_ascii=False, indent=2)
        + (f"\nRequired fields: {', '.join(required)}" if required else "")
    )


def _build_prompt(model: str, messages, tools, response_format) -> str:
    parts = [_COPY_START, _render_messages(messages)]
    if tools:
        parts += ["", "AVAILABLE TOOLS:", _render_tools(tools), "", _TOOL_FORMAT]
    else:
        schema = _schema_text(response_format)
        parts += ["", schema or _JSON_FORMAT]
    parts += [_COPY_END]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Web GUI server (stdlib only)
# --------------------------------------------------------------------------- #
_PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>CAR-bench — Manual LLM</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;background:#0f1320;color:#e6e9ef}
 h1{font-size:18px} .muted{color:#8b93a7}
 .badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:12px;font-weight:600}
 .TOOLCALL{background:#2a4d7a;color:#cfe3ff}.JSON{background:#5a3a7a;color:#e9d6ff}.TEXT{background:#3a5a3a;color:#d6ffd6}
 textarea{width:100%;box-sizing:border-box;background:#11162a;color:#e6e9ef;border:1px solid #2a3350;border-radius:8px;padding:10px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px}
 #prompt{height:300px}#reply{height:160px}
 button{cursor:pointer;border:0;border-radius:8px;padding:9px 16px;font-weight:600;margin-right:8px}
 #copy{background:#2a4d7a;color:#cfe3ff}#submit{background:#2e7d4f;color:#dfffe9}
 .row{margin:12px 0}.idle{color:#8b93a7;font-style:italic}
</style></head><body>
<h1>CAR-bench — Manual LLM <span id="meta" class="muted"></span></h1>
<div id="idle" class="idle">Waiting for the run to produce the next call…</div>
<div id="panel" style="display:none">
 <div class="row"><span id="kind" class="badge"></span> <span id="cid" class="muted"></span></div>
 <div class="row"><b>Prompt</b> — copy into your chat GUI:
   <button id="copy">Copy prompt</button></div>
 <div class="row"><textarea id="prompt" readonly></textarea></div>
 <div class="row"><b>Paste the model's reply here:</b></div>
 <div class="row"><textarea id="reply" placeholder="Paste the model output, then click Submit"></textarea></div>
 <div class="row"><button id="submit">Submit reply</button>
   <span id="status" class="muted"></span></div>
</div>
<script>
let cur=null;
async function poll(){
 try{
  const r=await fetch('/api/pending'); const d=await r.json();
  if(d&&d.id){
   if(!cur||cur!==d.id){
    cur=d.id;
    document.getElementById('idle').style.display='none';
    document.getElementById('panel').style.display='block';
    document.getElementById('prompt').value=d.prompt;
    document.getElementById('reply').value='';
    const k=document.getElementById('kind'); k.textContent=d.kind; k.className='badge '+d.kind.replace('-','');
    document.getElementById('cid').textContent='call #'+d.seq+'  ·  model='+d.model;
    document.getElementById('status').textContent='';
   }
  }else{
   cur=null;
   document.getElementById('panel').style.display='none';
   document.getElementById('idle').style.display='block';
  }
 }catch(e){}
}
document.getElementById('copy').onclick=()=>{
 const t=document.getElementById('prompt'); t.select(); document.execCommand('copy');
 document.getElementById('status').textContent='copied ✓';
};
document.getElementById('submit').onclick=async()=>{
 if(!cur)return;
 const reply=document.getElementById('reply').value;
 await fetch('/api/answer',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({id:cur,reply:reply})});
 document.getElementById('status').textContent='submitted ✓ — waiting for next call…';
 document.getElementById('panel').style.display='none';
 document.getElementById('idle').style.display='block';
 cur=null;
};
setInterval(poll,1200); poll();
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence per-request logging
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            return self._send(200, _PAGE, "text/html; charset=utf-8")
        if self.path.startswith("/api/pending"):
            with _SLOCK:
                for rid, rec in _STORE.items():
                    if rec.get("reply") is None:
                        return self._send(200, json.dumps(
                            {"id": rid, "prompt": rec["prompt"], "kind": rec["kind"],
                             "seq": rec["seq"], "model": rec["model"]}))
            return self._send(200, "{}")
        if self.path.startswith("/api/result"):
            from urllib.parse import urlparse, parse_qs
            rid = (parse_qs(urlparse(self.path).query).get("id") or [""])[0]
            with _SLOCK:
                rec = _STORE.get(rid)
                if rec and rec.get("reply") is not None:
                    return self._send(200, json.dumps({"ready": True, "reply": rec["reply"]}))
            return self._send(200, json.dumps({"ready": False}))
        return self._send(404, "{}")

    def do_POST(self):
        ln = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(ln) or b"{}")
        if self.path.startswith("/api/request"):
            global _SEQ
            rid = uuid.uuid4().hex
            with _SLOCK:
                _SEQ += 1
                _STORE[rid] = {"prompt": body.get("prompt", ""), "kind": body.get("kind", "TEXT"),
                               "model": body.get("model", "?"), "seq": _SEQ, "reply": None}
            return self._send(200, json.dumps({"id": rid}))
        if self.path.startswith("/api/answer"):
            with _SLOCK:
                rec = _STORE.get(body.get("id"))
                if rec is not None:
                    rec["reply"] = body.get("reply", "")
            return self._send(200, json.dumps({"ok": True}))
        return self._send(404, "{}")


def _ensure_server() -> None:
    """Start the GUI server in-process. The first process to call binds the port;
    the others find it already up and just use HTTP."""
    global _server_started
    with _server_lock:
        if _server_started:
            return
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", _port()), _Handler)
        except OSError:
            _server_started = True  # already running in another process
            return
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        _server_started = True
        print(f"[manual_llm] GUI ready -> http://127.0.0.1:{_port()}", flush=True)


# --------------------------------------------------------------------------- #
# Client side: post the prompt, wait for the form submission
# --------------------------------------------------------------------------- #
def _http(method: str, path: str, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"http://127.0.0.1:{_port()}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _ask_gui(model: str, kind: str, prompt: str) -> str:
    rid = _http("POST", "/api/request", {"prompt": prompt, "kind": kind, "model": model})["id"]
    print(f"[manual_llm] call queued ({kind}) -> answer it at http://127.0.0.1:{_port()}", flush=True)
    while True:
        try:
            res = _http("GET", f"/api/result?id={rid}")
            if res.get("ready"):
                return res.get("reply") or ""
        except Exception:
            pass
        time.sleep(1.0)


# --------------------------------------------------------------------------- #
def _loads(text: str):
    """Tolerant JSON load: strips markdown fences, then tries whole text, then the
    outermost {...} or [...] span. Returns dict/list or None."""
    import re

    if not text:
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    for cand in (t,):
        try:
            return json.loads(cand)
        except Exception:
            pass
    for op, cl in (("{", "}"), ("[", "]")):
        s, e = t.find(op), t.rfind(cl)
        if s != -1 and e > s:
            try:
                return json.loads(t[s : e + 1])
            except Exception:
                pass
    return None


def _norm_calls(raw) -> list:
    """Normalize a list of tool-call dicts (OpenAI or flat form) -> [(name, args_str)]."""
    out = []
    for c in raw or []:
        if not isinstance(c, dict):
            continue
        name = c.get("name") or (c.get("function") or {}).get("name")
        if not name:
            continue
        args = c.get("arguments")
        if args is None and isinstance(c.get("function"), dict):
            args = c["function"].get("arguments")
        if args is None:
            args = c.get("args")
        if isinstance(args, (dict, list)):
            args = json.dumps(args, ensure_ascii=False)
        out.append((str(name), str(args) if args is not None else "{}"))
    return out


def _scan_call_lines(text: str, tool_names: set) -> list:
    """Last-resort: extract `name({...})` / `name {...}` patterns where the braces
    hold valid JSON and the name is a known tool. Balanced-brace aware so nested
    argument objects are captured."""
    import re

    out, i, N = [], 0, len(text)
    pat = re.compile(r"([A-Za-z_][A-Za-z0-9_]{2,})\s*\(?\s*\{")
    while i < N:
        m = pat.search(text, i)
        if not m:
            break
        name = m.group(1)
        s = m.end() - 1  # index of the '{'
        depth, j = 0, s
        while j < N:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            break
        block = text[s : j + 1]
        i = j + 1
        if tool_names and name not in tool_names:
            continue
        try:
            json.loads(block)
        except Exception:
            continue
        out.append((name, block))
    return out


def _extract_calls(text: str, tool_names: set):
    """Return (content, [(name, args_str)]) from a tool-mode reply in any of:
    a JSON object {content, tool_calls}, a single {name, arguments}, a bare
    [ ... ] array, or loose name({...}) lines. Empty list => treat as plain reply."""
    data = _loads(text)
    if isinstance(data, dict):
        content = (data.get("content") or "").strip() or None
        raw = data.get("tool_calls")
        if not raw and (data.get("name") or data.get("function")):
            raw = [data]  # a single tool-call object
        calls = _norm_calls(raw or [])
        if calls or "tool_calls" in data:
            return content, calls
    if isinstance(data, list):
        calls = _norm_calls(data)
        if calls:
            return None, calls
    calls = _scan_call_lines(text, tool_names)
    if calls:
        return None, calls
    return None, []


def _build_response(model: str, text: str, tool_mode: bool, tool_names=None):
    from litellm import ModelResponse, Choices, Message
    from litellm.types.utils import Usage

    content = text or None
    tool_calls = None
    finish = "stop"
    if tool_mode:
        c2, pairs = _extract_calls(text, set(tool_names or []))
        if pairs:
            tool_calls = [
                {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                 "function": {"name": n, "arguments": a or "{}"}}
                for n, a in pairs
            ]
            finish = "tool_calls"
            content = c2
        else:
            content = c2 if c2 is not None else (text or None)
    msg = Message(role="assistant", content=content, tool_calls=tool_calls)
    resp = ModelResponse(
        id="manual-" + uuid.uuid4().hex[:12],
        choices=[Choices(finish_reason=finish, index=0, message=msg)],
        model=model,
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
    resp._hidden_params = {"response_cost": 0.0}
    return resp


def install() -> None:
    """Monkeypatch litellm.completion to route every call through the web GUI when
    MANUAL_LLM is enabled. Idempotent; no-op passthrough when MANUAL_LLM is off."""
    global _orig_completion
    import litellm

    if getattr(litellm, "_manual_patched", False):
        return
    _orig_completion = litellm.completion

    def wrapper(*args, **kwargs):
        if not enabled():
            return _orig_completion(*args, **kwargs)
        _ensure_server()
        model = kwargs.get("model") or (args[0] if args else "?")
        messages = kwargs.get("messages")
        if messages is None and len(args) >= 2:
            messages = args[1]
        tools = kwargs.get("tools")
        response_format = kwargs.get("response_format")
        kind = "TOOL-CALL" if tools else ("JSON" if response_format else "TEXT")
        prompt = _build_prompt(str(model), messages, tools, response_format)
        text = _ask_gui(str(model), kind, prompt)
        tool_names = [(t.get("function") or {}).get("name") for t in (tools or [])]
        return _build_response(str(model), text, tool_mode=bool(tools), tool_names=tool_names)

    litellm.completion = wrapper
    litellm._manual_patched = True
    if enabled():
        _ensure_server()
    print(f"[manual_llm] installed (MANUAL_LLM={'on' if enabled() else 'off'}, web GUI on port {_port()})", flush=True)
