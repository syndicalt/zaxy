"""Minimal OpenAI-compatible /v1/chat/completions shim backed by `codex exec`.

Lets the LongMemEval harness (reader) and the official evaluate_qa.py judge use
the Codex CLI (authenticated via the user's ChatGPT/Codex plan) instead of a
funded OpenAI API key. Embeddings are NOT supported (Codex can't embed) — the
harness must use hash embeddings for retrieval.

Each chat request flattens its messages into a single prompt, runs
`codex exec -o <file>` in a read-only sandbox, and returns the captured final
message in OpenAI response shape. Slow (~10-30s/call) but real.

Run:  python scripts/codex_openai_shim.py --port 8899 [--model <codex-model>]
Point the harness/judge at base_url=http://127.0.0.1:8899/v1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = None  # set from argv; None -> codex default


def _flatten(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):  # tolerate content-parts form
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        parts.append(f"[{role}]\n{content}")
    parts.append(
        "\n[system]\nRespond with ONLY the answer text requested above. "
        "No preamble, no explanation, no tool use, no file access."
    )
    return "\n\n".join(parts)


def _codex_complete(prompt: str) -> str:
    with tempfile.NamedTemporaryFile("r", suffix=".txt", delete=False) as fh:
        out_path = fh.name
    cmd = [
        "codex", "exec",
        "-s", "read-only",
        "--skip-git-repo-check",
        "-c", "hooks={}",  # avoid repo SessionStart/Stop hooks per call
        "-o", out_path,
    ]
    if MODEL:
        cmd += ["-m", MODEL]
    cmd.append(prompt)
    subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    try:
        with open(out_path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404, "only /v1/chat/completions is supported")
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        prompt = _flatten(body.get("messages", []))
        _t = time.time()
        try:
            answer = _codex_complete(prompt)
        except subprocess.TimeoutExpired:
            answer = ""
        resp = {
            "id": f"chatcmpl-codex-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", MODEL or "codex"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        print(f"[shim] codex call {time.time()-_t:.1f}s -> {answer[:60]!r}", flush=True)
        payload = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    MODEL = args.model
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"codex shim on http://127.0.0.1:{args.port}/v1 (model={MODEL or 'codex-default'})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
