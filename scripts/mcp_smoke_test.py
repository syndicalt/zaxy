"""MCP client smoke test for Zaxy server.

Spawns `zaxy serve` as a subprocess and speaks MCP protocol over stdio.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile


def send(stdin, stdout, method: str, params: dict | None = None, msg_id: int = 1) -> dict:
    """Send an MCP JSON-RPC request and return the response."""
    req = {
        "jsonrpc": "2.0",
        "id": msg_id,
        "method": method,
    }
    if params is not None:
        req["params"] = params

    line = json.dumps(req) + "\n"
    stdin.write(line.encode())
    stdin.flush()

    resp_line = stdout.readline().decode().strip()
    return json.loads(resp_line)


def main() -> int:
    """Run MCP smoke test against Zaxy server."""
    print("🚀 Starting zaxy serve...")

    with tempfile.TemporaryDirectory(prefix="zaxy-mcp-smoke-") as temp_eventloom:
        proc = subprocess.Popen(
            [sys.executable, "-m", "zaxy", "serve", "--eventloom-path", temp_eventloom],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            # 1. Initialize
            print("📡 Sending initialize...")
            resp = send(
                proc.stdin, proc.stdout,
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "zaxy-smoke-test", "version": "0.1.0"},
                },
            )
            if "error" in resp:
                print(f"❌ Initialize failed: {resp['error']}")
                return 1
            print(f"✅ Server: {resp['result']['serverInfo']['name']} {resp['result']['serverInfo']['version']}")

            # 2. List tools
            print("📋 Listing tools...")
            resp = send(proc.stdin, proc.stdout, "tools/list", {}, msg_id=2)
            if "error" in resp:
                print(f"❌ tools/list failed: {resp['error']}")
                return 1

            tools = resp["result"]["tools"]
            print(f"✅ Exposed {len(tools)} tools:")
            for t in tools:
                print(f"   • {t['name']}: {t.get('description', 'no description')}")

            # 3. Call memory_append
            print("📝 Calling memory_append...")
            resp = send(
                proc.stdin, proc.stdout,
                "tools/call",
                {
                    "name": "memory_append",
                    "arguments": {
                        "event_type": "goal.created",
                        "actor": "smoke_test",
                        "payload": {"title": "Verify MCP works"},
                    },
                },
                msg_id=3,
            )
            if "error" in resp:
                print(f"❌ memory_append failed: {resp['error']}")
                return 1
            content = resp["result"]["content"][0]["text"]
            print(f"✅ memory_append result: {content}")

            # 4. Call memory_query
            print("🔍 Calling memory_query...")
            resp = send(
                proc.stdin, proc.stdout,
                "tools/call",
                {
                    "name": "memory_query",
                    "arguments": {"query": "Verify MCP works"},
                },
                msg_id=4,
            )
            if "error" in resp:
                print(f"❌ memory_query failed: {resp['error']}")
                return 1
            content = resp["result"]["content"][0]["text"]
            print(f"✅ memory_query result: {content[:200]}...")

            print("\n🎉 All MCP smoke tests passed!")
            return 0

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
