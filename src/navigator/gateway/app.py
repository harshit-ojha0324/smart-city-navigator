"""
Flask API gateway with Server-Sent Events streaming.

Endpoints
  GET  /                 tiny demo page that streams the agent's reasoning live
  GET  /health           liveness + which tool transport is active
  POST /api/ask          {question, thread_id?} -> {answer, intent, steps}
  GET  /api/stream?q=..  text/event-stream of reasoning steps, then the answer

The agent stream is async (LangGraph astream); each request bridges it to a
synchronous SSE generator via a background thread + queue so reasoning is pushed
to the browser the instant each node emits it.
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import uuid

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from navigator.agent.graph import run_once, stream_once

TRANSPORT = os.environ.get("NAVIGATOR_TRANSPORT", "inprocess")
CHECKPOINT_PATH = os.environ.get("NAVIGATOR_CHECKPOINT", ":memory:")
MAX_QUESTION_CHARS = 500


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _read_question(raw) -> tuple[str | None, tuple | None]:
    """Validated question text, or (None, error response)."""
    question = (raw or "").strip()
    if not question:
        return None, (jsonify({"error": "question is required"}), 400)
    if len(question) > MAX_QUESTION_CHARS:
        return None, (jsonify({"error": f"question exceeds {MAX_QUESTION_CHARS} characters"}), 413)
    return question, None


def _thread_id(raw) -> str:
    """Caller's conversation id, else a fresh one — never a shared default, so
    two users can't land on the same checkpointed thread."""
    return (raw or "").strip()[:128] or uuid.uuid4().hex


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "transport": TRANSPORT,
                        "checkpoint": CHECKPOINT_PATH})

    @app.post("/api/ask")
    def ask():
        data = request.get_json(silent=True) or {}
        question, err = _read_question(data.get("question"))
        if err:
            return err
        thread_id = _thread_id(data.get("thread_id"))
        try:
            state = asyncio.run(run_once(
                question, thread_id=thread_id, transport=TRANSPORT,
                checkpoint_path=CHECKPOINT_PATH,
            ))
        except Exception as exc:  # e.g. MCP servers down — answer with JSON, not a stack page
            app.logger.exception("agent run failed")
            return jsonify({"error": "agent unavailable", "detail": type(exc).__name__}), 503
        return jsonify({
            "answer": state.get("answer", ""),
            "intent": state.get("intent", ""),
            "steps": state.get("steps", []),
            "thread_id": thread_id,
        })

    @app.get("/api/stream")
    def stream():
        question, err = _read_question(request.args.get("q"))
        if err:
            return err
        thread_id = _thread_id(request.args.get("thread_id"))

        def generate():
            q: queue.Queue = queue.Queue()
            sentinel = object()

            def producer():
                async def run():
                    try:
                        async for ev in stream_once(
                            question, thread_id=thread_id, transport=TRANSPORT,
                            checkpoint_path=CHECKPOINT_PATH,
                        ):
                            q.put(ev)
                    except Exception as exc:  # surface, don't hang the stream
                        q.put({"node": "error", "text": str(exc)})
                    finally:
                        q.put(sentinel)

                asyncio.run(run())

            threading.Thread(target=producer, daemon=True).start()
            while True:
                ev = q.get()
                if ev is sentinel:
                    break
                yield _sse(ev)

        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/")
    def index():
        return Response(_DEMO_HTML, mimetype="text/html")

    return app


_DEMO_HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>Smart City Navigator</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
 body{font-family:-apple-system,system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#111;background:#fafafa}
 h1{font-size:1.4rem}.sub{color:#666;margin-top:-.6rem}
 form{display:flex;gap:.5rem;margin:1rem 0}
 input{flex:1;padding:.6rem;border:1px solid #ccc;border-radius:8px;font-size:1rem}
 button{padding:.6rem 1rem;border:0;border-radius:8px;background:#0039A6;color:#fff;font-size:1rem;cursor:pointer}
 .steps{font-family:ui-monospace,monospace;font-size:.85rem;color:#555;background:#fff;border:1px solid #eee;border-radius:8px;padding:.6rem;min-height:2rem;white-space:pre-wrap}
 .answer{margin-top:1rem;padding:1rem;background:#EAF0FF;border-radius:8px;font-size:1.05rem;white-space:pre-wrap}
 .node{color:#0039A6;font-weight:600}
</style></head><body>
<h1>🚇 Smart City Navigator</h1>
<p class=sub>Multi-agent NYC transit planner — watch the supervisor delegate to specialist agents live.</p>
<form id=f><input id=q placeholder="How do I get from Times Square to Coney Island?" autofocus>
<button>Ask</button></form>
<div class=steps id=steps>Ask a question to see the agent's reasoning…</div>
<div class=answer id=answer hidden></div>
<script>
const f=document.getElementById('f'),q=document.getElementById('q'),
 steps=document.getElementById('steps'),answer=document.getElementById('answer');
f.onsubmit=e=>{e.preventDefault();if(!q.value.trim())return;
 steps.textContent='';answer.hidden=true;answer.textContent='';
 const es=new EventSource('/api/stream?q='+encodeURIComponent(q.value));
 es.onmessage=ev=>{const d=JSON.parse(ev.data);
  if(d.node==='done'){answer.hidden=false;answer.textContent=d.answer;es.close();return;}
  // textContent only: step text echoes user input and tool output (no HTML injection)
  const node=document.createElement('span');node.className='node';node.textContent=`[${d.node}] `;
  steps.append(node,(d.text||'')+'\\n');};
 es.onerror=()=>es.close();};
</script></body></html>"""


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
