"""
Flask API gateway with Server-Sent Events streaming.

Endpoints
  GET  /                 responsive trip planner with streamed results
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

from flask import Flask, Response, jsonify, render_template, request

from navigator.agent.graph import run_once, stream_once
from navigator.core.mta_feed import SUBWAY_LINE_IDS

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
                        # Same contract as /api/ask: the type, never the message.
                        app.logger.exception("agent stream failed")
                        q.put({"node": "error", "text": f"agent unavailable ({type(exc).__name__})"})
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
        return render_template("index.html", lines=SUBWAY_LINE_IDS)

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
