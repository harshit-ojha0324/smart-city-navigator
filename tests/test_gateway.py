from navigator.gateway.app import create_app


def _client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_ask_returns_answer():
    r = _client().post("/api/ask", json={"question": "Is the L train running?"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["answer"].startswith("The L train")
    assert body["intent"] == "status"


def test_ask_requires_question():
    r = _client().post("/api/ask", json={})
    assert r.status_code == 400


def test_stream_emits_reasoning_and_done():
    r = _client().get("/api/stream?q=from Penn Station to Fulton St")
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["Content-Type"]
    text = r.get_data(as_text=True)
    assert '"node": "understand"' in text
    assert '"node": "done"' in text
    assert "Take the" in text


def test_index_serves_demo_page():
    r = _client().get("/")
    assert r.status_code == 200 and b"Smart City Navigator" in r.data
