"""
LLM factory + deterministic offline planner.

Two ways the agent can reason:

  * Gemini (langchain-google-genai) when GOOGLE_API_KEY / GEMINI_API_KEY is set —
    the LLM does tool-calling itself.
  * A deterministic planner otherwise — regex intent + entity extraction that
    emits the same tool calls, so the graph, the MCP servers, and the 20-prompt
    eval suite all run end-to-end in CI with no API key and no flakiness.

Both drive the identical graph topology and the identical MCP tools; only the
"which tool, which args" decision differs.
"""
from __future__ import annotations

import os
import re

MODEL_NAME = os.environ.get("NAVIGATOR_MODEL", "gemini-2.5-flash")

# Subway line tokens, longest first so "SIR" would beat "S" if added later.
_LINE_IDS = ["1", "2", "3", "4", "5", "6", "7", "A", "C", "E", "B", "D",
             "F", "M", "G", "J", "Z", "L", "N", "Q", "R", "W", "S"]

# Order matters — most specific phrasings first.
_ROUTE_PATTERNS = [
    ("from_to", re.compile(r"from\s+(?P<o>.+?)\s+to\s+(?P<d>.+?)(?:[.?!]|$)", re.I)),
    ("to_from", re.compile(r"\bto\s+(?P<d>.+?)\s+from\s+(?P<o>.+?)(?:[.?!]|$)", re.I)),
    ("verb_to", re.compile(r"(?:get|go|travel|commute|route)\s+(?:from\s+)?(?P<o>.+?)\s+to\s+(?P<d>.+?)(?:[.?!]|$)", re.I)),
    ("bare_to", re.compile(r"(?P<o>.+?)\s+to\s+(?P<d>.+?)(?:[.?!]|$)", re.I)),
]

# Leading interrogative stems stripped off a captured origin/destination.
_LEADIN = re.compile(
    r"^(?:how\s+(?:do|can|would|should)\s+i\s+(?:get|go|travel|commute)|"
    r"what['’]?s?\s+the\s+(?:best|fastest|quickest)\s+(?:way|route)|"
    r"i\s+(?:want|need)\s+to\s+(?:get|go)|can\s+i\s+get|get\s+me|take\s+me|"
    r"how\s+to\s+(?:get|go)|directions?)\b",
    re.I,
)

_STATUS_WORDS = re.compile(
    r"\b(delay|delays|delayed|status|running|service|suspend|suspended|work|"
    r"alert|alerts|disrupt|disrupted|slow|good service|on time)\b",
    re.I,
)
_INFO_WORDS = re.compile(r"\b(which lines?|what lines?|stops? at|serve|near|nearest|closest)\b", re.I)
_NEAR_WORDS = re.compile(r"\b(nearest|closest)\b", re.I)
_TRANSIT_WORDS = re.compile(r"\b(subway|train|station|mta|line|transfer|stop|route|ride|commute)\b", re.I)


def get_chat_model():
    """Return a bound-capable chat model, or None to use the deterministic planner."""
    key = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception:
        return None
    os.environ.setdefault("GOOGLE_API_KEY", key)
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


# ── Deterministic planner ──────────────────────────────────────────────
class DeterministicPlanner:
    """Regex-based intent + entity extraction used when no LLM key is present."""

    def classify(self, question: str) -> str:
        q = question.strip()
        # "nearest station to X" is a lookup, not a trip — check before route
        # extraction, whose bare "... to ..." pattern would otherwise grab it.
        if _NEAR_WORDS.search(q):
            return "info"
        if self._extract_od(q):
            return "route"
        if _STATUS_WORDS.search(q):
            return "status"
        if _INFO_WORDS.search(q):
            return "info"
        if _TRANSIT_WORDS.search(q):
            return "status"  # generic transit → show overall status
        return "other"

    def _extract_od(self, question: str) -> tuple[str, str] | None:
        for _label, pat in _ROUTE_PATTERNS:
            m = pat.search(question)
            if m:
                o = _clean_place(m.group("o"))
                d = _clean_place(m.group("d"))
                # Allow equal endpoints — plan_route answers "already there"
                # rather than letting a greedy fallback grab "from <origin>".
                if o and d:
                    return o, d
        return None

    def _extract_line(self, question: str) -> str | None:
        """Extract a valid line only when unambiguous.

        Requires a "<line> train/line" adjacency for letters (so the "s" in
        "What's" is never read as the S train); a bare digit 1-7 is safe alone.
        """
        m = re.search(r"\b([1-7A-Za-z])\s+(?:train|line)\b", question, re.I)
        if m and m.group(1).upper() in _LINE_IDS:
            return m.group(1).upper()
        m = re.search(r"\b(?:train|line)\s+([1-7A-Za-z])\b", question, re.I)
        if m and m.group(1).upper() in _LINE_IDS:
            return m.group(1).upper()
        m = re.search(r"\b([1-7])\b", question)  # standalone digit line
        if m:
            return m.group(1)
        return None

    def _extract_line_token(self, question: str) -> str | None:
        """Any 1-3 char token named as a train/line, valid or not.

        Lets "Is the QZ train running?" call the tool with "QZ" so it returns a
        clean "unknown line" error instead of silently showing overall status.
        """
        m = re.search(r"\b([A-Za-z0-9]{1,3})\s+(?:train|line)\b", question, re.I)
        if m:
            return m.group(1).upper()
        m = re.search(r"\b(?:train|line)\s+([A-Za-z0-9]{1,3})\b", question, re.I)
        if m:
            return m.group(1).upper()
        return None

    def initial_tool_calls(self, question: str, intent: str) -> list[dict]:
        """Tool calls to make on the first plan pass (empty → straight to synth)."""
        if intent == "route":
            od = self._extract_od(question)
            if od:
                return [_call("plan_trip", {"origin": od[0], "destination": od[1]})]
            return []
        if intent == "status":
            line = self._extract_line(question)
            if line:
                return [_call("get_line_status", {"line": line})]
            token = self._extract_line_token(question)  # e.g. "QZ" — invalid line
            if token:
                return [_call("get_line_status", {"line": token})]
            return [_call("get_service_status", {})]
        if intent == "info":
            # "nearest/closest ... to X" → geocode X to its nearest station.
            if _NEAR_WORDS.search(question):
                m = re.search(r"(?:nearest|closest)\s+(?:station\s+)?(?:to\s+)?(?P<p>.+?)(?:[.?!]|$)",
                              question, re.I)
                place = _clean_place(m.group("p")) if m else question
                return [_call("geocode_place", {"query": place})]
            # "which lines stop at X" → resolve the station itself.
            m = re.search(r"(?:stops? at|serve[s]?)\s+(?P<p>.+?)(?:[.?!]|$)", question, re.I)
            place = _clean_place(m.group("p")) if m else question
            return [_call("resolve_station", {"query": place})]
        return []  # 'other' → no tools, synth issues a polite redirect


_CALL_SEQ = {"n": 0}


def _call(name: str, args: dict) -> dict:
    _CALL_SEQ["n"] += 1
    return {"name": name, "args": args, "id": f"det-{name}-{_CALL_SEQ['n']}", "type": "tool_call"}


def _clean_place(text: str) -> str:
    text = text.strip().strip(",.?! ")
    # Drop a trailing conjunction clause so compound questions
    # ("... to Fulton St and are there delays?") still resolve the place.
    text = re.split(r"\s+(?:and|but|also|plus|,|;)\s+", text, maxsplit=1)[0]
    text = _LEADIN.sub("", text).strip()
    text = re.sub(r"^(from|to)\s+", "", text, flags=re.I)
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.I)
    text = re.sub(r"\s+(please|now|today|right now)$", "", text, flags=re.I)
    return text.strip()
