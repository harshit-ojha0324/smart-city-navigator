"""
LLM factory + deterministic offline planner.

Two ways the agent can reason, and the choice is one env var:

  * A tool-calling chat model — Gemini, any OpenAI-compatible endpoint, or a
    local Ollama model. It classifies the question, picks the tools, and writes
    the answer.
  * A deterministic planner when no model is configured — regex intent + entity
    extraction emitting the same tool calls, so the graph, the MCP servers and
    the eval suite run end-to-end in CI with no key, no network and no
    flakiness.

Both drive the identical graph topology and the identical MCP tools; only the
"which tool, which args" decision differs. `describe_reasoner()` names whichever
is active, so an eval score can never be ambiguous about what produced it.

    NAVIGATOR_LLM_PROVIDER = auto (default) | gemini | ollama | openai | none
    NAVIGATOR_MODEL        = model id for the chosen provider
    NAVIGATOR_LLM_BASE_URL = endpoint for ollama / OpenAI-compatible servers
"""
from __future__ import annotations

import os
import re

_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "ollama": "qwen3:8b",
    "openai": "gpt-4o-mini",
}

# Subway line tokens, longest first so "SIR" would beat "S" if added later.
_LINE_IDS = ["1", "2", "3", "4", "5", "6", "7", "A", "C", "E", "B", "D",
             "F", "M", "G", "J", "Z", "L", "N", "Q", "R", "W", "S"]

# Order matters — most specific phrasings first.
_ROUTE_PATTERNS = [
    # "I'm at Union Square, how do I get to Harlem?" — origin stated up front.
    ("at_to", re.compile(r"\bi['’]?m\s+(?:at|near|in)\s+(?P<o>.+?)(?:\s*[,;]|\s+and\b)"
                         r".*?\b(?:to|reach|for)\s+(?P<d>.+?)(?:[.?!]|$)", re.I)),
    ("from_to", re.compile(r"from\s+(?P<o>.+?)\s+to\s+(?P<d>.+?)(?:[.?!]|$)", re.I)),
    ("to_from", re.compile(r"\bto\s+(?P<d>.+?)\s+from\s+(?P<o>.+?)(?:[.?!]|$)", re.I)),
    ("verb_to", re.compile(r"(?:get|go|travel|commute|route)\s+(?:from\s+)?(?P<o>.+?)"
                           r"\s+to\s+(?P<d>.+?)(?:[.?!]|$)", re.I)),
    ("bare_to", re.compile(r"(?P<o>.+?)\s+to\s+(?P<d>.+?)(?:[.?!]|$)", re.I)),
]

# Leading interrogative stems stripped off a captured origin/destination.
_LEADIN = re.compile(
    r"^(?:how\s+(?:do|can|would|should)\s+i\s+(?:get|go|travel|commute|reach)|"
    r"what['’]?s?\s+the\s+(?:best|fastest|quickest)\s+(?:way|route)|"
    r"what['’]?s?\s+the\s+(?:ride|travel|trip)\s+time|how\s+long|"
    r"i\s+(?:want|need)\s+to\s+(?:get|go|reach)|can\s+i\s+get|get\s+me|take\s+me|"
    r"can\s+you\s+route\s+me|route\s+me|how\s+to\s+(?:get|go)|directions?)\b",
    re.I,
)

# How riders actually ask about service: not just "status" and "delays" but
# "any problems on the J?" and "trains messed up on the 6?".
_STATUS_WORDS = re.compile(
    r"\b(delay|delays|delayed|status|running|runs|service|suspend|suspended|work|"
    r"alert|alerts|disrupt|disrupted|slow|good service|on time|problems?|issues?|"
    r"trouble|messed up|screwed up|down|stuck|late|ok|okay|fine)\b",
    re.I,
)
# A trip request with no origin ("How do I get to Coney Island?") — still a route
# question; the Route Planner asks where the rider is starting from.
_DEST_ONLY = re.compile(
    r"\b(?:get|go|travel|commute|head|directions?|take\s+me|get\s+me)\s+to\s+(?P<d>.+?)(?:[.?!]|$)"
    r"|\breach\s+(?P<d2>.+?)(?:[.?!]|$)", re.I)
# Question stems stripped from a station lookup ("what lines are at ...").
_INFO_LEADIN = re.compile(
    r"^(?:which|what)\s+(?:lines?|trains?)\s+(?:are|is|stop|stops|serve|serves|can\s+i\s+catch)?\s*",
    re.I,
)
# "get me to Wall St" must ask where the rider is, not plan a trip from "me".
_NOT_A_PLACE = {"me", "us", "i", "you", "myself", "there", "here", "it", "them"}
_ACCESS_WORDS = re.compile(
    r"\b(elevators?|escalators?|outages?|accessib\w*|wheelchair|ada)\b", re.I)
_INFO_WORDS = re.compile(
    r"\b(which lines?|what lines?|which trains?|what trains?|lines? (?:are |at)|"
    r"trains? at|show me the (?:trains?|lines?)|stops? at|catch at|serves?|"
    r"near|nearest|closest)\b", re.I)
_NEAR_WORDS = re.compile(r"\b(nearest|closest)\b", re.I)
_TRANSIT_WORDS = re.compile(
    r"\b(subway|trains?|stations?|mta|lines?|transfers?|stops?|routes?|ride|commute)\b", re.I)


def _gemini_key() -> str:
    return (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()


def active_provider() -> str:
    """Which reasoner this process will use, resolving "auto"."""
    choice = os.environ.get("NAVIGATOR_LLM_PROVIDER", "auto").strip().lower()
    if choice != "auto":
        return choice
    if _gemini_key():
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "none"


def model_name(provider: str | None = None) -> str:
    provider = provider or active_provider()
    return os.environ.get("NAVIGATOR_MODEL") or _DEFAULT_MODELS.get(provider, "")


def describe_reasoner() -> str:
    """Human label for reports: the model in use, or the offline planner."""
    provider = active_provider()
    if provider in ("none", ""):
        return "deterministic planner (no LLM configured)"
    return f"{provider}:{model_name(provider)}"


def prompt_suffix() -> str:
    """Qwen3 and friends emit a long <think> block by default, which costs
    tens of seconds per call and confuses tool parsing; "/no_think" turns it
    off. Harmless to any model that doesn't recognise it."""
    if active_provider() == "ollama" and "qwen3" in model_name("ollama"):
        return " /no_think"
    return ""


def get_chat_model():
    """Return a tool-calling chat model, or None to use the deterministic planner.

    Any import or construction failure falls back to None rather than breaking
    the run: a missing optional dependency should degrade to the offline
    planner, not take the agent down.
    """
    provider = active_provider()
    temperature = float(os.environ.get("NAVIGATOR_TEMPERATURE", "0"))
    try:
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            key = _gemini_key()
            if not key:
                return None
            os.environ.setdefault("GOOGLE_API_KEY", key)
            return ChatGoogleGenerativeAI(model=model_name(provider), temperature=temperature)

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=model_name(provider), temperature=temperature,
                # `or` not a get() default: compose passes the var through as
                # an empty string when the host hasn't set it.
                base_url=os.environ.get("NAVIGATOR_LLM_BASE_URL") or "http://127.0.0.1:11434",
                # A local model has to hold the tool schemas plus the question;
                # Ollama's 2k default silently truncates them.
                num_ctx=int(os.environ.get("NAVIGATOR_NUM_CTX", "8192")),
            )

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=model_name(provider), temperature=temperature,
                              base_url=os.environ.get("NAVIGATOR_LLM_BASE_URL") or None)
    except Exception as exc:  # pragma: no cover - depends on optional extras
        print(f"[llm] {provider} unavailable ({exc}) — using the deterministic planner")
    return None


# ── Deterministic planner ──────────────────────────────────────────────
class DeterministicPlanner:
    """Regex-based intent + entity extraction used when no LLM key is present."""

    def classify(self, question: str) -> str:
        q = question.strip()
        # "nearest station to X" is a lookup, not a trip — check before route
        # extraction, whose bare "... to ..." pattern would otherwise grab it.
        if _NEAR_WORDS.search(q):
            return "info"
        if _ACCESS_WORDS.search(q):
            return "status"  # elevator/escalator outages are the Service Advisor's
        if self._extract_od(q) or _DEST_ONLY.search(q):
            return "route"
        if _STATUS_WORDS.search(q):
            return "status"
        if _INFO_WORDS.search(q):
            return "info"
        if self._extract_line(q):
            return "status"  # "what's the deal with the 7 today" names a line
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
                if o and d and o.lower() not in _NOT_A_PLACE and d.lower() not in _NOT_A_PLACE:
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
        # "Is the Q running?" / "delays on the F" — a capital letter right after
        # "the/on" (case-sensitive, so "the a..." in prose never matches).
        m = re.search(r"\b(?:the|on)\s+([A-Z])\b(?!['’])", question)
        if m and m.group(1) in _LINE_IDS:
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

    def place_in(self, question: str) -> str:
        """The station or place a lookup question is about.

        Tried most specific first: an explicit "stops at X" phrasing, then a
        trailing "at X", then whatever is left after stripping the question
        stem — "what lines are at 86 St" must ask about "86 St", not about the
        whole sentence.
        """
        for pattern in (
            r"(?:stops? at|stop at|serves?|catch at|catch)\s+(?P<p>.+?)(?:[.?!]|$)",
            r"\b(?:at|in|for|to)\s+(?P<p>.+?)(?:[.?!]|$)",
        ):
            m = re.search(pattern, question, re.I)
            if m:
                place = _clean_place(m.group("p"))
                if place:
                    return place
        return _clean_place(_INFO_LEADIN.sub("", question))

    def destination_only(self, question: str) -> str | None:
        """The destination of a trip request that names no origin."""
        if self._extract_od(question):
            return None
        m = _DEST_ONLY.search(question)
        if not m:
            return None
        return _clean_place(m.group("d") or m.group("d2") or "")

    def initial_tool_calls(self, question: str, intent: str,
                           context_lines: list[str] | None = None) -> list[dict]:
        """Tool calls to make on the first plan pass (empty → straight to synth).

        `context_lines` is handed over by the supervisor when another agent has
        already planned a route: the Service Advisor then checks exactly the
        lines the rider will use instead of dumping the whole system status.
        """
        if intent == "route":
            od = self._extract_od(question)
            if od:
                return [_call("plan_trip", {"origin": od[0], "destination": od[1]})]
            return []
        if intent == "status" and context_lines:
            return [_call("get_line_status", {"line": ln}) for ln in context_lines]
        if intent == "status" and _ACCESS_WORDS.search(question):
            m = re.search(r"\b(?:at|in|for|near)\s+(?P<p>.+?)(?:[.?!]|$)", question, re.I)
            station = _clean_place(m.group("p")) if m else ""
            return [_call("list_elevator_outages", {"station_contains": station})]
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
            # "which lines stop at X" / "what trains can I catch at X" →
            # resolve the station itself.
            return [_call("resolve_station", {"query": self.place_in(question)})]
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
    # "route me: Marcy Av" — the verb pattern eats "route", leaving the pronoun.
    text = re.sub(r"^(?:me|us|i|you)\b\s*[:,\-]?\s*", "", text, flags=re.I)
    text = re.sub(r"^(from|to)\s+", "", text, flags=re.I)
    # "...way to get to Coney Island" leaves "get to Coney Island" behind, and
    # "need to reach Barclays Center" leaves the verb on the front too.
    # \b and a required space: without them "be" eats the start of "Bedford Av".
    text = re.sub(r"^(?:get|go|head|travel|ride|reach|meet|be|arrive|take\s+me)\b"
                  r"\s+(?:to\s+|at\s+|down\s+to\s+)?", "", text, flags=re.I)
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.I)
    text = re.sub(r"\s+(please|now|today|right now)$", "", text, flags=re.I)
    # Stripping a lead-in ("can you route me:") can leave its punctuation behind.
    return text.strip(" \t:;,-–—")
