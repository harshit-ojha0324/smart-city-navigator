"""Smart City Navigator — a multi-agent NYC transit planner.

A LangGraph supervisor delegates to specialist worker agents (Route Planner,
Service Advisor, Station Info), each reasoning over MCP tool servers (live
GTFS-RT alerts, Dijkstra route planning, geocoding) to answer natural-language
transit questions, streamed to clients through a Flask Server-Sent-Events gateway.
"""

__version__ = "0.1.0"
