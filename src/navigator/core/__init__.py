"""Transit domain core — the shared, framework-free logic the MCP servers wrap.

Nothing in this package imports LangGraph, MCP, or Flask; it is plain Python so
it can be unit-tested in isolation and reused anywhere.
"""
