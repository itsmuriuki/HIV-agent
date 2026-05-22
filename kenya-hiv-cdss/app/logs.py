"""Minimal logging helpers for the Streamlit app.

This module is intentionally lightweight: logging is optional.
"""


def log_interaction_to_file(agent, messages):
    """Optionally log agent interactions. No-op by default."""
    return


def print_recent_logs(n: int = 5):
    """Optionally show recent logs. No-op by default."""
    return
