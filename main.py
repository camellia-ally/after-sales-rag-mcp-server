"""Console entry point for the MCP stdio server.

Stdout is reserved for MCP protocol messages.
"""

import sys


def main() -> int:
    """Start the actual stdio server used by the console script."""
    from src.mcp_server.server import main as run_server

    return run_server()


if __name__ == "__main__":
    sys.exit(main())
