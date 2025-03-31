from typing import Any
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("linear-regression")

if __name__ == "__main__":
    mcp.run()