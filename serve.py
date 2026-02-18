#!/usr/bin/env python3
"""Simple HTTP server serving the output/ directory on port 8888."""

import http.server
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

def main():
    config_path = SCRIPT_DIR / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    port = config["settings"].get("port", 8888)
    output_dir = SCRIPT_DIR / config["settings"]["output_dir"]
    output_dir.mkdir(exist_ok=True)

    os.chdir(output_dir)

    handler = http.server.SimpleHTTPRequestHandler
    server = http.server.HTTPServer(("127.0.0.1", port), handler)
    print(f"Serving {output_dir} on http://localhost:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
