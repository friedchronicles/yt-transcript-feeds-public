#!/usr/bin/env python3
"""Build a simple index page listing feed URLs."""

from pathlib import Path
import json

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
OUTPUT_DIR = SCRIPT_DIR / "output"


def main():
    config = json.loads(CONFIG_PATH.read_text())
    base = config["settings"].get("public_base_url", "").rstrip("/")
    rows = []
    for ch in config["channels"]:
        slug = ch["slug"]
        name = ch["name"]
        url = f"{base}/{slug}.xml" if base else f"./{slug}.xml"
        rows.append(f"<li><a href=\"{url}\">{name}</a></li>")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YouTube Transcript Feeds</title>
</head>
<body>
  <h1>YouTube Transcript Feeds</h1>
  <p>Subscribe to these RSS URLs in NetNewsWire:</p>
  <ul>
    {"".join(rows)}
  </ul>
</body>
</html>
"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "index.html").write_text(html, encoding="utf-8")
    (OUTPUT_DIR / ".nojekyll").write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
