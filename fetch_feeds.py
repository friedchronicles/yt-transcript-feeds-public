#!/usr/bin/env python3
"""Fetch YouTube channel videos, grab transcripts, and generate RSS feeds."""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser
from feedgen.feed import FeedGenerator
from youtube_transcript_api import YouTubeTranscriptApi

# Setup paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

# Logging
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fetch.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_seen_videos(data_dir: Path) -> dict:
    path = data_dir / "seen_videos.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_seen_videos(data_dir: Path, seen: dict):
    path = data_dir / "seen_videos.json"
    with open(path, "w") as f:
        json.dump(seen, f, indent=2)


def load_cleanup_state(data_dir: Path) -> dict:
    path = data_dir / "cleanup_state.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"cutoff_utc": None, "cleaned_video_ids": []}


def save_cleanup_state(data_dir: Path, state: dict):
    path = data_dir / "cleanup_state.json"
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def parse_yt_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def should_run_cleanup(video: dict, settings: dict, cleanup_state: dict) -> bool:
    if not settings.get("enable_cleanup", False):
        return False
    if video["video_id"] in set(cleanup_state.get("cleaned_video_ids", [])):
        return False
    if not settings.get("cleanup_only_new_videos", True):
        return True

    cutoff_raw = cleanup_state.get("cutoff_utc")
    cutoff = parse_yt_datetime(cutoff_raw) if cutoff_raw else None
    if cutoff is None:
        return False

    published = parse_yt_datetime(video.get("published", ""))
    if published is None:
        return False
    return published > cutoff


def maybe_run_cleanup(entry: dict, settings: dict) -> str:
    """Optionally run external cleanup command on transcript HTML.

    The command receives JSON on stdin and must return cleaned HTML on stdout.
    """
    command = settings.get("cleanup_command", "").strip()
    if not command:
        return entry["transcript_html"]

    payload = json.dumps(entry).encode("utf-8")
    proc = subprocess.run(
        command,
        input=payload,
        capture_output=True,
        shell=True,
        check=False,
    )
    if proc.returncode != 0:
        log.warning("Cleanup command failed for %s: %s", entry["video_id"], proc.stderr.decode("utf-8", errors="replace"))
        return entry["transcript_html"]

    cleaned = proc.stdout.decode("utf-8", errors="replace").strip()
    return cleaned or entry["transcript_html"]


def fetch_channel_feed(channel_id: str) -> list[dict]:
    """Fetch the YouTube Atom feed for a channel and return video entries."""
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    feed = feedparser.parse(url)
    videos = []
    for entry in feed.entries:
        video_id = entry.get("yt_videoid", "")
        if not video_id:
            # Try to extract from link
            link = entry.get("link", "")
            if "v=" in link:
                video_id = link.split("v=")[-1].split("&")[0]
        if not video_id:
            continue
        videos.append({
            "video_id": video_id,
            "title": entry.get("title", "Untitled"),
            "link": entry.get("link", f"https://www.youtube.com/watch?v={video_id}"),
            "published": entry.get("published", ""),
            "summary": entry.get("summary", ""),
        })
    return videos


def video_matches_channel_filters(channel: dict, video: dict) -> bool:
    """Return True if a video passes optional per-channel filters."""
    prefixes = channel.get("include_title_prefixes", [])
    if prefixes:
        title_lower = (video.get("title") or "").strip().lower()
        allowed = any(title_lower.startswith(p.strip().lower()) for p in prefixes if p.strip())
        if not allowed:
            return False
    return True


_transcript_api = YouTubeTranscriptApi()


def fetch_transcript(video_id: str) -> list[dict] | None:
    """Fetch transcript for a video. Returns list of segment dicts or None."""
    try:
        result = _transcript_api.fetch(video_id, languages=["en"])
        # Convert FetchedTranscriptSnippet objects to plain dicts
        return [{"text": s.text, "start": s.start, "duration": s.duration} for s in result]
    except Exception as e:
        log.warning("No transcript for %s: %s", video_id, e)
        return None


def format_transcript_html(segments: list[dict]) -> str:
    """Format transcript segments into readable HTML paragraphs.

    Groups segments into ~30-second chunks for paragraphs.
    Adds timestamp markers every ~3 minutes for navigation.
    """
    if not segments:
        return "<p><em>No transcript available.</em></p>"

    html_parts = []
    current_paragraph = []
    paragraph_start_time = 0.0
    last_timestamp_marker = -180.0  # Start so first marker shows at 0:00

    for seg in segments:
        start = seg["start"]
        text = seg["text"].strip()
        if not text:
            continue

        # Add timestamp marker every ~3 minutes
        if start - last_timestamp_marker >= 180.0:
            if current_paragraph:
                html_parts.append("<p>" + " ".join(current_paragraph) + "</p>")
                current_paragraph = []
            minutes = int(start) // 60
            seconds = int(start) % 60
            marker = f"<p><strong>[{minutes:02d}:{seconds:02d}]</strong></p>"
            html_parts.append(marker)
            last_timestamp_marker = start
            paragraph_start_time = start

        current_paragraph.append(text)

        # Start new paragraph every ~30 seconds
        if start - paragraph_start_time >= 30.0:
            html_parts.append("<p>" + " ".join(current_paragraph) + "</p>")
            current_paragraph = []
            paragraph_start_time = start

    # Flush remaining
    if current_paragraph:
        html_parts.append("<p>" + " ".join(current_paragraph) + "</p>")

    return "\n".join(html_parts)


def generate_feed(channel_name: str, slug: str, videos_with_transcripts: list[dict], output_dir: Path, base_url: str):
    """Generate an RSS XML feed file for a channel."""
    fg = FeedGenerator()
    fg.title(f"Transcripts: {channel_name}")
    fg.link(href=f"{base_url.rstrip('/')}/{slug}.xml", rel="self")
    fg.description(f"YouTube video transcripts from {channel_name}")
    fg.language("en")

    for video in videos_with_transcripts:
        fe = fg.add_entry()
        fe.id(video["link"])
        fe.title(video["title"])
        fe.link(href=video["link"])
        fe.content(video["transcript_html"], type="CDATA")

        # Parse published date
        if video.get("published"):
            try:
                pub_date = datetime.fromisoformat(video["published"].replace("Z", "+00:00"))
                fe.published(pub_date)
                fe.updated(pub_date)
            except (ValueError, TypeError):
                fe.published(datetime.now(timezone.utc))
                fe.updated(datetime.now(timezone.utc))
        else:
            fe.published(datetime.now(timezone.utc))
            fe.updated(datetime.now(timezone.utc))

        # Add original description if available
        if video.get("summary"):
            fe.summary(video["summary"])

    output_path = output_dir / f"{slug}.xml"
    fg.rss_file(str(output_path), pretty=True)
    log.info("Wrote feed: %s (%d items)", output_path, len(videos_with_transcripts))


def process_channel(channel: dict, seen: dict, cleanup_state: dict, settings: dict, output_dir: Path, max_videos: int, base_url: str) -> tuple[dict, dict]:
    """Process a single channel: fetch feed, get transcripts, generate RSS."""
    name = channel["name"]
    slug = channel["slug"]
    channel_id = channel["channel_id"]

    log.info("Processing channel: %s", name)

    # Initialize seen list for this channel
    if channel_id not in seen:
        seen[channel_id] = []

    # Fetch channel's video feed
    videos = fetch_channel_feed(channel_id)
    if not videos:
        log.warning("No videos found for %s", name)
        return seen, cleanup_state

    log.info("Found %d videos in feed for %s", len(videos), name)
    videos = [v for v in videos if video_matches_channel_filters(channel, v)]
    log.info("After filters: %d videos for %s", len(videos), name)

    # Process new videos (that we haven't seen before)
    videos_with_transcripts = []
    new_count = 0

    for video in videos:
        vid = video["video_id"]

        if vid in seen[channel_id]:
            # Already processed — load cached transcript if we have it
            cache_path = output_dir.parent / "data" / f"transcript_{vid}.json"
            if cache_path.exists():
                with open(cache_path) as f:
                    cached = json.load(f)
                videos_with_transcripts.append(cached)
            continue

        # New video — fetch transcript
        log.info("Fetching transcript for: %s (%s)", video["title"], vid)
        segments = fetch_transcript(vid)

        if segments is None:
            log.info("Skipping %s (no transcript available)", vid)
            seen[channel_id].append(vid)
            # Cache a no-transcript entry
            entry = {
                **video,
                "transcript_html": "<p><em>No transcript available for this video.</em></p>",
                "cleanup_applied": False,
                "cleanup_reason": "no_transcript",
            }
            cache_path = output_dir.parent / "data" / f"transcript_{vid}.json"
            with open(cache_path, "w") as f:
                json.dump(entry, f)
            videos_with_transcripts.append(entry)
            new_count += 1
        else:
            transcript_html = format_transcript_html(segments)
            entry = {**video, "transcript_html": transcript_html}
            if should_run_cleanup(video, settings, cleanup_state):
                entry["transcript_html"] = maybe_run_cleanup(entry, settings)
                entry["cleanup_applied"] = True
                entry["cleanup_reason"] = "new_video_after_cutoff"
                cleanup_state.setdefault("cleaned_video_ids", []).append(vid)
                log.info("Cleanup applied to: %s", vid)
            else:
                entry["cleanup_applied"] = False
                entry["cleanup_reason"] = "legacy_or_disabled"
            # Cache the transcript
            cache_path = output_dir.parent / "data" / f"transcript_{vid}.json"
            with open(cache_path, "w") as f:
                json.dump(entry, f)
            videos_with_transcripts.append(entry)
            seen[channel_id].append(vid)
            new_count += 1
            log.info("Got transcript for: %s (%d segments)", video["title"], len(segments))

        # Rate limiting between transcript API calls
        time.sleep(1)

    log.info("Processed %d new videos for %s", new_count, name)

    # Trim to max_videos (keep most recent)
    videos_with_transcripts = videos_with_transcripts[:max_videos]

    # Generate RSS feed
    if videos_with_transcripts:
        generate_feed(name, slug, videos_with_transcripts, output_dir, base_url)

    return seen, cleanup_state


def main():
    log.info("=== Starting fetch_feeds run ===")
    config = load_config()
    settings = config["settings"]

    output_dir = SCRIPT_DIR / settings["output_dir"]
    data_dir = SCRIPT_DIR / settings["data_dir"]
    output_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    max_videos = settings.get("max_videos_per_channel", 15)
    base_url = settings.get("public_base_url", "").strip() or f"http://127.0.0.1:{settings.get('port', 8888)}"

    seen = load_seen_videos(data_dir)
    cleanup_state = load_cleanup_state(data_dir)
    if not cleanup_state.get("cutoff_utc"):
        configured_cutoff = settings.get("cleanup_cutoff_utc", "").strip()
        cleanup_state["cutoff_utc"] = configured_cutoff or utc_now_iso()
        log.info("Initialized cleanup cutoff: %s", cleanup_state["cutoff_utc"])

    for channel in config["channels"]:
        try:
            seen, cleanup_state = process_channel(channel, seen, cleanup_state, settings, output_dir, max_videos, base_url)
        except Exception:
            log.exception("Error processing channel %s", channel["name"])

    save_seen_videos(data_dir, seen)
    save_cleanup_state(data_dir, cleanup_state)
    log.info("=== Finished fetch_feeds run ===")


if __name__ == "__main__":
    main()
