# YouTube Transcript Feeds

This project builds RSS feeds with transcript content for selected YouTube channels and publishes them via GitHub Pages.

## Local run

```bash
python fetch_feeds.py
python build_index.py
python serve.py
```

## Public hosting

- GitHub Actions workflow: `.github/workflows/publish.yml`
- Pages output: `output/`

## Notes on secrets

- This setup does not require API keys for transcript fetching.
- Do not commit `.env` files or private tokens.
