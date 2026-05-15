# Streamlit Community Cloud Deployment

Use this for the hosted Defence Agent demo.

## Recommended Deployment

- Repository: `hananather/cohere-sa-interview-prep`
- Branch: `streamlit-community-static-demo`
- Main file path: `app.py`
- Secrets: none required

This entrypoint imports `streamlit_app.py`, so Streamlit Community Cloud opens the actual application. The default run mode uses bundled guided replays, so the demo can work without a `COHERE_API_KEY`.

## Why This Path

- It does not require `COHERE_API_KEY`.
- It opens the same Streamlit application used locally.
- It bundles the curated transcript runs needed for guided replay mode.
- It works as a hosted demo even if your laptop is unavailable.

## Full Local App

Use `streamlit_app.py` locally when you want to run the app directly.

For Streamlit Community Cloud, use `app.py`; it delegates to the same app.

## Static Backup

The standalone fallback remains in `presentation_backup/index.html`.
