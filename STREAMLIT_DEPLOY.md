# Streamlit Community Cloud Deployment

Use this for the reliable hosted backup demo.

## Recommended Deployment

- Repository: `hananather/cohere-sa-interview-prep`
- Branch: `codex/streamlit-community-static-demo`
- Main file path: `app.py`
- Secrets: none required

This entrypoint renders `presentation_backup/index.html`, which is a static replay of verified Defence Agent runs.

## Why This Path

- It does not require `COHERE_API_KEY`.
- It does not require Chroma runtime state.
- It does not require ignored transcript folders at runtime.
- It works as a hosted demo even if your laptop is unavailable.

## Full Local App

Use `streamlit_app.py` locally when you want the full guided/live app.

For Streamlit Community Cloud, use `app.py`.
