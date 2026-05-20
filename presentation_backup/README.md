# Defence Agent Presentation Backup

This folder contains a static backup page for the Streamlit demo.

## Use It

- Open `presentation_backup/index.html` directly in a browser.
- For a local URL, run:

```bash
python -m http.server 4173 --directory presentation_backup
```

- Then open `http://127.0.0.1:4173`.

## Refresh It

Run this after updating demo transcripts, corpus metadata, or the walkthrough order:

```bash
python presentation_backup/build_backup.py
```

The generator reads:

- `defence_agent/data/corpus/manifest.yaml`
- selected saved runs under `defence_agent/data/transcripts/`
- selected reviewer challenge runs under `defence_agent/data/evals/reviewer_challenge_runs/`

It writes:

- `presentation_backup/index.html`

## Extend It

- Edit `DEMO_RUNS` in `presentation_backup/build_backup.py` to add, remove, or reorder cases.
- Keep the first section as the database/source catalog.
- Keep the first demo case as the planning brief comparison.
- Use reviewer, access-boundary, scanned-manual, refusal, follow-up, and bilingual runs as modular proof sections.

## Online Backup Options

- Fastest: drag the `presentation_backup` folder into Netlify Drop.
- GitHub Pages: publish this folder through a Pages workflow, or copy `index.html` into a Pages-enabled `docs/` folder.
- Avoid putting `.env`, Cohere keys, Chroma data, or session databases online.

## Presenter Line

If the live app fails:

> This is a static replay of prior verified Defence Agent runs. It preserves the same database, query, answer, trace, evidence, citation, and access-control proof so we can keep walking through how the system works.
