FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY defence_agent ./defence_agent

RUN pip install --no-cache-dir -e .

EXPOSE 8501
CMD ["streamlit", "run", "defence_agent/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
