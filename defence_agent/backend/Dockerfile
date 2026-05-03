FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY defence_agent ./defence_agent

RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "defence_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
