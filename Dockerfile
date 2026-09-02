FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data ./data
COPY static ./static

RUN useradd --create-home app && chown -R app /app
USER app

RUN wc-forecast train --matches data/sample_matches.csv --model artifacts/model.pkl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["wc-forecast-api", "--model", "artifacts/model.pkl", "--matches", "data/sample_matches.csv", "--host", "0.0.0.0", "--port", "8000"]
