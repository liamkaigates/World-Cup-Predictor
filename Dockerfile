FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data ./data
COPY src ./src
COPY tests ./tests

ENV PYTHONPATH=src

RUN python -m wc_forecast.cli train --matches data/sample_matches.csv --model artifacts/model.pkl

EXPOSE 8000
CMD ["python", "-m", "wc_forecast.api", "--model", "artifacts/model.pkl", "--host", "0.0.0.0", "--port", "8000"]

