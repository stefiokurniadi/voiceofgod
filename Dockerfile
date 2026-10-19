FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/voiceofgod
COPY pyproject.toml README.md ./
COPY app ./app
COPY eval ./eval
COPY glossary ./glossary
COPY db ./db
RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1 \
    VOICEOFGOD_ROOT=/opt/voiceofgod
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
