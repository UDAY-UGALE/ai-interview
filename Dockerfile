# Server image. The client (overlay + audio capture) is NOT in here -- it
# runs on the interviewee's own Windows PC and talks to this over wss://.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements first so a code-only change reuses the cached install layer.
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY app ./app

# answer_hub writes a per-session JSONL transcript here. On a free host the
# filesystem is ephemeral (wiped on every redeploy/restart), so treat these
# as debug output, not storage.
RUN mkdir -p logs

# $PORT is injected by Render / Cloud Run / Koyeb; 8000 is the local default.
ENV PORT=8000
EXPOSE 8000

# One worker on purpose: SESSION_STORE_BACKEND=memory and the in-process
# answer hub both assume a single process. Going multi-worker without
# switching to Redis would put a client's audio socket and its answer
# socket in different processes, and answers would never arrive.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --ws-ping-interval 20 --ws-ping-timeout 20"]
