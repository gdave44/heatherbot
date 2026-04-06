FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG REPO_URL=https://github.com/gdave44/heatherbot.git
RUN git clone --depth=1 $REPO_URL .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

VOLUME ["/app/data", "/app/config"]

CMD ["python", "heather_telegram_bot.py", "--monitoring", "--small-model"]
