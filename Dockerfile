FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "if [ \"$SKIP_MIGRATIONS\" != '1' ]; then alembic upgrade head || echo 'WARNING: alembic migration failed — bot starting anyway, manual migration may be required.'; fi; python -m app.main"]
