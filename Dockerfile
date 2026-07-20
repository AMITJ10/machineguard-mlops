FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN addgroup --system appgroup \
 && adduser \
    --system \
    --ingroup appgroup \
    --home /home/appuser \
    appuser

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p \
    /app/logs \
    /app/artifacts \
    /app/reports/drift \
 && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD [ "uvicorn", "api.main:app", "--host","0.0.0.0", "--port","8000" ]