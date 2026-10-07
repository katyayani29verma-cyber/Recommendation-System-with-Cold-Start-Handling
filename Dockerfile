FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY src/ src/
# Trained Track A/B artifacts (empty until those tracks land them).
COPY artifacts/ artifacts/

# The API reads MovieLens at startup; fetch it at build time since data/ is
# not baked into the build context.
RUN python -m src.data.download

# Cloud Run injects $PORT; default to 8080 for local `docker run`.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
