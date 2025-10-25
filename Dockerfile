# Minimal runtime for CLI and API
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn

# Optional extras can be installed by uncommenting:
# COPY requirements-ecapa.txt /app/
# RUN pip install --no-cache-dir -r requirements-ecapa.txt
# COPY requirements-extra.txt /app/
# RUN pip install --no-cache-dir -r requirements-extra.txt

COPY speaker_id /app/speaker_id
COPY setup.cfg /app/

# Expose API port
EXPOSE 8000

# Default command runs API
CMD ["python", "-m", "uvicorn", "speaker_id.api:app", "--host", "0.0.0.0", "--port", "8000"]
