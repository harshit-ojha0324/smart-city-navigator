FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY eval/ eval/

# Bind on all interfaces inside the container network.
ENV NAVIGATOR_MCP_HOST=0.0.0.0

# Default command runs the gateway; docker-compose overrides `command` per service.
CMD ["python", "-m", "navigator.gateway.app"]
