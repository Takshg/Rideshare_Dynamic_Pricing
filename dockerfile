FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml ./
COPY src/ src/
COPY artifacts/ artifacts/

RUN pip install --upgrade pip && pip install -e ".[api]"

EXPOSE 8000

CMD ["uvicorn", "dynamic_pricing.api.main:app", "--host", "0.0.0.0", "--port", "8000"]