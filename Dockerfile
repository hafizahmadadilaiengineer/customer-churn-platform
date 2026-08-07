FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

ENV PYTHONPATH=/app

CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]