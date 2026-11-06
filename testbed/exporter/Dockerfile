FROM python:3.12-slim
RUN pip install --no-cache-dir docker prometheus-client
COPY stats_exporter.py /app/stats_exporter.py
CMD ["python", "/app/stats_exporter.py"]
