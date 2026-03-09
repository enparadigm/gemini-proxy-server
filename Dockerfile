FROM python:3.11-slim

WORKDIR /app

# Install system deps if needed (e.g. for ssl, certifi usually fine)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8080 8000

CMD ["python", "server.py"]