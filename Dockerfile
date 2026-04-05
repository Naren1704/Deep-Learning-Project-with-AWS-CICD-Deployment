FROM python:3.10-slim

WORKDIR /app

# Copy everything
COPY . .

# 🔥 DEBUG (this will print during build logs)
RUN echo "Checking artifacts..." && ls -R artifacts || echo "NO ARTIFACTS FOUND"

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8080

CMD ["python3", "app.py"]