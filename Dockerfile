FROM python:3.10-slim

WORKDIR /app

COPY . .

# 🔥 HARD CHECK
RUN ls -R /app/artifacts || echo "ARTIFACTS STILL MISSING"

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8080

CMD ["python3", "app.py"]