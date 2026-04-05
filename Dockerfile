FROM python:3.10-slim-buster

RUN apt update -y && apt install awscli -y

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8080

CMD ["python3", "app.py"]