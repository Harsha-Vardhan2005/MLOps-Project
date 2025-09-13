FROM python:3.8-slim-buster

RUN apt-get update -y && apt-get install -y --no-install-recommends awscli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python3", "app.py"]
