FROM python:3.11

WORKDIR /app

RUN pip install --upgrade pip setuptools

RUN apt-get update && apt-get install -y ca-certificates curl gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | tee /etc/apt/keyrings/nodesource.gpg > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /docker-entrypoint-initdb.d

COPY sql_init/skillcrawl.sql.gz /docker-entrypoint-initdb.d/skillcrawl.sql.gz

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

RUN mkdir -p /app/longterm_storage