FROM python:3.11

WORKDIR /app

RUN pip install --upgrade pip setuptools

RUN apt-get update && apt-get install -y ca-certificates curl gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | tee /etc/apt/keyrings/nodesource.gpg > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci --omit=dev; else npm i --omit=dev; fi

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY sql_init/backup.tar.gz /sql_init/backup.tar.gz
RUN mkdir -p /docker-entrypoint-initdb.d \
 && tar -xzf /sql_init/backup.tar.gz -C /docker-entrypoint-initdb.d


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
