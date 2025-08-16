FROM python:3.11

WORKDIR /app

# --- Python setup
RUN pip install --upgrade pip setuptools

# --- Node.js 20
RUN apt-get update && apt-get install -y ca-certificates curl gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | tee /etc/apt/keyrings/nodesource.gpg > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# (Optional) Apify CLI, not required to run crawlers via node
# RUN npm i -g apify@3

# --- Install Node dependencies (crawlee) first for better caching
COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci --omit=dev; else npm i --omit=dev; fi

# If you use PlaywrightCrawler and need bundled browsers:
# RUN npx playwright install --with-deps

# --- Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy source after deps
COPY . .

# (Optional) ensure apify.js exists at /app/apify.js
# RUN test -f /app/apify.js

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
