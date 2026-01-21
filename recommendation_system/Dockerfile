# =============================================
# Dockerfile για Backend
# =============================================

# Βασική εικόνα Python slim
FROM python:3.11-slim

# Εγκατάσταση συστημικών εξαρτήσεων
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Ορισμός φακέλου εργασίας
WORKDIR /app

# Αντιγραφή requirements
COPY requirements.txt /app/requirements.txt

# Εγκατάσταση Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi "uvicorn[standard]" pymysql sqlalchemy

# Αντιγραφή υπόλοιπου κώδικα
COPY . /app

# Έκθεση port για Uvicorn
EXPOSE 8000

# Ορισμός environment variables για τη σύνδεση με τη βάση
ENV DATABASE_HOST=db
ENV DATABASE_PORT=3306
ENV DATABASE_NAME=recommender_test1
ENV DATABASE_USER=root
ENV DATABASE_PASSWORD=2003Sept!

# Εκκίνηση backend
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]