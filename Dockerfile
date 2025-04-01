FROM python:3.11
#FROM python:3.9-slim

WORKDIR /app

COPY . /app

#RUN pip install --no-cache-dir pdfplumber requests beautifulsoup4 urllib3 fuzzywuzzy pyfiglet esco-skill-extractor python-Levenshtein
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN apt-get update && apt-get upgrade -y  # Update package lists
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('words');"

#CMD ["python", "skillcrawl.py"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
