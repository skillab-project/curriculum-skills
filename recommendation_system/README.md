# Academic Recommendation System

## Start

Run the full stack and seed the database in one go:

```bash
docker compose up -d --build && docker compose exec api python -m backend.seed
```

This will:

* Start the **MySQL** database, **FastAPI** backend, and **frontend** containers
* Automatically populate demo universities, programs, courses, and skills

---

## Access

* Frontend → [http://localhost:3000](http://localhost:3000)
* API Docs → [http://localhost:8000/docs](http://localhost:8000/docs)

To reset everything:

```bash
docker compose down -v
```
