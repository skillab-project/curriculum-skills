from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from recommendation_system.backend.database import init_db
from recommendation_system.backend.routers import  recommendations, electives, filters, policy


app = FastAPI(
    title="Academic Recommender API",
    version="1.0",
    description="API for course, skill, and degree recommendations."
)

# ----------------------------------
# CORS (Allow frontend to connect)
# ----------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Αφού τρέχεις local frontend, το αφήνουμε ανοιχτό
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------
# DB INIT
# ----------------------------------
@app.on_event("startup")
def on_startup():
    try:
        init_db()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")

# ----------------------------------
# ROUTES (ΣΩΣΤΑ prefix)
# ----------------------------------



# Recommendations
# ❗ ΧΩΡΙΣ prefix="/recommend" για να μην σπάσουν όλα τα URLs
app.include_router(
    recommendations.router,
    tags=["Recommendations"]
)

# Electives
app.include_router(
    electives.router,
    tags=["Electives"]
)

# Filters
app.include_router(
    filters.router,
    tags=["Filters"]
)

# --- Education Policy (MySQL) ---
app.include_router(
    policy.router,
    tags=["Education Policy"]
)

# ----------------------------------
# Root Check
# ----------------------------------
@app.get("/")
def root():
    return {"status": "API is running", "version": "1.0"}
