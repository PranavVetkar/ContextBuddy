import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="ContextBuddy API",
    description="Domain-independent AI conversation context portability backend",
    version="0.1.0",
)

# CORS configuration for Angular frontend
origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API router under /api
app.include_router(api_router, prefix="/api")


@app.get("/")
def root():
    return {
        "app": "ContextBuddy API",
        "version": "0.1.0",
        "docs_url": "/docs",
        "health_url": "/api/health",
    }
