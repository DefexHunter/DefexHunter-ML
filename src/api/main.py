from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.router import router
from src.api.predictor import load_artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ───────────────────────────────────────────────────────────────
    print("Loading model artifacts...")
    load_artifacts()
    print("API ready.")
    yield
    # ── shutdown ──────────────────────────────────────────────────────────────
    print("Shutting down.")


app = FastAPI(
    title="DefexHunter ML API",
    description=(
        "Predicts software defects from NASA JM1 static analysis metrics.\n\n"
        "**Pipeline:** corr drop → NearMiss → 70/30 split → StandardScaler → GridSearchCV\n\n"
        "**Models:** Decision Tree · KNN · Random Forest · SVM · XGBoost"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Info"])
def root():
    return {
        "name":    "DefexHunter ML API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
        "models":  "/models",
    }