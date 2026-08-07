from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="Customer Churn API",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Customer Churn Intelligence Platform API",
        "docs": "/docs",
        "health": "/health"
    }


app.include_router(router)