from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import httpx
import json

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient()
    yield
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan)

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    try:
        response = await app.state.http_client.get(f"http://10.68.198.184:8000/insights/{item_id}", timeout=15.0)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=404, detail="Item not found")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=str(exc))