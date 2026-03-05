from fastapi import FastAPI
import json

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/welcome")
async def welcome():
    return {"message": "Welcome to our API"}

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    with open("data.json", "r") as f:
        data = f.read()
    return {"item_id": item_id, "data": data}