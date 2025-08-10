# app.py
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os
import requests

app = FastAPI(title="Chatbot API")

HF_TOKEN = os.getenv("HF_TOKEN")                # optional - if using HF inference
DEFAULT_MODEL = os.getenv("MODEL_NAME", "google/flan-t5-large")

class GenReq(BaseModel):
    prompt: str
    model: str = None
    max_length: int = 100

@app.post("/generate")
def generate(req: GenReq, authorization: str = Header(None)):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    model = req.model or DEFAULT_MODEL

    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    payload = {
        "inputs": req.prompt,
        "parameters": {"max_new_tokens": req.max_length}
    }
    try:
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers=headers,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        out = resp.json()
        # parse common HF formats
        if isinstance(out, list) and isinstance(out[0], dict):
            text = out[0].get("generated_text") or str(out[0])
        elif isinstance(out, dict):
            text = out.get("generated_text") or str(out)
        else:
            text = str(out)
        return {"result": text}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=str(e))
