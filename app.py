from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests

app = FastAPI(title="Chatbot API")

HF_TOKEN = os.getenv("HF_TOKEN_2")  # Use HF_TOKEN_2 as requested
DEFAULT_MODEL = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolLM2-1.7B-Instruct")

class GenerateRequest(BaseModel):
    prompt: str
    model: str = None
    max_length: int = 100

@app.post("/generate")
def generate(req: GenerateRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    model = req.model or DEFAULT_MODEL
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    else:
        raise HTTPException(status_code=401, detail="Hugging Face token missing")

    payload = {
        "inputs": req.prompt,
        "parameters": {"max_new_tokens": req.max_length},
    }

    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        output = response.json()

        # Extract generated text from output
        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
            text = output[0].get("generated_text") or str(output[0])
        elif isinstance(output, dict):
            text = output.get("generated_text") or str(output)
        else:
            text = str(output)

        return {"result": text}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=str(e))
