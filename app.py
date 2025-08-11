from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests

app = FastAPI(title="Commercial Chatbot API")

# Use your Hugging Face token for private or paid models
HF_TOKEN = os.getenv("HF_TOKEN_2")
if not HF_TOKEN:
    raise RuntimeError("Please set HF_TOKEN_2 environment variable with your Hugging Face API token")

DEFAULT_MODEL = "google/flan-t5-large"

class GenerateRequest(BaseModel):
    prompt: str
    model: str = None
    max_length: int = 100

@app.post("/generate")
def generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    model_name = req.model or DEFAULT_MODEL

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    payload = {
        "inputs": req.prompt,
        "parameters": {"max_new_tokens": req.max_length},
    }

    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model_name}",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        output = response.json()

        # Parse the generated text safely
        if isinstance(output, list) and isinstance(output[0], dict):
            text = output[0].get("generated_text") or str(output[0])
        elif isinstance(output, dict):
            text = output.get("generated_text") or str(output)
        else:
            text = str(output)

        return {"result": text}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"HF API error: {e}")
