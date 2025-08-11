from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI(title="Chatbot API")

# Default model (can be overridden in request)
DEFAULT_MODEL = "google/flan-t5-large"

# Load default model at startup
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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

        if isinstance(output, list) and isinstance(output[0], dict):
            text = output[0].get("generated_text") or str(output[0])
        elif isinstance(output, dict):
            text = output.get("generated_text") or str(output)
        else:
            text = str(output)

        return {"result": text}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=str(e))
