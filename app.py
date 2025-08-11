from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import requests

app = FastAPI(title="Chatbot API")

# Token for private models
HF_TOKEN = os.getenv("HF_TOKEN_2")

# Default model
DEFAULT_MODEL = "google/flan-t5-large"

# Load default model locally at startup
tokenizer = AutoTokenizer.from_pretrained(
    DEFAULT_MODEL, use_auth_token=HF_TOKEN if HF_TOKEN else None
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    DEFAULT_MODEL, use_auth_token=HF_TOKEN if HF_TOKEN else None
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


class GenerateRequest(BaseModel):
    prompt: str
    model: str = None
    max_length: int = 100
    use_remote: bool = False  # If True, use HF API instead of local


@app.post("/generate")
def generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        # Decide model
        model_name = req.model or DEFAULT_MODEL

        # ----------- Remote Inference -----------
        if req.use_remote:
            headers = {}
            if HF_TOKEN:
                headers["Authorization"] = f"Bearer {HF_TOKEN}"

            payload = {
                "inputs": req.prompt,
                "parameters": {"max_new_tokens": req.max_length},
            }

            response = requests.post(
                f"https://api-inference.huggingface.co/models/{model_name}",
                headers=headers,
                json=payload,
                timeout=60,
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

        # ----------- Local Inference -----------
        else:
            if req.model and req.model != DEFAULT_MODEL:
                temp_tokenizer = AutoTokenizer.from_pretrained(
                    model_name, use_auth_token=HF_TOKEN if HF_TOKEN else None
                )
                temp_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, use_auth_token=HF_TOKEN if HF_TOKEN else None
                ).to(device)
            else:
                temp_tokenizer = tokenizer
                temp_model = model

            input_ids = temp_tokenizer(req.prompt, return_tensors="pt").input_ids.to(device)
            outputs = temp_model.generate(input_ids, max_new_tokens=req.max_length)
            text = temp_tokenizer.decode(outputs[0], skip_special_tokens=True)

            return {"result": text}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Remote API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local inference error: {e}")
