from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI(title="Hybrid GPT-Neo Chatbot")

# Env vars
HF_TOKEN = os.getenv("HF_TOKEN_2")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "google/flan-t5-large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try loading local model once at startup
try:
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    #model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    local_model_loaded = True
except Exception as e:
    print(f"Local model loading failed: {e}")
    local_model_loaded = False

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 100
    model: str = None  # Optional override

@app.post("/generate")
def generate(req: GenerateRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    model_name = req.model or DEFAULT_MODEL

    if local_model_loaded and model_name == DEFAULT_MODEL:
        # Run local inference
        try:
            inputs = tokenizer(req.prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_length,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                temperature=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"result": generated_text}
        except Exception as e:
            # If local inference fails, fallback to API
            print(f"Local inference failed: {e}")

    # Fallback: call Hugging Face Inference API
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "inputs": req.prompt,
        "parameters": {"max_new_tokens": req.max_length},
    }

    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model_name}",
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
        raise HTTPException(status_code=502, detail=f"API call failed: {e}")
