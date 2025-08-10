import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Initialize model and tokenizer
device = torch.device("cpu")  # Hugging Face Spaces usually run on CPU unless GPU requested
model_name = "google/flan-t5-large"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    #model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
except Exception as e:
    raise RuntimeError(f"Error loading model or tokenizer: {e}")

def generate_response(user_input, max_length=100):
    try:
        # Create prompt with context
        prompt = f"""
You are a helpful assistant. Respond to the user's input in a conversational and friendly manner.

User: {user_input}
Assistant:
"""
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = inputs.to(device)
        
        # Generate response
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=5,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        return f"Error generating response: {e}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 💬 Flan-T5 Chatbot\nType a message and chat with the AI.")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your message")
    clear = gr.Button("Clear")

    def respond(message, chat_history):
        bot_response = generate_response(message)
        chat_history.append((message, bot_response))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()






   
