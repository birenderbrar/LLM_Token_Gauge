import os
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
# We set a "Soft Limit" for the demo so you can actually see the bar fill up.
# Real model limit is 128k, but that's hard to visualize in a quick chat.
MAX_CONTEXT_WINDOW = 4096 

print(f"Loading {MODEL_ID}... This may take a minute.")

# --- MODEL LOADING ---
# 1. Configure 4-bit Quantization to save RAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

# 2. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 3. Load Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto", # Uses GPU if available, otherwise CPU/RAM
    low_cpu_mem_usage=True
)

# 4. Create LangChain Pipeline
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512, # Max tokens the model can generate in one go
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# --- STATE MANAGEMENT ---
# In a production app, use a database. For this local demo, we use a global list.
conversation_history = []

def count_tokens(text):
    """Returns the number of tokens in a text string."""
    return len(tokenizer.encode(text))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # 1. Format the prompt with history
    # We manually construct the prompt to strictly control and count tokens
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
    full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant.
<|eot_id|>
{formatted_history}
<|start_header_id|>user<|end_header_id|>
{user_input}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    # 2. Calculate Input Tokens (History + Current Prompt)
    input_tokens = count_tokens(full_prompt)
    
    # Check if we are out of gas
    if input_tokens >= MAX_CONTEXT_WINDOW:
        return jsonify({
            "response": "⚠️ Context Window Exceeded! Time to implement RAG (Phase 2).",
            "usage_percent": 100,
            "current_tokens": input_tokens,
            "max_tokens": MAX_CONTEXT_WINDOW
        })

    # 3. Generate Response
    # We use invoke() from LangChain
    response_text = llm.invoke(full_prompt)
    
    # Clean up response (LangChain sometimes returns the full prompt + response)
    # For Llama 3.2 pipeline, we usually get just the generation, but let's be safe:
    if response_text.startswith(full_prompt):
        response_text = response_text[len(full_prompt):]

    # 4. Update History
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": response_text})

    # 5. Calculate Total Tokens (Input + New Output)
    output_tokens = count_tokens(response_text)
    total_tokens = input_tokens + output_tokens
    usage_percent = round((total_tokens / MAX_CONTEXT_WINDOW) * 100, 2)

    return jsonify({
        "response": response_text,
        "usage_percent": usage_percent,
        "current_tokens": total_tokens,
        "max_tokens": MAX_CONTEXT_WINDOW
    })

@app.route('/reset', methods=['POST'])
def reset():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "reset"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
