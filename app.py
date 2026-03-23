import os
import uuid
from datetime import datetime
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
import chromadb
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_ID = "google/gemma-2-2b-it"
MAX_CONTEXT_WINDOW = 4096 
print(f"Loading {MODEL_ID}... This may take a minute.")

# --- MODEL LOADING ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)

text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# --- CHROMA DB & EMBEDDING ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
knowledge_collection = chroma_client.get_or_create_collection(name="user_knowledge")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={"torch_dtype": torch.bfloat16}
)

# --- STATE MANAGEMENT ---
conversation_history = []
current_session_id = str(uuid.uuid4())

def count_tokens(text):
    """Returns the number of tokens in a text string."""
    return len(tokenizer.encode(text))

@app.route('/')
def home():
    return render_template('index.html')

def generate_retrieval_summary(messages):
    """Uses Gemma-2 to distill the conversation into a search-optimized memory block."""
    if not messages:
        return ""
    
    convo_text = "\n".join([f"User: {m['user']}\nAssistant: {m['assistant']}" for m in messages])
    prompt = (
    "<start_of_turn>user\n"
    "Distill this conversation into a fact-dense, searchable summary. "
    "Identify and include key entities, specific technical terms, core problems discussed, "
    "and any final decisions or solutions reached. "
    "Avoid conversational filler. Focus on high-signal keywords to ensure this record is easily retrievable by its core concepts. "
    "Respond with a plain text summary and do not use markdown formatting or headers.\n\n"
    f"Conversation:\n{convo_text}<end_of_turn>\n<start_of_turn>model\n"
    )
    response = llm.invoke(prompt)
    if response.startswith(prompt):
        response = response[len(prompt):]        
    cleaned_response = response.strip()
    return cleaned_response if cleaned_response else "General"

def generate_primary_topic(messages):
    """Generates a 1-word label for the conversation topic."""
    if not messages:
        return "General"
        
    convo_text = "\n".join([f"User: {m['user']}\nAssistant: {m['assistant']}" for m in messages])
    prompt = (
        "<start_of_turn>user\n"
        "Provide exactly one word that describes the primary topic of the following conversation. "
        "Only output the single word.\n\n"
        f"Conversation:\n{convo_text}<end_of_turn>\n<start_of_turn>model\n"
    )
    response = llm.invoke(prompt)
    if response.startswith(prompt):
        response = response[len(prompt):]
    return response.strip()

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Query ChromaDB for relevant memory
    retrieved_context = ""
    is_memory = False
    score = 0
    memory_date = ""

    if knowledge_collection.count() > 0:
        query_embedding = embedding_model.encode(user_input).tolist()
        results = knowledge_collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )
        if results and results['documents'] and results['documents'][0]:
            distance = results['distances'][0][0]
            if distance < 1.8:  # Adjust threshold as needed based on embedding space
                retrieved_doc = results['documents'][0][0]  
                meta = results['metadatas'][0][0]
                retrieved_context = f"You are an assistant. Here is some context from the user's archives:\n[ARCHIVE: {retrieved_doc}]\n\n"
                is_memory = True
                score = max(0, min(100, int((1.0 - (distance / 2.0)) * 100))) # Calculate pseudo similarity score
                memory_date = meta.get("timestamp", "").split("T")[0] if meta.get("timestamp") else "Unknown"

    # Format the prompt with history
    formatted_history = ""
    for turn in conversation_history:
        formatted_history += f"<start_of_turn>user\n{turn['user']}<end_of_turn>\n<start_of_turn>model\n{turn['assistant']}<end_of_turn>\n"

    system_context = f"<start_of_turn>user\n{retrieved_context}Please keep this context in mind.<end_of_turn>\n<start_of_turn>model\nUnderstood. I will use this archived context to assist you.<end_of_turn>\n" if retrieved_context else ""
    full_prompt = f"{system_context}{formatted_history}<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
    
    # Calculate Input Tokens
    input_tokens = count_tokens(full_prompt)
    
    # Exit if context window is exceeded
    if input_tokens >= MAX_CONTEXT_WINDOW:
        return jsonify({
            "response": "⚠️ Context Window Exceeded! Either delete or archive to continuing enriching the conversation.",
            "usage_percent": 100,
            "current_tokens": input_tokens,
            "max_tokens": MAX_CONTEXT_WINDOW
        })

    # Generate Response
    response_text = llm.invoke(full_prompt)
    
    # Clean up response
    if response_text.startswith(full_prompt):
        response_text = response_text[len(full_prompt):]

    # Update History
    turn_id = str(uuid.uuid4())
    turn_timestamp = datetime.now().isoformat()
    conversation_history.append({
        "id": turn_id,
        "timestamp": turn_timestamp,
        "user": user_input,
        "assistant": response_text
    })
    
    # Calculate Total Tokens
    output_tokens = count_tokens(response_text)
    total_tokens = input_tokens + output_tokens
    usage_percent = round((total_tokens / MAX_CONTEXT_WINDOW) * 100, 2)

    return jsonify({
        "response": response_text,
        "usage_percent": usage_percent,
        "current_tokens": total_tokens,
        "max_tokens": MAX_CONTEXT_WINDOW,
        "turn_id": turn_id,
        "timestamp": turn_timestamp,
        "is_memory": is_memory,
        "score": f"{score}%" if is_memory else None,
        "memory_date": memory_date
    })

@app.route('/delete_message', methods=['POST'])
def delete_message():
    global conversation_history
    turn_id = request.json.get('id')
    if not turn_id:
        return jsonify({"error": "No ID provided"}), 400
        
    # Remove the specific Q&A pair from the context limit
    conversation_history = [turn for turn in conversation_history if turn.get('id') != turn_id]
    
    # Recalculate tokens against the newly pruned history list
    formatted_history = ""
    for turn in conversation_history:
        formatted_history += f"<start_of_turn>user\n{turn['user']}<end_of_turn>\n<start_of_turn>model\n{turn['assistant']}<end_of_turn>\n"
        
    current_tokens = count_tokens(formatted_history)
    usage_percent = round((current_tokens / MAX_CONTEXT_WINDOW) * 100, 2)
    
    return jsonify({
        "status": "success",
        "current_tokens": current_tokens,
        "usage_percent": usage_percent,
        "max_tokens": MAX_CONTEXT_WINDOW
    })

@app.route('/reset', methods=['POST'])
def reset():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "reset"})

@app.route('/archive_session', methods=['POST'])
def archive_session():
    global conversation_history
    global current_session_id
    
    if not conversation_history:
        return jsonify({"error": "No conversation to archive"}), 400
        
    # Reconstruct history to calculate token weight accurately
    formatted_history = ""
    for turn in conversation_history:
        formatted_history += f"<start_of_turn>user\n{turn['user']}<end_of_turn>\n<start_of_turn>model\n{turn['assistant']}<end_of_turn>\n"
    
    token_weight = count_tokens(formatted_history)
    
    # Generate Summary and Primary Topic
    summary = generate_retrieval_summary(conversation_history)
    topic = generate_primary_topic(conversation_history)
    
    # Embed the summary
    embedding = embedding_model.encode(summary).tolist()
    
    # Prepare metadata
    doc_id = str(uuid.uuid4())
    metadata = {
        "session_id": current_session_id,
        "timestamp": datetime.now().isoformat(),
        "primary_topic": topic,
        "token_weight": token_weight
    }
    
    # Store in ChromaDB
    knowledge_collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[summary],
        metadatas=[metadata]
    )
    
    # Reset conversation state
    conversation_history = []
    
    return jsonify({
        "status": "success",
        "message": "Session committed to memory successfully.",
        "current_tokens": 0,
        "usage_percent": 0.0,
        "max_tokens": MAX_CONTEXT_WINDOW
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
