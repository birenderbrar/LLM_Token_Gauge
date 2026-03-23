# Super Chat

A "Glass Box" Chat Application designed for exploring **deep chats** while actively visualizing and managing the memory constraints of Large Language Models (LLMs).

This project runs a **Gemma-2-2B-IT** model locally on your machine and displays a real-time "Gas Gauge" representing your current Context Window consumption. It empowers users to take full control over the AI's memory. By using features like **delete**, users can actively prune unwanted conversational branches from their deep chat tree, keeping the context highly relevant and saving precious tokens. It practically demonstrates token limits, memory curation, and the transition to RAG (Retrieval-Augmented Generation) for long-term storage.

## 🚀 Features

*   **Local Inference**: Runs completely offline (after model download) using Hugging Face Transformers.
*   **4-Bit Quantization**: Uses `bitsandbytes` to run efficiently on consumer hardware (fits easily in 32GB RAM).
*   **BFloat16 Precision**: Strictly uses `torch.bfloat16` to prevent known activation overflow issues in Gemma-2 models.
*   **Context Visualization**: A dynamic progress bar shows exactly how many tokens are used by the prompt + history + response.
*   **RAG Memory Persistence**: Uses ChromaDB and SentenceTransformers to archive conversational context into searchable vector embeddings.
*   **LangChain Integration**: Uses LangChain for pipeline orchestration.
*   **Flask Backend**: Simple, lightweight Python web server.

## 🛠️ Tech Stack

*   **Python 3.10+**
*   **Flask** (Web Framework)
*   **LangChain** (LLM Orchestration)
*   **Hugging Face Transformers** (Model Loading)
*   **BitsAndBytes** (Quantization)
*   **ChromaDB** (Vector Database)
*   **SentenceTransformers** (Embeddings)
*   **HTML/CSS/JS** (Frontend)

## 📋 Prerequisites

*   **Hardware**:
    *   RAM: 8GB minimum (16GB+ recommended).
    *   GPU: NVIDIA GPU recommended for speed (CUDA), but runs on CPU with sufficient RAM.
*   **Hugging Face Account**:
    *   You need a Hugging Face account and an Access Token.
    *   You must accept the license for google/gemma-2-2b-it.

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/birenderbrar/Super-Chat.git
    cd super-chat
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Authenticate with Hugging Face**
    ```bash
    huggingface-cli login
    ```
    Paste your Write/Read token when prompted.

## 🏃‍♂️ Usage

1.  **Start the Flask App**
    ```bash
    python app.py
    ```
    *First run will take a few minutes to download the model (~2-3GB).*

2.  **Open the Interface**
    Navigate to `http://localhost:5000` in your browser.

3.  **Chat & Observe**
    *   Type messages to the AI.
    *   Watch the "Context Window Consumption" gauge fill up.
    *   The app is artificially capped at **4096 tokens** for demonstration purposes.
    *   Once the gauge hits 100%, the context is full (simulating the need for RAG).
    *   Click **Archive Session** to distill the conversation into a vector memory, clearing the gauge and allowing the AI to recall the information in future chats.

4.  **Inspect Vector DB (Optional)**
    *   Run `python peek_db.py` in a separate terminal to see exactly how ChromaDB is storing and retrieving your archived summaries via semantic search.


```
