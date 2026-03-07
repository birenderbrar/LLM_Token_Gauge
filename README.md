# Local LLM Context Gauge

A "Glass Box" Chat Application that visualizes the memory constraints of Large Language Models (LLMs).

This project runs a **Llama-3.2-3B-Instruct** model locally on your machine and displays a real-time "Gas Gauge" representing the Context Window consumption. It is designed to teach the concepts of token limits, memory management, and the necessity of RAG (Retrieval-Augmented Generation).

## 🚀 Features

*   **Local Inference**: Runs completely offline (after model download) using Hugging Face Transformers.
*   **4-Bit Quantization**: Uses `bitsandbytes` to run efficiently on consumer hardware (fits easily in 32GB RAM).
*   **Context Visualization**: A dynamic progress bar shows exactly how many tokens are used by the prompt + history + response.
*   **LangChain Integration**: Uses LangChain for the orchestration layer.
*   **Flask Backend**: Simple, lightweight Python web server.

## 🛠️ Tech Stack

*   **Python 3.10+**
*   **Flask** (Web Framework)
*   **LangChain** (LLM Orchestration)
*   **Hugging Face Transformers** (Model Loading)
*   **BitsAndBytes** (Quantization)
*   **HTML/CSS/JS** (Frontend)

## 📋 Prerequisites

*   **Hardware**:
    *   RAM: 8GB minimum (16GB+ recommended).
    *   GPU: NVIDIA GPU recommended for speed (CUDA), but runs on CPU with sufficient RAM.
*   **Hugging Face Account**:
    *   You need a Hugging Face account and an Access Token.
    *   You must accept the license for meta-llama/Llama-3.2-3B-Instruct.

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/llm-context-gauge.git
    cd llm-context-gauge
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

## 📂 Project Structure

```text
llm-context-gauge/
├── app.py                 # Main Flask application & LLM logic
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Chat interface & Gauge UI
└── README.md              # Documentation
```