# Research Agent with LLaMA and Falcon Models

This project implements a **Research Agent** that combines state-of-the-art language models, like **LLaMA** and **Falcon**, with search and summarization capabilities. The agent can:
- **Perform web searches** using DuckDuckGo.
- **Summarize long texts** with advanced LLMs.
- **Automatically use GPU if available** or fallback to CPU for processing.

The project is built using **LangChain** and **Hugging Face Transformers**.

---

## Features

1. **Web Search**: Automatically fetch relevant information from the internet.
2. **Summarization**: Generate concise summaries of large texts.
3. **Model Agnostic**: Use smaller, quantized models like Falcon-7B-Instruct or LLaMA-2 for faster processing.
4. **Device Detection**: Automatically detects and utilizes GPU if available.
5. **Streamlit UI**: Provides an interactive interface for easy access.

---

## Installation

### Prerequisites
- Python 3.8 or later
- PyTorch installed (with CUDA support for GPU usage)

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/your-username/research-agent.git
   cd research-agent
   ```
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Install bitsandbytes for 8-bit quantization (optional):
   ```
   pip install bitsandbytes
   ```
4. Authenticate with Hugging Face:
   ```
   huggingface-cli login
   ```
### Usage
#### Run the Application

1. Launch the Streamlit app:
   ```
   streamlit run app.py
   ```
### Features in the App
**Hugging Face Token Input:** Enter your token to load gated models if access is needed like for Llama  models.
**Query Input:** Ask the agent a question or provide a topic for research.
**Chat History:** View the conversation history with the agent.  

### Troubleshooting
#### Model Loading Errors
- Ensure you have access to the model on Hugging Face.
- Verify your Hugging Face token by running:
  ```
  huggingface-cli whoami
  ```
### GPU Not Detected
- Install the correct PyTorch version with CUDA
  ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
