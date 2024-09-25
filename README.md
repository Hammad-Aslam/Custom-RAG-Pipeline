
---

# Simple Local RAG: NutriChat

This repository contains a simple **local Retrieval-Augmented Generation (RAG)** pipeline designed to run on a local **NVIDIA GPU ON GOOGLE COLAB** for document processing and embedding creation, followed by search and answer functionality. 

In this specific example, we build **NutriChat**, a RAG workflow that allows users to query a 1200-page PDF Nutrition Textbook and get responses from an LLM based on passages of text from the textbook.

## Workflow Overview

This flowchart describes a simple local RAG workflow for document processing and embedding creation, followed by search and answer functionality. 

1. **Document Collection**: A set of documents (e.g., PDFs or a 1200-page Nutrition textbook) is preprocessed into smaller chunks, such as groups of 10 sentences.
2. **Embedding Generation**: A user query is transformed into a numerical embedding using sentence transformers or Hugging Face models. These embeddings are stored in a `torch.tensor` format for efficient handling, especially for large datasets.
3. **Document Retrieval**: Based on the user's query, the relevant document chunks are retrieved for context using vector search algorithms like cosine similarity.
4. **LLM Output**: The LLM generates a response using the retrieved chunks, running all computations on a local GPU (e.g., RTX 4090).
5. **User Interaction**: The response is delivered via a chat-like interface for easy user interaction.

All processing happens **locally** using open-source tools, running on a **NVIDIA GPU**.

## Getting Started


### Setup

**Note**: Tested in Python 3.11, on Windows 11 with an NVIDIA RTX 4090 with CUDA 12.1.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Hammad-Aslam/Custom-RAG-Pipeline.git
   cd Custom-RAG-Pipeline
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the environment**:

   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

4. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

   *Note*: You may need to install `torch` manually (torch 2.1.1+ with CUDA for better inference speed):
   ```bash
   pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

5. **Launch the notebook**:
   - In VS Code:
     ```bash
     code .
     ```
   - In Jupyter Notebook:
     ```bash
     jupyter notebook
     ```

### Model Setup

To use the **Gemma LLM models** from Hugging Face, you must first agree to their terms & conditions and authenticate your machine via the Hugging Face CLI or the `login()` function in your script. Once authenticated, you can download the models for local use.

For advanced users, **Flash Attention 2** can speed up attention-based processing. It's commented out in the `requirements.txt` due to long compile times, but you can install it using:
```bash
pip install flash-attn
```

### PDF Source

We use a publicly available PDF as our document source:
- **Human Nutrition PDF**: [Nutrition Textbook](https://pressbooks.oer.hawaii.edu/humannutrition2/)

You can replace this with any large document collection you wish to process.

## What is RAG?

RAG (Retrieval-Augmented Generation) is a framework that enhances the output of Large Language Models (LLMs) by incorporating relevant information retrieved from external documents. This helps in:

1. **Reducing hallucinations**: By using factual inputs, the model provides more grounded outputs.
2. **Customizing responses**: The LLM's knowledge is augmented with domain-specific data such as medical textbooks or customer support documents.

## Use Cases for RAG

- **Textbook Q&A**: Answer questions based on specific textbook references.
- **Customer Support Chatbots**: Enhance customer support with tailored document retrieval and response generation.
- **Internal Documentation Search**: Help employees find information faster by querying company documentation.
- **Email Chain Analysis**: Retrieve and summarize long email threads for better insights.

## Why Run Locally?

Running locally has multiple advantages:
- **Privacy**: No need to send sensitive data to an external API.
- **Speed**: Avoid API queues or downtime.
- **Cost**: Leverage your own hardware for long-term cost savings.

While APIs may perform better on general tasks, local models can be tailored for specific use cases.

## Key Terms

| Term                  | Description |
|-----------------------|-------------|
| **Token**             | Sub-word units that make up text. 1 token ≈ 4 characters. |
| **Embedding**         | A numerical representation of text or data. |
| **Similarity Search** | A method to find vectors close in high-dimensional space (e.g., cosine similarity). |
| **Large Language Model (LLM)** | A generative model trained to represent and generate text patterns. |
| **LLM Context Window**| The number of tokens an LLM can handle. GPT-4 has a context window of 32k tokens. |
| **Prompt**            | The input text to a generative LLM. |

---

Let me know if you'd like any further customization or additional sections!
