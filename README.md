# Explore Singapore
### *Legal, Historical, and Infrastructural Knowledge Engine*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-API-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![LangChain](https://img.shields.io/badge/🦜🔗_LangChain-RAG-1C3C3C?style=for-the-badge)](https://python.langchain.com/)
[![FAISS](https://img.shields.io/badge/Meta_FAISS-Vector_Search-0081CB?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/)

## 📌 Project Overview
The **Singapore Intelligence RAG System** is an advanced AI-powered platform designed to provide precise, context-aware information regarding Singapore’s legal framework, government policies, historical milestones, and critical infrastructure.

Unlike standard LLMs which can "hallucinate" facts, this system uses **Retrieval-Augmented Generation (RAG)**. It references a strictly curated corpus of Singaporean data (33,000+ pages of PDFs) to ground every answer in factual reality before responding.


---

## 🏗 System Architecture
The system follows a high-performance RAG pipeline optimized for low-resource environments:

1.  **Ingestion:** Processed 33,000+ pages of Singaporean legal and historical documents.
2.  **Vectorization:** Used `all-MiniLM-L6-v2` to create 384-dimensional semantic embeddings.
3.  **Retrieval:** Implemented **FAISS (Facebook AI Similarity Search)** for millisecond-latency vector lookups.
4.  **Generation:** A "Triple-Failover" logic ensures 99.9% uptime.

---

## 🚀 Key Features

### **1. Triple-AI Failover Backend**
To ensure reliability during demos and high traffic, the system creates a resilient chain of command for LLM inference:
* **Primary:** Google Gemini 2.0 Flash (Fastest, High Context)
* **Secondary:** Llama 3.3 70B via OpenRouter (Robust fallback)
* **Tertiary:** Llama 3.3 70B via Groq (Emergency backup)

### **2. "Lquid-Glass" Interactive UI**
The frontend interface is a custom-built **Framer Code Component** (React + Framer Motion).
* **Glassmorphism:** Real-time backdrop blur (`backdrop-filter: blur(25px)`).
* **Spring Physics:** Smooth sideways expansion on hover.
* **Minimalist Design:** SVG iconography and San Francisco typography.

### **3. Local Embedding Inference**
Instead of relying on external API calls for vectorization (which adds latency and cost), the embedding model runs **locally** within the application container, ensuring data privacy and speed.

---

## 🛠 Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | React, Framer Motion | Interactive "Ask AI" widget. |
| **Backend** | Flask, Gunicorn | REST API handling RAG logic. |
| **Vector DB** | FAISS (CPU) | Local, high-speed similarity search. |
| **Embeddings** | Sentence-Transformers | `all-MiniLM-L6-v2` (Local(server based)). |
| **LLMs** | Gemini 2.0, Llama 3.3 | Text generation and synthesis. |
| **Deployment** | Hugging Face Spaces | Docker-based cloud hosting. |

---

## ⚙️ Installation & Local Setup

### **Prerequisites**
* Python 3.10+
* Git

### **1. Clone the Repository**
```bash
git clone [git clone https://github.com/adityaprasad-sudo/Explore-Singapore.git)
