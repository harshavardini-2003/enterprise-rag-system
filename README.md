# enterprise-rag-system
Enterprise Document QA using RAG (FAISS + OpenAI/HuggingFace)
# Enterprise RAG System

This project implements a Retrieval-Augmented Generation (RAG) pipeline using:

- Sentence Transformers for embeddings
- FAISS for vector search
- GPT-2 for text generation

## Architecture

1. Document Loading
2. Embedding Creation
3. FAISS Vector Indexing
4. Query Embedding
5. Context Retrieval
6. LLM-based Answer Generation

## How to Run

pip install -r requirements.txt
python src/main.py
