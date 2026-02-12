# Enterprise RAG System (Retrieval-Augmented Generation)

## 📌 Overview

This project implements an end-to-end Retrieval-Augmented Generation (RAG) pipeline that enhances LLM responses by retrieving relevant contextual information before generating answers.

RAG improves factual grounding by combining:

1. Dense vector retrieval
2. Semantic search
3. Large Language Model generation

---

## 🏗️ Architecture

User Query
   ↓
Query Embedding (Sentence Transformer)
   ↓
FAISS Vector Search
   ↓
Top-K Relevant Context Retrieval
   ↓
LLM Prompt Construction
   ↓
Generated Answer

---

## 🧠 What is RAG?

Retrieval-Augmented Generation (RAG) is a hybrid architecture that:

- Retrieves relevant information from a knowledge base
- Injects retrieved context into a prompt
- Uses an LLM to generate grounded responses

This reduces hallucination and improves response accuracy.

---

## 🔎 Embedding Model Used

Model: `all-MiniLM-L6-v2`  
Library: Sentence Transformers  

This model converts text into dense vector representations for semantic similarity search.

---

## 📦 Vector Database

FAISS (Facebook AI Similarity Search)

Index Type: `IndexFlatL2`

- Uses L2 (Euclidean) distance
- Suitable for smaller datasets
- Performs exact nearest neighbor search

---

## 🤖 LLM Used

Model: GPT-2  
Library: HuggingFace Transformers  

Used for text generation based on retrieved context.

---

## ⚙️ Technologies

- Python
- Sentence Transformers
- FAISS
- HuggingFace Transformers
- Modular architecture

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/main.py
