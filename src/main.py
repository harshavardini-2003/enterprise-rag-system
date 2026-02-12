from loader import load_documents
from embedder import create_embeddings, model
from retriever import create_faiss_index, retrieve
from generator import generate_answer

def run_rag():
    text = load_documents("../data/sample_document.txt")
    sentences, embeddings = create_embeddings(text)

    index = create_faiss_index(embeddings)

    question = input("Enter your question: ")

    query_embedding = model.encode([question])
    retrieved_context = retrieve(query_embedding, index, sentences)

    context = " ".join(retrieved_context)
    answer = generate_answer(context, question)

    print("\nGenerated Answer:\n")
    print(answer)

if __name__ == "__main__":
    run_rag()
