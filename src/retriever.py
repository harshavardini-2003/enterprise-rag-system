import faiss
import numpy as np

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def retrieve(query_embedding, index, sentences, k=2):
    D, I = index.search(np.array([query_embedding]), k)
    results = [sentences[i] for i in I[0]]
    return results
