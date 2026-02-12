import faiss
import numpy as np

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    # Using IndexFlatL2 for exact nearest neighbor search
    # L2 = Euclidean distance metric
    # Best suited for small-to-medium sized datasets
    # Provides exact search (not approximate)
    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings).astype("float32"))
    return index


def retrieve(query_embedding, index, sentences, k=2):
    D, I = index.search(np.array(query_embedding).astype("float32"), k)
    results = [sentences[i] for i in I[0]]
    return results
