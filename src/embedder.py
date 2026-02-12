from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(text):
    sentences = text.split("\n")
    embeddings = model.encode(sentences)
    return sentences, embeddings
