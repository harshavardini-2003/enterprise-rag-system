from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def generate_answer(context, question):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    result = generator(prompt, max_length=150, num_return_sequences=1)
    return result[0]['generated_text']
