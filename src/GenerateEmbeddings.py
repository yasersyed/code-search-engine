from sentence_transformers import SentenceTransformer
import numpy as np
# Initialize Sentence BERT model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Sample code snippets
code_snippets = [
    "def add(a, b): return a + b",
    "def multiply(a, b): return a * b",
    "def subtract(a, b): return a – b"
]
# Generate embeddings
embeddings = model.encode(code_snippets)
# Save embeddings
np.save('code_embeddings.npy', embeddings)
