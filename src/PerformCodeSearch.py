import json
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import ollama
# Initialize Sentence BERT model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Load Annoy index
annoy_index = AnnoyIndex(384, 'angular')
annoy_index.load('code_search.ann')
# Load code snippets
with open('code_snippets.json', 'r') as f:
    code_snippets = json.load(f)

#Search for code
def search_code(query, top_n=5):
    # Generate query embedding
    query_embedding = model.encode([query])[0]
    # Perform search
    indices = annoy_index.get_nns_by_vector(query_embedding, top_n)
    # Retrieve code snippets
    results = [code_snippets[i] for i in indices]
    return results

#explain code
def explain_code(code):
    # Use Ollama to generate code explanation
    explanation = ollama.generate('codestral', code)
    return explanation


if __name__ == "__main__":
    # Sample query
    query = "function to add two numbers"
    # Search for code
    search_results = search_code(query)
    print("Search Results:")
    for result in search_results:
        print(result)
    # Generate explanation for the first result
    explanation = explain_code(search_results[0])
    print("Code Explanation:")
    print(explanation)
