# code-search-engine
Reads existing code-snippet/project, produces embeddings using Sentence BERT, Build an Annoy Index using generated embeddings and Use Ollama for Code Understanding and Generation

follows this course
<a href url=https://trainingportal.linuxfoundation.org/learn/course/building-a-code-search-engine-using-open-source/main/building-a-code-search-engine-using-open-source?page=1 />


Hands-On Exercise: Building Embeddings

To build our code search engine, we first need to generate embeddings for our code snippets. Embeddings are numerical representations of text that capture semantic meaning, making it possible to perform similarity searches.

The example provided in this lab exercise is based on the python, but generating embeddings with Sentence Bert, integrating Annoy for an efficient search and using Ollama for code understanding and generation are all language agnostic.

 
Step 1: Install Dependencies

Ensure you have the following dependencies installed:

“`bash
pip install ollama sentence-transformers annoy
“`

 
Step 2: Generate Embeddings with Sentence BERT

We’ll use Sentence BERT to generate embeddings for our code snippets.

We have provided a sample script to achieve this below. This script initializes a Sentence BERT model, generates embeddings for a list of code snippets, and saves them to a file.

“`python
from sentence_transformers import SentenceTransformer
import numpy as np
# Initialize Sentence BERT model
model = SentenceTransformer(‘sentence-transformers/all-MiniLM-L6-v2’)
# Sample code snippets
code_snippets = [
    “def add(a, b): return a + b”,
    “def multiply(a, b): return a * b”,
    “def subtract(a, b): return a – b”
]
# Generate embeddings
embeddings = model.encode(code_snippets)
# Save embeddings
np.save(‘code_embeddings.npy’, embeddings)
“`


Hands-On Exercise: Integrating Annoy for Efficient Search

Annoy (Approximate Nearest Neighbors Oh Yeah) is a library that enables efficient vector similarity searches. We’ll use it to search for code snippets based on their embeddings.

 

Build a Vector Store Based on Embedding For Every Matching File


Step 3: Build an Annoy Index

Next, using the script below, we’ll build an Annoy index using the embeddings generated earlier. This script initializes an Annoy index, adds the embeddings, and builds the index for efficient search.

“`python
from annoy import AnnoyIndex
import numpy as np
# Load embeddings
embeddings = np.load(‘code_embeddings.npy’)
# Initialize Annoy index
dimension = embeddings.shape[1]
annoy_index = AnnoyIndex(dimension, ‘angular’)
# Add embeddings to index
for i, embedding in enumerate(embeddings):
    annoy_index.add_item(i, embedding)
# Build the index
annoy_index.build(10)
annoy_index.save(‘code_search.ann’)
“`

Hands-On Exercise: Use Ollama for Code Understanding and Generation

Ollama is an open source LLM that can understand and generate code. We’ll use it to enhance our code search engine by providing code explanations and generation capabilities. One limitation of deploying Ollama locally is that we are restricted by the hardware capabilities of the laptop or desktop.
Step 4: Perform a Code Search and Generate Explanations

Let’s write the python script that performs a search and uses Codestral to generate explanations. This script integrates the components we’ve discussed to perform code searches and generate explanations using Ollama.

“`python
import json
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import ollama
# Initialize Sentence BERT model
model = SentenceTransformer(‘sentence-transformers/all-MiniLM-L6-v2’)
# Load Annoy index
annoy_index = AnnoyIndex(384, ‘angular’)
annoy_index.load(‘code_search.ann’)
# Load code snippets
with open(‘code_snippets.json’, ‘r’) as f:
    code_snippets = json.load(f)
def search_code(query, top_n=5):
    # Generate query embedding
    query_embedding = model.encode([query])[0]
    # Perform search
    indices = annoy_index.get_nns_by_vector(query_embedding, top_n)
    # Retrieve code snippets
    results = [code_snippets[i] for i in indices]
    return results
def explain_code(code):
    # Use Ollama to generate code explanation
    explanation = ollama.generate(code, model=’codestral’)
    return explanation
if __name__ == “__main__”:
    # Sample query
    query = “function to add two numbers”
    # Search for code
    search_results = search_code(query)
    print(“Search Results:”)
    for result in search_results:
        print(result)
    # Generate explanation for the first result
    explanation = explain_code(search_results[0])
    print(“Code Explanation:”)
    print(explanation)
“`


Implementation of this application is based on the work found on lablab.ai CodeBaseBuddy "Semantic Code Search Leveraging Ollama and Codestral,https://lablab.ai/event/codestral-ai-hackathon/codebasebuddy/codebasebuddy" and "Open Interpreter Hackathon" https://lablab.ai/event/open-interpreter-hackathon/githubbuddy/codebasebuddy.

