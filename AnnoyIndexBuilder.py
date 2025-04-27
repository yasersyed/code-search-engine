from annoy import AnnoyIndex
import numpy as np
# Load embeddings
embeddings = np.load('code_embeddings.npy')
# initialize Annoy index
dimension = embeddings.shape[1]
annoy_index = AnnoyIndex(dimension, 'angular')
# Add embeddings to index
for i, embedding in enumerate(embeddings):
    annoy_index.add_item(i, embedding)
# Build the index
annoy_index.build(10)
annoy_index.save('code_search.ann')
