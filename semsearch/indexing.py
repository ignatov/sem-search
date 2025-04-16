"""
Indexing module for semantic search.

This module contains functionality for indexing and searching vectors.
"""

import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional

from semsearch.models import CodeUnit


class VectorIndex:
    """
    Vector index for semantic search using FAISS.
    
    This class handles building, saving, loading, and searching a vector index.
    """
    
    def __init__(self, dimensions=1536):
        """
        Initialize the vector index.
        
        Args:
            dimensions: The dimensions of the embedding vectors
        """
        self.dimensions = dimensions
        self.index = None
        self.id_to_code_unit = {}

    def build_index(self, embeddings: Dict[CodeUnit, np.ndarray]):
        """
        Build the vector index from embeddings.
        
        Args:
            embeddings: Dictionary mapping CodeUnit objects to their embedding vectors
        """
        vectors = []
        code_units = []

        for unit, vector in embeddings.items():
            # Check if vector has the expected dimension
            if len(vector) != self.dimensions:
                print(f"Warning: Vector dimension mismatch. Expected {self.dimensions}, got {len(vector)}.")
                # Resize vector to match expected dimensions
                if len(vector) > self.dimensions:
                    vector = vector[:self.dimensions]  # Truncate
                else:
                    # Pad with zeros
                    padded = np.zeros(self.dimensions)
                    padded[:len(vector)] = vector
                    vector = padded

            vectors.append(vector)
            code_units.append(unit)

        if vectors:
            # Create index
            self.index = faiss.IndexFlatL2(self.dimensions)

            vectors_array = np.array(vectors).astype('float32')
            self.index.add(vectors_array)

            self.id_to_code_unit = {i: unit for i, unit in enumerate(code_units)}

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[CodeUnit, float]]:
        """
        Search the index for the most similar vectors to the query vector.
        
        Args:
            query_vector: The query vector to search for
            top_k: The number of results to return
            
        Returns:
            List of tuples containing (CodeUnit, similarity_score)
        """
        query_vector = query_vector.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:  # FAISS may return -1 for no results
                continue
            distance = distances[0][i]
            similarity = 1.0 / (1.0 + distance)
            results.append((self.id_to_code_unit[int(idx)], similarity))

        return results

    def save(self, path: str):
        """
        Save the index to disk.
        
        Args:
            path: Path to save the index to (without extension)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.meta", "wb") as f:
            pickle.dump(self.id_to_code_unit, f)

    @classmethod
    def load(cls, path: str):
        """
        Load an index from disk.
        
        Args:
            path: Path to load the index from (without extension)
            
        Returns:
            VectorIndex instance with loaded index
        """
        index = faiss.read_index(f"{path}.index")
        with open(f"{path}.meta", "rb") as f:
            id_to_code_unit = pickle.load(f)

        instance = cls(dimensions=index.d)
        instance.index = index
        instance.id_to_code_unit = id_to_code_unit
        return instance