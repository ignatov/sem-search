"""
Tests for the indexing module.
"""

import unittest
import os
import tempfile
import numpy as np
from semsearch.models import CodeUnit
from semsearch.indexing import VectorIndex


class TestVectorIndex(unittest.TestCase):
    """Tests for the VectorIndex class."""

    def setUp(self):
        """Set up test fixtures."""
        self.index = VectorIndex(dimensions=4)
        self.code_unit1 = CodeUnit(
            path="path/to/file1.py",
            content="def hello(): pass",
            unit_type="function",
            name="hello"
        )
        self.code_unit2 = CodeUnit(
            path="path/to/file2.py",
            content="def world(): pass",
            unit_type="function",
            name="world"
        )
        self.embedding1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        self.embedding2 = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        self.embeddings = {
            self.code_unit1: self.embedding1,
            self.code_unit2: self.embedding2
        }

    def test_build_index(self):
        """Test build_index method."""
        self.index.build_index(self.embeddings)
        
        # Check that the index was built correctly
        self.assertIsNotNone(self.index.index)
        self.assertEqual(self.index.index.ntotal, 2)
        self.assertEqual(len(self.index.id_to_code_unit), 2)
        self.assertIn(0, self.index.id_to_code_unit)
        self.assertIn(1, self.index.id_to_code_unit)
        self.assertIn(self.index.id_to_code_unit[0], [self.code_unit1, self.code_unit2])
        self.assertIn(self.index.id_to_code_unit[1], [self.code_unit1, self.code_unit2])

    def test_search(self):
        """Test search method."""
        self.index.build_index(self.embeddings)
        
        # Search for something close to embedding1
        query_vector = np.array([0.15, 0.25, 0.35, 0.45], dtype=np.float32)
        results = self.index.search(query_vector, 2)
        
        # Check the results
        self.assertEqual(len(results), 2)
        # First result should be code_unit1 (closest to query)
        self.assertEqual(results[0][0], self.code_unit1)
        # Second result should be code_unit2
        self.assertEqual(results[1][0], self.code_unit2)
        # Scores should be between 0 and 1
        self.assertTrue(0 <= results[0][1] <= 1)
        self.assertTrue(0 <= results[1][1] <= 1)
        # First result should have higher score than second
        self.assertGreater(results[0][1], results[1][1])

    def test_save_and_load(self):
        """Test save and load methods."""
        self.index.build_index(self.embeddings)
        
        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "test_index")
            
            # Save the index
            self.index.save(index_path)
            
            # Check that the files were created
            self.assertTrue(os.path.exists(f"{index_path}.index"))
            self.assertTrue(os.path.exists(f"{index_path}.meta"))
            
            # Load the index
            loaded_index = VectorIndex.load(index_path)
            
            # Check that the loaded index has the same properties
            self.assertEqual(loaded_index.dimensions, self.index.dimensions)
            self.assertEqual(loaded_index.index.ntotal, self.index.index.ntotal)
            self.assertEqual(len(loaded_index.id_to_code_unit), len(self.index.id_to_code_unit))
            
            # Check that the loaded index can be searched
            query_vector = np.array([0.15, 0.25, 0.35, 0.45], dtype=np.float32)
            results = loaded_index.search(query_vector, 2)
            
            # Check the results
            self.assertEqual(len(results), 2)
            # Results should contain both code units
            result_units = [r[0] for r in results]
            self.assertIn(self.code_unit1, result_units)
            self.assertIn(self.code_unit2, result_units)

    def test_dimension_mismatch(self):
        """Test handling of dimension mismatches."""
        # Create an embedding with wrong dimensions
        wrong_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # Only 3 dimensions
        embeddings = {
            self.code_unit1: wrong_embedding,
            self.code_unit2: self.embedding2
        }
        
        # This should not raise an error, but should print a warning
        self.index.build_index(embeddings)
        
        # Check that the index was built correctly
        self.assertIsNotNone(self.index.index)
        self.assertEqual(self.index.index.ntotal, 2)


if __name__ == "__main__":
    unittest.main()