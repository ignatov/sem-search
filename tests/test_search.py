"""
Tests for the search module.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from semsearch.models import CodeUnit
from semsearch.search import SearchEngine


class TestSearchEngine(unittest.TestCase):
    """Tests for the SearchEngine class."""

    def setUp(self):
        """Set up test fixtures."""
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
        self.embedding1 = np.array([0.1, 0.2, 0.3, 0.4])
        self.embedding2 = np.array([0.5, 0.6, 0.7, 0.8])
        
        # Mock embedder
        self.embedder = MagicMock()
        self.embedder.embed_query.return_value = np.array([0.15, 0.25, 0.35, 0.45])
        
        # Mock index
        self.index = MagicMock()
        self.index.search.return_value = [
            (self.code_unit1, 0.9),
            (self.code_unit2, 0.7)
        ]
        
        # Create search engine
        self.search_engine = SearchEngine(self.embedder, self.index)

    def test_search(self):
        """Test search method."""
        # Call the method
        results = self.search_engine.search("test query", 2)
        
        # Check that the embedder was called correctly
        self.embedder.embed_query.assert_called_once_with("test query")
        
        # Check that the index was called correctly
        self.index.search.assert_called_once()
        np.testing.assert_array_equal(
            self.index.search.call_args[0][0],
            np.array([0.15, 0.25, 0.35, 0.45])
        )
        self.assertEqual(self.index.search.call_args[0][1], 2)
        
        # Check the results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], self.code_unit1)
        self.assertEqual(results[0][1], 0.9)
        self.assertEqual(results[1][0], self.code_unit2)
        self.assertEqual(results[1][1], 0.7)


if __name__ == "__main__":
    unittest.main()