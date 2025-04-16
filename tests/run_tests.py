#!/usr/bin/env python3
"""
Test runner for the semantic search system.

This script runs all the tests for the semantic search system.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import the semsearch package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all the test modules
from tests.test_models import TestCodeUnit
from tests.test_embedding import TestCodeEmbedder
from tests.test_indexing import TestVectorIndex
from tests.test_search import TestSearchEngine
from tests.test_parsers import TestGenericFileParser, TestUnifiedParser, TestJavaParser
from tests.test_erlang_files import TestErlangFiles
from tests.test_functional import TestFunctionalWorkflow


def run_tests():
    """Run all the tests."""
    # Create a test suite
    test_suite = unittest.TestSuite()

    # Add all the test cases
    test_suite.addTest(unittest.makeSuite(TestCodeUnit))
    test_suite.addTest(unittest.makeSuite(TestCodeEmbedder))
    test_suite.addTest(unittest.makeSuite(TestVectorIndex))
    test_suite.addTest(unittest.makeSuite(TestSearchEngine))
    test_suite.addTest(unittest.makeSuite(TestGenericFileParser))
    test_suite.addTest(unittest.makeSuite(TestUnifiedParser))
    test_suite.addTest(unittest.makeSuite(TestJavaParser))
    test_suite.addTest(unittest.makeSuite(TestErlangFiles))
    test_suite.addTest(unittest.makeSuite(TestFunctionalWorkflow))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Return the result
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
