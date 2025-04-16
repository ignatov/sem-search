"""
Tests for the models module.
"""

import unittest
import hashlib
from semsearch.models import CodeUnit


class TestCodeUnit(unittest.TestCase):
    """Tests for the CodeUnit class."""

    def test_init(self):
        """Test initialization of CodeUnit."""
        unit = CodeUnit(
            path="path/to/file.py",
            content="def hello(): pass",
            unit_type="function",
            name="hello"
        )
        self.assertEqual(unit.path, "path/to/file.py")
        self.assertEqual(unit.content, "def hello(): pass")
        self.assertEqual(unit.unit_type, "function")
        self.assertEqual(unit.name, "hello")
        self.assertIsNone(unit.package)
        self.assertIsNone(unit.class_name)

    def test_init_with_optional_fields(self):
        """Test initialization of CodeUnit with optional fields."""
        unit = CodeUnit(
            path="path/to/file.java",
            content="public void hello() {}",
            unit_type="method",
            name="hello",
            package="com.example",
            class_name="MyClass"
        )
        self.assertEqual(unit.path, "path/to/file.java")
        self.assertEqual(unit.content, "public void hello() {}")
        self.assertEqual(unit.unit_type, "method")
        self.assertEqual(unit.name, "hello")
        self.assertEqual(unit.package, "com.example")
        self.assertEqual(unit.class_name, "MyClass")

    def test_get_content_hash(self):
        """Test get_content_hash method."""
        content = "def hello(): pass"
        unit = CodeUnit(
            path="path/to/file.py",
            content=content,
            unit_type="function",
            name="hello"
        )
        expected_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        self.assertEqual(unit.get_content_hash(), expected_hash)

    def test_equality(self):
        """Test equality comparison."""
        unit1 = CodeUnit(
            path="path/to/file.py",
            content="def hello(): pass",
            unit_type="function",
            name="hello"
        )
        unit2 = CodeUnit(
            path="path/to/file.py",
            content="def hello(): pass",
            unit_type="function",
            name="hello"
        )
        unit3 = CodeUnit(
            path="path/to/other.py",
            content="def hello(): pass",
            unit_type="function",
            name="hello"
        )
        self.assertEqual(unit1, unit2)
        self.assertNotEqual(unit1, unit3)
        self.assertNotEqual(unit1, "not a code unit")

    def test_hash(self):
        """Test hash function."""
        unit1 = CodeUnit(
            path="path/to/file.py",
            content="def hello(): pass",
            unit_type="function",
            name="hello"
        )
        unit2 = CodeUnit(
            path="path/to/file.py",
            content="def hello(): pass",
            unit_type="function",
            name="hello"
        )
        # Same units should have the same hash
        self.assertEqual(hash(unit1), hash(unit2))
        
        # Units can be used as dictionary keys
        d = {unit1: "value"}
        self.assertEqual(d[unit2], "value")


if __name__ == "__main__":
    unittest.main()