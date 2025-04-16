"""
Models module for semantic search.

This module contains the data models used throughout the semantic search system.
"""

from dataclasses import dataclass
from typing import Optional
import hashlib


@dataclass(frozen=True)  # Make it immutable and hashable
class CodeUnit:
    """
    Represents a unit of code (class, method, file, etc.) for semantic search.
    
    Attributes:
        path: The file path relative to the repository root
        content: The actual code content
        unit_type: The type of code unit (class, method, field, file, etc.)
        name: The name of the code unit
        package: Optional package name (for languages with package concepts)
        class_name: Optional class name (for methods within classes)
    """
    path: str
    content: str
    unit_type: str  # class, method, field, etc.
    name: str
    package: Optional[str] = None
    class_name: Optional[str] = None

    def get_content_hash(self) -> str:
        """
        Generate a stable hash of the content for caching purposes.
        
        Returns:
            A hexadecimal string representing the MD5 hash of the content
        """
        # Use a more stable hashing method that won't change between sessions
        return hashlib.md5(self.content.encode('utf-8')).hexdigest()

    def __hash__(self):
        # Create a hash based on all fields to make the class hashable
        return hash((self.path, self.name, self.unit_type, self.package, self.class_name))

    def __eq__(self, other):
        if not isinstance(other, CodeUnit):
            return False
        return (self.path == other.path and 
                self.name == other.name and 
                self.unit_type == other.unit_type and 
                self.package == other.package and 
                self.class_name == other.class_name)