import os
import sys

# Add the parent directory to the path so we can import the semsearch package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semsearch.parsers import TreeSitterParser

# Initialize the parser
parser = TreeSitterParser()

# Print the loaded languages
print("Loaded languages:")
for lang_name in parser.languages:
    print(f"- {lang_name}")

print("\nLoaded parsers:")
for lang_name in parser.parsers:
    print(f"- {lang_name}")
