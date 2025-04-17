import os
import sys
import tempfile

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

# Check if Erlang is available
if 'erlang' in parser.languages:
    print("\nErlang language is available")

    # Create a simple Erlang file for testing
    with tempfile.NamedTemporaryFile(suffix='.erl', mode='w', delete=False) as f:
        f.write("""
-module(hello).
-export([hello_world/0]).

hello_world() ->
    io:format("Hello, World!~n").
        """)
        erlang_file_path = f.name

    print(f"Created temporary Erlang file: {erlang_file_path}")
    print(f"File content:\n{open(erlang_file_path).read()}")

    try:
        # Add debug code to print the AST structure
        import tree_sitter

        # Get the parser for Erlang
        erlang_parser = parser.parsers.get('erlang')
        if not erlang_parser:
            print("No Erlang parser found in parser.parsers")
            # Try to create a new parser
            erlang_parser = tree_sitter.Parser()
            erlang_parser.set_language(parser.languages['erlang'])

        # Parse the file
        with open(erlang_file_path, 'rb') as f:
            source_code = f.read()
            tree = erlang_parser.parse(source_code)

        # Print the AST structure
        print("\nAST structure:")

        def print_node(node, indent=0):
            content = source_code[node.start_byte:node.end_byte].decode('utf-8')
            # Truncate content if it's too long
            if len(content) > 50:
                content = content[:47] + "..."
            # Replace newlines with spaces
            content = content.replace("\n", " ")
            print(f"{' ' * indent}{node.type}: {content}")

            # Print the node's children
            for child in node.children:
                print_node(child, indent + 2)

        print_node(tree.root_node)

        # Parse the file with our parser
        code_units = parser.parse_file(erlang_file_path, os.path.dirname(erlang_file_path))

        # Print the code units
        print("\nParsed code units:")
        for unit in code_units:
            print(f"- {unit.unit_type}: {unit.name}")

        # Print the unit types
        unit_types = set(unit.unit_type for unit in code_units)
        print(f"\nUnit types: {unit_types}")

    finally:
        # Clean up the temporary file
        os.unlink(erlang_file_path)
else:
    print("\nErlang language is not available")
