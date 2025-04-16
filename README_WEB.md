# Semantic Code Search Web Interface

This is a simple web interface for the Semantic Code Search tool. It allows you to search through indexed code repositories using natural language queries.

## Prerequisites

- Python 3.9 or higher
- All dependencies listed in `requirements.txt`
- OpenAI API key (set in `.env` file or as environment variable)
- At least one indexed repository (see main README.md for indexing instructions)

## Installation

1. Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key in a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

## Running the Web Server

To start the web server, run:

```bash
python web_server.py
```

This will start a Flask server on http://localhost:5002

## Using the Web Interface

1. Open your browser and navigate to http://localhost:5002
2. Select a repository from the dropdown menu
3. Enter your search query in the text input field
4. Click the "Search" button
5. View the search results displayed below

## Features

- Repository selection with Git version information
- Natural language search queries
- Display of search results with:
  - File path
  - Code unit type (class, method, etc.)
  - Name and other metadata
  - Code content
  - Relevance score

## Troubleshooting

- If no repositories appear in the dropdown, make sure you have indexed at least one repository using the CLI tool
- If search fails, check that your OpenAI API key is correctly set
- For other issues, check the Flask server logs for error messages
