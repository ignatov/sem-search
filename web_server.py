import os
from flask import Flask, render_template_string, request, jsonify
from semantic_search import list_available_indexes, search, display_results, CodeUnit
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# HTML template for the main page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Semantic Code Search</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            background-color: #f5f5f5;
        }
        .container {
            width: 80%;
            max-width: 800px;
            margin: 50px auto;
            text-align: center;
        }
        h1 {
            color: #333;
        }
        .search-box {
            margin: 30px 0;
        }
        input[type="text"] {
            width: 70%;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        select {
            width: 70%;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        .results {
            width: 100%;
            text-align: left;
            margin-top: 30px;
        }
        .result-item {
            background-color: white;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .result-path {
            font-weight: bold;
            color: #333;
        }
        .result-score {
            color: #666;
        }
        .result-meta {
            color: #666;
            margin-bottom: 10px;
        }
        .result-content {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 4px;
            white-space: pre-wrap;
            overflow-x: auto;
        }
        .loading {
            display: none;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Semantic Code Search</h1>

        <div class="search-box">
            <select id="repository-select">
                <option value="">Select a repository...</option>
            </select>
            <br>
            <input type="text" id="search-input" placeholder="Enter your search query...">
            <br><br>
            <button onclick="performSearch()">Search</button>
        </div>

        <div class="loading" id="loading">
            Searching...
        </div>

        <div class="results" id="results">
            <!-- Results will be displayed here -->
        </div>
    </div>

    <script>
        // Load available repositories when the page loads
        window.onload = function() {
            fetch('/get_repositories')
                .then(response => response.json())
                .then(data => {
                    const select = document.getElementById('repository-select');
                    data.repositories.forEach(repo => {
                        const option = document.createElement('option');
                        option.value = repo.name;
                        option.textContent = `${repo.name} (${repo.git_version || 'Unknown'})`;
                        select.appendChild(option);
                    });
                });
        };

        function performSearch() {
            const query = document.getElementById('search-input').value;
            const repository = document.getElementById('repository-select').value;

            if (!query) {
                alert('Please enter a search query');
                return;
            }

            if (!repository) {
                alert('Please select a repository');
                return;
            }

            // Show loading indicator
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';

            fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    repository: repository
                }),
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                document.getElementById('loading').style.display = 'none';

                // Display results
                const resultsContainer = document.getElementById('results');
                resultsContainer.innerHTML = '';

                if (data.results.length === 0) {
                    resultsContainer.innerHTML = '<p>No results found.</p>';
                    return;
                }

                data.results.forEach(result => {
                    const resultItem = document.createElement('div');
                    resultItem.className = 'result-item';

                    const header = document.createElement('div');
                    header.className = 'result-header';

                    const path = document.createElement('div');
                    path.className = 'result-path';
                    path.textContent = result.path;

                    const score = document.createElement('div');
                    score.className = 'result-score';
                    score.textContent = `Score: ${result.score.toFixed(2)}`;

                    header.appendChild(path);
                    header.appendChild(score);

                    const meta = document.createElement('div');
                    meta.className = 'result-meta';
                    meta.innerHTML = `Type: ${result.unit_type}, Name: ${result.name}`;

                    if (result.package) {
                        meta.innerHTML += `<br>Package: ${result.package}`;
                    }

                    if (result.class_name) {
                        meta.innerHTML += `<br>Class: ${result.class_name}`;
                    }

                    const content = document.createElement('pre');
                    content.className = 'result-content';
                    content.textContent = result.content;

                    resultItem.appendChild(header);
                    resultItem.appendChild(meta);
                    resultItem.appendChild(content);

                    resultsContainer.appendChild(resultItem);
                });
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('loading').style.display = 'none';
                document.getElementById('results').innerHTML = '<p>An error occurred while searching. Please try again.</p>';
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/get_repositories')
def get_repositories():
    # Get available indexes
    indexes = list_available_indexes()
    repositories = []

    for index_name in indexes:
        # Parse repo name and git version from index_name
        if "/" in index_name:
            repo_dir, commit_dir = index_name.split("/", 1)
            parts = repo_dir.split('.')
            if len(parts) > 1:
                name = parts[0]
                hash_part = parts[1]

                # Check if this is a modified version
                if commit_dir.endswith("-latest"):
                    git_version = f"{commit_dir[:-7]} (modified)"
                else:
                    git_version = commit_dir[:8]  # Show first 8 chars of commit hash

                repositories.append({
                    'name': index_name,
                    'git_version': git_version
                })
            else:
                repositories.append({
                    'name': index_name,
                    'git_version': 'Unknown'
                })
        else:
            # Old-style index
            parts = index_name.split('.')
            if len(parts) > 1:
                name = parts[0]
                hash_part = parts[1]
                repositories.append({
                    'name': index_name,
                    'git_version': hash_part
                })
            else:
                repositories.append({
                    'name': index_name,
                    'git_version': 'Unknown'
                })

    return jsonify({'repositories': repositories})

@app.route('/search', methods=['POST'])
def perform_search():
    data = request.json
    query = data.get('query')
    repository = data.get('repository')

    if not query or not repository:
        return jsonify({'error': 'Missing query or repository'}), 400

    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({'error': 'OpenAI API key not found'}), 500

    # Perform search
    try:
        results = search(query, 10, api_key, repository)

        # Check if results is None
        if results is None:
            print(f"No results found for query: {query} in repository: {repository}")
            return jsonify({'results': []})

        # Format results for JSON response
        formatted_results = []
        for unit, score in results:
            formatted_results.append({
                'path': unit.path,
                'unit_type': unit.unit_type,
                'name': unit.name,
                'package': unit.package,
                'class_name': unit.class_name,
                'content': unit.content[:1000] if len(unit.content) > 1000 else unit.content,
                'score': float(score)
            })

        return jsonify({'results': formatted_results})
    except Exception as e:
        import traceback
        print(f"Error in search: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
