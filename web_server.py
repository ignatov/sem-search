import os
import datetime
from flask import Flask, render_template, request, jsonify
from semsearch import list_available_indexes, search, display_results, CodeUnit
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

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
                    git_version = f"{commit_dir[:8]} (modified)"
                    # Use current time for modified versions
                    timestamp = datetime.datetime.now().isoformat()
                else:
                    git_version = commit_dir[:8]  # Show first 8 chars of commit hash
                    # Try to get timestamp from the directory name or file metadata
                    try:
                        # Get the creation time of the index directory
                        index_path = os.path.join(".semsearch", repo_dir, commit_dir)
                        if os.path.exists(index_path):
                            timestamp = datetime.datetime.fromtimestamp(
                                os.path.getctime(index_path)
                            ).isoformat()
                        else:
                            timestamp = datetime.datetime.now().isoformat()
                    except:
                        timestamp = datetime.datetime.now().isoformat()

                # Store the shortened name (up to the first dot) and the full name
                repositories.append({
                    'name': name,  # Shortened name (up to the first dot)
                    'full_name': index_name,  # Original full name
                    'git_version': git_version,
                    'timestamp': timestamp
                })
            else:
                # For names without parts, use the full name
                repositories.append({
                    'name': index_name,  # No dot to split on, use full name
                    'full_name': index_name,  # Original full name
                    'git_version': 'Unknown',
                    'timestamp': datetime.datetime.now().isoformat()
                })
        else:
            # Old-style index
            parts = index_name.split('.')
            if len(parts) > 1:
                name = parts[0]
                hash_part = parts[1][:8]  # Shorten hash to 8 chars

                # Try to get timestamp from file metadata
                try:
                    index_path = os.path.join(".semsearch", index_name)
                    if os.path.exists(index_path):
                        timestamp = datetime.datetime.fromtimestamp(
                            os.path.getctime(index_path)
                        ).isoformat()
                    else:
                        timestamp = datetime.datetime.now().isoformat()
                except:
                    timestamp = datetime.datetime.now().isoformat()

                # Store the shortened name (up to the first dot) and the full name
                repositories.append({
                    'name': name,  # Shortened name (up to the first dot)
                    'full_name': index_name,  # Original full name
                    'git_version': hash_part,
                    'timestamp': timestamp
                })
            else:
                # For names without parts, use the full name
                repositories.append({
                    'name': index_name,  # No dot to split on, use full name
                    'full_name': index_name,  # Original full name
                    'git_version': 'Unknown',
                    'timestamp': datetime.datetime.now().isoformat()
                })

    # Sort repositories by timestamp (newest first)
    repositories.sort(key=lambda x: x['timestamp'], reverse=True)

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
