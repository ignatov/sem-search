// Load available repositories when the page loads
window.onload = function() {
    fetch('/get_repositories')
        .then(response => response.json())
        .then(data => {
            const select = document.getElementById('repository-select');
            // Sort repositories by date if available
            if (data.repositories.length > 0 && data.repositories[0].timestamp) {
                data.repositories.sort((a, b) => {
                    return new Date(b.timestamp) - new Date(a.timestamp);
                });
            }
            data.repositories.forEach(repo => {
                const option = document.createElement('option');
                option.value = repo.full_name;  // Use full_name as the value for search functionality
                option.textContent = `${repo.name} (${repo.git_version || 'Unknown'})`;  // Display shortened name to the user
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

            // Create header with path and score
            const header = document.createElement('div');
            header.className = 'result-header';

            const path = document.createElement('div');
            path.className = 'result-path';
            path.textContent = result.path;

            const score = document.createElement('div');
            score.className = 'result-score';
            // Create a badge-like element for the score
            score.innerHTML = `<span class="inline-flex items-center rounded-full bg-blue-100 px-2.5 py-0.5 text-xs font-medium text-blue-800">
                Score: ${result.score.toFixed(2)}
            </span>`;

            header.appendChild(path);
            header.appendChild(score);

            // Create metadata section
            const meta = document.createElement('div');
            meta.className = 'result-meta';

            // Create a more structured metadata display
            let metaHTML = `<div class="flex flex-wrap gap-2 mb-2">`;

            // Add type badge
            metaHTML += `<span class="inline-flex items-center rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-medium text-gray-800">
                Type: ${result.unit_type}
            </span>`;

            // Add name badge
            metaHTML += `<span class="inline-flex items-center rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-medium text-gray-800">
                Name: ${result.name}
            </span>`;

            // Add package badge if available
            if (result.package) {
                metaHTML += `<span class="inline-flex items-center rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-medium text-gray-800">
                    Package: ${result.package}
                </span>`;
            }

            // Add class badge if available
            if (result.class_name) {
                metaHTML += `<span class="inline-flex items-center rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-medium text-gray-800">
                    Class: ${result.class_name}
                </span>`;
            }

            metaHTML += `</div>`;
            meta.innerHTML = metaHTML;

            // Create content section with syntax highlighting
            const content = document.createElement('pre');
            content.className = 'result-content';
            content.textContent = result.content;

            // Assemble the result item
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
