// Function to get previous queries from localStorage
function getPreviousQueries() {
    const queries = localStorage.getItem('previousQueries');
    return queries ? JSON.parse(queries) : [];
}

// Function to save a query to localStorage
function saveQuery(query) {
    if (!query.trim()) return; // Don't save empty queries

    let queries = getPreviousQueries();

    // Remove the query if it already exists (to avoid duplicates)
    queries = queries.filter(q => q !== query);

    // Add the new query to the beginning of the array
    queries.unshift(query);

    // Limit to 10 queries
    if (queries.length > 10) {
        queries = queries.slice(0, 10);
    }

    localStorage.setItem('previousQueries', JSON.stringify(queries));
}

// Variable to track the currently selected item in the dropdown
let selectedIndex = -1;

// Function to display previous queries dropdown
function displayPreviousQueries() {
    const queries = getPreviousQueries();
    const container = document.getElementById('previous-queries');

    // Clear the container
    container.innerHTML = '';
    // Reset selected index
    selectedIndex = -1;

    if (queries.length === 0) {
        container.classList.add('hidden');
        return;
    }

    // Add each query to the dropdown
    queries.forEach((query, index) => {
        const item = document.createElement('div');
        item.className = 'p-2 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer text-left';
        item.dataset.index = index;
        item.textContent = query;
        item.addEventListener('click', function() {
            document.getElementById('search-input').value = query;
            container.classList.add('hidden');
        });
        container.appendChild(item);
    });

    // Show the dropdown
    container.classList.remove('hidden');
}

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

            // Preselect the first (latest) repository if available
            if (data.repositories.length > 0) {
                select.value = data.repositories[0].full_name;
            }

            // Add event listener for keyboard navigation on search input
            const searchInput = document.getElementById('search-input');
            const previousQueries = document.getElementById('previous-queries');

            searchInput.addEventListener('keydown', function(event) {
                // Only handle keyboard navigation if the dropdown is visible
                if (!previousQueries.classList.contains('hidden')) {
                    const items = previousQueries.querySelectorAll('div');

                    if (event.key === 'ArrowDown') {
                        event.preventDefault();
                        // Move selection down
                        selectedIndex = Math.min(selectedIndex + 1, items.length - 1);
                        updateSelectedItem(items);
                    } else if (event.key === 'ArrowUp') {
                        event.preventDefault();
                        // Move selection up
                        selectedIndex = Math.max(selectedIndex - 1, -1);
                        updateSelectedItem(items);
                    } else if (event.key === 'Enter' && selectedIndex >= 0) {
                        event.preventDefault();
                        // Select the highlighted item
                        if (items[selectedIndex]) {
                            searchInput.value = items[selectedIndex].textContent;
                            previousQueries.classList.add('hidden');
                            selectedIndex = -1;
                        }
                    } else if (event.key === 'Escape') {
                        // Hide dropdown on Escape
                        previousQueries.classList.add('hidden');
                        selectedIndex = -1;
                    }
                }

                // If Enter is pressed and no item is selected, perform search
                if (event.key === 'Enter' && (previousQueries.classList.contains('hidden') || selectedIndex === -1)) {
                    event.preventDefault(); // Prevent form submission if inside a form
                    performSearch();
                }
            });

            // Function to update the visual indication of the selected item
            function updateSelectedItem(items) {
                // Remove highlight from all items
                items.forEach(item => {
                    item.classList.remove('bg-gray-100', 'dark:bg-gray-700');
                });

                // Add highlight to the selected item
                if (selectedIndex >= 0 && items[selectedIndex]) {
                    items[selectedIndex].classList.add('bg-gray-100', 'dark:bg-gray-700');
                    // Ensure the selected item is visible (scroll into view if needed)
                    items[selectedIndex].scrollIntoView({ block: 'nearest' });
                }
            }

            // Add event listener for focus on search input to show previous queries
            searchInput.addEventListener('focus', function() {
                displayPreviousQueries();
            });

            // Hide dropdown when clicking outside
            document.addEventListener('click', function(event) {
                if (!searchInput.contains(event.target) && !previousQueries.contains(event.target)) {
                    previousQueries.classList.add('hidden');
                }
            });

            // Add event listener for input to filter previous queries
            searchInput.addEventListener('input', function() {
                const value = this.value.toLowerCase();
                const queries = getPreviousQueries();

                if (!value) {
                    displayPreviousQueries();
                    return;
                }

                const filteredQueries = queries.filter(query => 
                    query.toLowerCase().includes(value)
                );

                const container = document.getElementById('previous-queries');
                container.innerHTML = '';

                if (filteredQueries.length === 0) {
                    container.classList.add('hidden');
                    return;
                }

                // Reset selected index
                selectedIndex = -1;

                filteredQueries.forEach((query, index) => {
                    const item = document.createElement('div');
                    item.className = 'p-2 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer text-left';
                    item.dataset.index = index;
                    item.textContent = query;
                    item.addEventListener('click', function() {
                        searchInput.value = query;
                        container.classList.add('hidden');
                    });
                    container.appendChild(item);
                });

                container.classList.remove('hidden');
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

    // Save the query to localStorage
    saveQuery(query);

    // Hide the previous queries dropdown
    document.getElementById('previous-queries').classList.add('hidden');

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
