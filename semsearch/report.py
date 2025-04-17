"""
Report generation module for semantic search.

This module provides functionality to generate HTML reports with detailed information
about file processing, code units, and parsing statistics.
"""

import os
import time
import datetime
from typing import Dict, List, Optional
import json
from jinja2 import Template

from semsearch.models import CodeUnit


class ReportGenerator:
    """Generator for HTML reports about file processing and code units."""

    def __init__(self, repo_path: str):
        """
        Initialize the report generator.

        Args:
            repo_path: Path to the repository
        """
        self.repo_path = repo_path
        self.report_dir = os.path.join(repo_path, ".semsearch", "reports")
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Initialize timing data
        self.start_time = time.time()
        self.timings = {
            "parsing": {},
            "embedding": {},
            "indexing": {},
            "total": 0
        }
        
        # Initialize file processing data
        self.file_processing = {
            "total_files": 0,
            "processed_files": [],
            "skipped_files": [],
            "error_files": []
        }
        
        # Initialize code unit data
        self.code_units_data = {
            "total": 0,
            "by_type": {},
            "by_language": {},
            "largest_units": []
        }

    def start_timing(self, category: str, label: str):
        """
        Start timing a specific operation.

        Args:
            category: Category of the operation (parsing, embedding, indexing)
            label: Label for the specific operation
        """
        if category not in self.timings:
            self.timings[category] = {}
        
        self.timings[category][label] = {
            "start": time.time(),
            "end": None,
            "duration": None
        }

    def end_timing(self, category: str, label: str):
        """
        End timing a specific operation.

        Args:
            category: Category of the operation (parsing, embedding, indexing)
            label: Label for the specific operation
        """
        if category in self.timings and label in self.timings[category]:
            self.timings[category][label]["end"] = time.time()
            self.timings[category][label]["duration"] = (
                self.timings[category][label]["end"] - self.timings[category][label]["start"]
            )

    def record_file_processing(self, file_path: str, status: str, details: Optional[Dict] = None):
        """
        Record information about a processed file.

        Args:
            file_path: Path to the file
            status: Status of the processing (processed, skipped, error)
            details: Additional details about the processing
        """
        relative_path = os.path.relpath(file_path, self.repo_path)
        
        file_info = {
            "path": relative_path,
            "extension": os.path.splitext(file_path)[1],
            "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            "details": details or {}
        }
        
        if status == "processed":
            self.file_processing["processed_files"].append(file_info)
        elif status == "skipped":
            self.file_processing["skipped_files"].append(file_info)
        elif status == "error":
            self.file_processing["error_files"].append(file_info)
            
        self.file_processing["total_files"] += 1

    def record_code_units(self, code_units: List[CodeUnit]):
        """
        Record information about code units.

        Args:
            code_units: List of code units
        """
        self.code_units_data["total"] = len(code_units)
        
        # Track by type
        for unit in code_units:
            unit_type = unit.unit_type
            if unit_type in self.code_units_data["by_type"]:
                self.code_units_data["by_type"][unit_type]["count"] += 1
                self.code_units_data["by_type"][unit_type]["size"] += len(unit.content)
            else:
                self.code_units_data["by_type"][unit_type] = {
                    "count": 1,
                    "size": len(unit.content)
                }
            
            # Track by language (based on file extension)
            extension = os.path.splitext(unit.path)[1]
            language = extension[1:] if extension else "unknown"  # Remove the dot
            
            if language in self.code_units_data["by_language"]:
                self.code_units_data["by_language"][language]["count"] += 1
                self.code_units_data["by_language"][language]["size"] += len(unit.content)
            else:
                self.code_units_data["by_language"][language] = {
                    "count": 1,
                    "size": len(unit.content)
                }
        
        # Track largest units (top 10)
        sorted_units = sorted(code_units, key=lambda u: len(u.content), reverse=True)
        for unit in sorted_units[:10]:
            self.code_units_data["largest_units"].append({
                "path": unit.path,
                "name": unit.name,
                "type": unit.unit_type,
                "size": len(unit.content),
                "package": unit.package,
                "class_name": unit.class_name
            })

    def record_parser_stats(self, stats: Dict):
        """
        Record parser statistics.

        Args:
            stats: Parser statistics
        """
        self.parser_stats = stats

    def generate_report(self) -> str:
        """
        Generate an HTML report.

        Returns:
            Path to the generated HTML report
        """
        # End total timing
        self.timings["total"] = time.time() - self.start_time
        
        # Create report data
        report_data = {
            "repository": self.repo_path,
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "timings": self.timings,
            "file_processing": self.file_processing,
            "code_units": self.code_units_data,
            "parser_stats": getattr(self, 'parser_stats', {})
        }
        
        # Generate HTML report using template
        html = self._render_html_report(report_data)
        
        # Save the report
        report_filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = os.path.join(self.report_dir, report_filename)
        
        with open(report_path, 'w') as f:
            f.write(html)
        
        # Also save the raw data as JSON for potential future use
        json_path = os.path.join(self.report_dir, f"report_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return report_path

    def _render_html_report(self, data: Dict) -> str:
        """
        Render the HTML report using a template.

        Args:
            data: Report data

        Returns:
            HTML content
        """
        template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Semantic Search Processing Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .section {
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            background-color: #f9f9f9;
        }
        .summary {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .summary-item {
            flex: 1;
            min-width: 200px;
            background-color: #fff;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .summary-item h3 {
            margin-top: 0;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .chart-container {
            height: 300px;
            margin-bottom: 30px;
        }
        .error {
            color: #e74c3c;
        }
        .success {
            color: #2ecc71;
        }
        .warning {
            color: #f39c12;
        }
        .code {
            font-family: monospace;
            background-color: #f8f8f8;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 0.9em;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Semantic Search Processing Report</h1>
    
    <div class="section">
        <h2>Overview</h2>
        <p>Repository: <span class="code">{{ data.repository }}</span></p>
        <p>Generated: {{ data.timestamp }}</p>
        
        <div class="summary">
            <div class="summary-item">
                <h3>Files</h3>
                <p>Total: {{ data.file_processing.total_files }}</p>
                <p>Processed: {{ data.file_processing.processed_files|length }}</p>
                <p>Skipped: {{ data.file_processing.skipped_files|length }}</p>
                <p>Errors: {{ data.file_processing.error_files|length }}</p>
            </div>
            
            <div class="summary-item">
                <h3>Code Units</h3>
                <p>Total: {{ data.code_units.total }}</p>
                {% for type, info in data.code_units.by_type.items() %}
                <p>{{ type }}: {{ info.count }}</p>
                {% endfor %}
            </div>
            
            <div class="summary-item">
                <h3>Timing</h3>
                <p>Total: {{ "%.2f"|format(data.timings.total) }} seconds</p>
                {% if data.parser_stats.parsing_time %}
                <p>Parsing: {{ "%.2f"|format(data.parser_stats.parsing_time) }} seconds</p>
                {% endif %}
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>File Processing</h2>
        
        <div class="chart-container">
            <canvas id="fileTypesChart"></canvas>
        </div>
        
        <h3>Processed Files ({{ data.file_processing.processed_files|length }})</h3>
        {% if data.file_processing.processed_files %}
        <table>
            <thead>
                <tr>
                    <th>Path</th>
                    <th>Extension</th>
                    <th>Size (bytes)</th>
                </tr>
            </thead>
            <tbody>
                {% for file in data.file_processing.processed_files[:50] %}
                <tr>
                    <td>{{ file.path }}</td>
                    <td>{{ file.extension }}</td>
                    <td>{{ file.size }}</td>
                </tr>
                {% endfor %}
                {% if data.file_processing.processed_files|length > 50 %}
                <tr>
                    <td colspan="3">... and {{ data.file_processing.processed_files|length - 50 }} more files</td>
                </tr>
                {% endif %}
            </tbody>
        </table>
        {% else %}
        <p>No files were processed.</p>
        {% endif %}
        
        <h3>Error Files ({{ data.file_processing.error_files|length }})</h3>
        {% if data.file_processing.error_files %}
        <table>
            <thead>
                <tr>
                    <th>Path</th>
                    <th>Extension</th>
                    <th>Error Details</th>
                </tr>
            </thead>
            <tbody>
                {% for file in data.file_processing.error_files %}
                <tr>
                    <td>{{ file.path }}</td>
                    <td>{{ file.extension }}</td>
                    <td>{{ file.details.error if file.details.error else "Unknown error" }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p>No errors were encountered during file processing.</p>
        {% endif %}
    </div>
    
    <div class="section">
        <h2>Code Units</h2>
        
        <div class="chart-container">
            <canvas id="codeUnitTypesChart"></canvas>
        </div>
        
        <div class="chart-container">
            <canvas id="languageDistributionChart"></canvas>
        </div>
        
        <h3>Code Unit Types</h3>
        <table>
            <thead>
                <tr>
                    <th>Type</th>
                    <th>Count</th>
                    <th>Total Size (chars)</th>
                    <th>Average Size (chars)</th>
                </tr>
            </thead>
            <tbody>
                {% for type, info in data.code_units.by_type.items() %}
                <tr>
                    <td>{{ type }}</td>
                    <td>{{ info.count }}</td>
                    <td>{{ info.size }}</td>
                    <td>{{ "%.2f"|format(info.size / info.count) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <h3>Largest Code Units</h3>
        {% if data.code_units.largest_units %}
        <table>
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Path</th>
                    <th>Size (chars)</th>
                </tr>
            </thead>
            <tbody>
                {% for unit in data.code_units.largest_units %}
                <tr>
                    <td>{{ unit.name }}</td>
                    <td>{{ unit.type }}</td>
                    <td>{{ unit.path }}</td>
                    <td>{{ unit.size }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p>No code units were processed.</p>
        {% endif %}
    </div>
    
    <div class="section">
        <h2>Parser Statistics</h2>
        
        {% if data.parser_stats %}
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Total files found</td>
                    <td>{{ data.parser_stats.total_files }}</td>
                </tr>
                <tr>
                    <td>Files parsed</td>
                    <td>{{ data.parser_stats.parsed_files }}</td>
                </tr>
                <tr>
                    <td>Files skipped (extension)</td>
                    <td>{{ data.parser_stats.skipped_files_extension }}</td>
                </tr>
                <tr>
                    <td>Files skipped (blacklisted)</td>
                    <td>{{ data.parser_stats.skipped_files_blacklisted }}</td>
                </tr>
                <tr>
                    <td>Files skipped (gitignore)</td>
                    <td>{{ data.parser_stats.skipped_files_gitignore }}</td>
                </tr>
                <tr>
                    <td>Folders skipped (gitignore)</td>
                    <td>{{ data.parser_stats.skipped_folders_gitignore }}</td>
                </tr>
                <tr>
                    <td>Parsing errors</td>
                    <td>{{ data.parser_stats.parsing_errors }}</td>
                </tr>
                <tr>
                    <td>Parsing time</td>
                    <td>{{ "%.2f"|format(data.parser_stats.parsing_time) }} seconds</td>
                </tr>
            </tbody>
        </table>
        
        {% if data.parser_stats.parsing_errors_details %}
        <h3>Parsing Error Types</h3>
        <table>
            <thead>
                <tr>
                    <th>Error Type</th>
                    <th>Count</th>
                </tr>
            </thead>
            <tbody>
                {% for error_type, count in data.parser_stats.parsing_errors_details.items() %}
                <tr>
                    <td>{{ error_type }}</td>
                    <td>{{ count }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
        
        {% else %}
        <p>No parser statistics available.</p>
        {% endif %}
    </div>
    
    <script>
        // File types chart
        const fileExtensions = {};
        {% for file in data.file_processing.processed_files %}
        const ext = "{{ file.extension }}" || "unknown";
        fileExtensions[ext] = (fileExtensions[ext] || 0) + 1;
        {% endfor %}
        
        const fileExtLabels = Object.keys(fileExtensions);
        const fileExtData = Object.values(fileExtensions);
        
        new Chart(document.getElementById('fileTypesChart'), {
            type: 'bar',
            data: {
                labels: fileExtLabels,
                datasets: [{
                    label: 'Files by Extension',
                    data: fileExtData,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        
        // Code unit types chart
        const unitTypeLabels = [{% for type in data.code_units.by_type %}"{{ type }}",{% endfor %}];
        const unitTypeCounts = [{% for type, info in data.code_units.by_type.items() %}{{ info.count }},{% endfor %}];
        
        new Chart(document.getElementById('codeUnitTypesChart'), {
            type: 'pie',
            data: {
                labels: unitTypeLabels,
                datasets: [{
                    label: 'Code Units by Type',
                    data: unitTypeCounts,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.5)',
                        'rgba(54, 162, 235, 0.5)',
                        'rgba(255, 206, 86, 0.5)',
                        'rgba(75, 192, 192, 0.5)',
                        'rgba(153, 102, 255, 0.5)',
                        'rgba(255, 159, 64, 0.5)'
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)',
                        'rgba(255, 159, 64, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
        
        // Language distribution chart
        const langLabels = [{% for lang in data.code_units.by_language %}"{{ lang }}",{% endfor %}];
        const langCounts = [{% for lang, info in data.code_units.by_language.items() %}{{ info.count }},{% endfor %}];
        
        new Chart(document.getElementById('languageDistributionChart'), {
            type: 'doughnut',
            data: {
                labels: langLabels,
                datasets: [{
                    label: 'Code Units by Language',
                    data: langCounts,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.5)',
                        'rgba(54, 162, 235, 0.5)',
                        'rgba(255, 206, 86, 0.5)',
                        'rgba(75, 192, 192, 0.5)',
                        'rgba(153, 102, 255, 0.5)',
                        'rgba(255, 159, 64, 0.5)',
                        'rgba(201, 203, 207, 0.5)',
                        'rgba(255, 159, 64, 0.5)',
                        'rgba(153, 102, 255, 0.5)',
                        'rgba(75, 192, 192, 0.5)'
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)',
                        'rgba(255, 159, 64, 1)',
                        'rgba(201, 203, 207, 1)',
                        'rgba(255, 159, 64, 1)',
                        'rgba(153, 102, 255, 1)',
                        'rgba(75, 192, 192, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    </script>
</body>
</html>
        """
        
        # Render the template with the data
        return Template(template).render(data=data)