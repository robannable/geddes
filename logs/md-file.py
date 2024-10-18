import json

# Load the JSON data
with open('response_log.json', 'r') as file:
    data = json.load(file)

# Create Markdown content
md_content = ""

for entry in data:
    md_content += f"# Name\n{entry['name']}\n\n"
    md_content += f"## Date\n{entry['date']}\n\n"
    md_content += f"## Time\n{entry['time']}\n\n"
    md_content += f"## Question\n{entry['question']}\n\n"
    md_content += f"## Response\n{entry['response']}\n\n"
    md_content += "---\n\n"  # Add a horizontal line between entries

# Write the Markdown content to a file
with open('extracted_data.md', 'w') as file:
    file.write(md_content)

print("Markdown file 'extracted_data.md' has been created successfully.")
