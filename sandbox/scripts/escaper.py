import json

# Read the markdown text from a file
with open('../docs/notepad.md', 'r') as f:
    markdown_text = f.read()

# Escape the markdown text
escaped_markdown = json.dumps(markdown_text)

# Remove all occurrences of multiple spaces
clean_text = escaped_markdown.replace('  ', ' ')

# Write the JSON object to a file
with open('../docs/notepad.txt', 'w') as f:
    f.write(clean_text)
