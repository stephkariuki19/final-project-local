# Open the file for reading
with open('./dataChat/medscapePneumonia.txt', 'r') as file:
    # Read the contents of the file
    content = file.read()

# Remove whitespace (spaces, tabs, newlines) from the content
content = ''.join(content.split())

# Open the file for writing and overwrite its contents
with open('./dataChat/medscapePneumonia.txt', 'w') as file:
    # Write the modified content back to the file
    file.write(content)
