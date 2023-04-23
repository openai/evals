import os

# Get the directory path, old and new substrings, primary key positions, and target type from the user
dir_path = input("Enter the directory path: ")
old_substring = input("Enter the substring you want to replace: ")
new_substring = input("Enter the new substring to replace the old one: ")
new_pk_start = int(input("Enter the starting position of the primary key in the new name (use negative indices for counting from the end): "))
new_pk_end = int(input("Enter the ending position of the primary key in the new name (use negative indices for counting from the end): "))
target_type = input("Do you want to rename files or directories? (Enter 'files' or 'dirs'): ")

# Iterate through the items in the directory
for item in os.listdir(dir_path):
    item_path = os.path.join(dir_path, item)

    # Check if the item is a file or a directory based on the user's choice
    if (target_type == 'files' and os.path.isfile(item_path)) or (target_type == 'dirs' and os.path.isdir(item_path)):
        # Replace the old substring with the new substring in the item name
        new_item_name = item.replace(old_substring, new_substring)

        # Extract the primary key based on the given positions in the new item name
        primary_key = new_item_name[new_pk_start:new_pk_end+1]

        # Reinsert the primary key at the specified positions in the new item name
        new_item_name = new_item_name[:new_pk_start] + primary_key + new_item_name[new_pk_end+1:]

        # Get the full paths of the old and new item names
        old_item_path = os.path.join(dir_path, item)
        new_item_path = os.path.join(dir_path, new_item_name)

        # Rename the item
        os.rename(old_item_path, new_item_path)

print("Selected items have been renamed.")
