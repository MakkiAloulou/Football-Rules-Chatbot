import os # Provides a portable way of using operating system dependent functionality like reading or writing to the file system.
import shutil # Utility functions for copying and archiving files and directory trees.

def ensure_file_exists(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"❌ File not found: {path}")

def ensure_directory_exists(path: str):
    if not os.path.isdir(path):
        print(f"Creating extraction directory: {path}")
        os.makedirs(path, exist_ok=True)
    else:
        print(f"Using existing extraction directory: {path}")