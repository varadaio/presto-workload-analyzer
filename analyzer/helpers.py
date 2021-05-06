import os


def remove_local_file(file_path):
    # Deletes the local file if it exists otherwise does nothing
    if os.path.exists(file_path):
        os.remove(file_path)
