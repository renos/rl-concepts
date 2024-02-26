def change_pwd():
    import os

    # Get the current working directory
    current_dir = os.getcwd()

    # Get the parent directory
    parent_dir = os.path.dirname(current_dir)

    # Change the working directory to the parent directory
    os.chdir(parent_dir)
