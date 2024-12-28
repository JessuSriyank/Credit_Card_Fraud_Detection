import os

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model(model, path):
    model.save(path)
    print(f"Model saved at {path}")
