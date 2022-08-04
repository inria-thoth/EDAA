import os
import zipfile

DATA_PATH = "./data.zip"
TARGET_DIR = "./data/"


def main():
    assert os.path.exists(DATA_PATH)
    if os.path.exists(TARGET_DIR):
        print(f"{TARGET_DIR} already exists. Aborting...")
        return
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"{TARGET_DIR} has been created. Extracting data...")
    with zipfile.ZipFile(DATA_PATH, "r") as zip_ref:
        zip_ref.extractall(TARGET_DIR)
    print("Extraction complete.")


if __name__ == "__main__":
    main()
