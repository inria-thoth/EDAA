import os
import subprocess
import zipfile

DATA_PATH = "./data.zip"
TARGET_DIR = "./data/"
URL = "http://pascal.inrialpes.fr/data2/azouaoui/data.zip"


def main():

    if not os.path.exists(DATA_PATH):
        cmd_dl = f"wget --no-check-certificate {URL!r} -O {DATA_PATH!r}"
        subprocess.check_call(cmd_dl, shell=True)
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
