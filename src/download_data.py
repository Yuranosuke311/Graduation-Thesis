import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "aryan208/cybersecurity-threat-detection-logs"
DATA_DIR = "data"

def download():
    os.makedirs(DATA_DIR, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        DATASET,
        path=DATA_DIR,
        unzip=True
    )

    print("Dataset downloaded and extracted.")

if __name__ == "__main__":
    download()
