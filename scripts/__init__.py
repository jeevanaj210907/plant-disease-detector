import subprocess
import sys
from pathlib import Path


def main():
    target_dir = Path("data/raw")
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading the PlantVillage dataset via KaggleHub...")
    print("Make sure you have your Kaggle credentials configured before running this.")

    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Install it with: pip install kagglehub")
        sys.exit(1)

    try:
        path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
        print(f"Dataset downloaded to: {path}")
    except Exception as exc:
        print(f"Download failed: {exc}")
        print("If the dataset name has changed, update this script or use an alternate public dataset URL.")
        sys.exit(1)


if __name__ == "__main__":
    main()
