from pathlib import Path
from shutil import copy2
from random import shuffle


def prepare_dataset(dataset_root: str, train_dir: str = "data/train", valid_dir: str = "data/valid", valid_ratio: float = 0.2):
    dataset_path = Path(dataset_root)
    train_path = Path(train_dir)
    valid_path = Path(valid_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    for target in (train_path, valid_path):
        if target.exists():
            for child in target.iterdir():
                if child.is_dir():
                    for nested in child.iterdir():
                        if nested.is_file():
                            nested.unlink()
                    child.rmdir()
        target.mkdir(parents=True, exist_ok=True)

    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    class_dirs = [p for p in dataset_path.iterdir() if p.is_dir()]

    if not class_dirs:
        raise ValueError(
            f"No class folders found under {dataset_path}. "
            "Please point this script to the folder that contains disease labels, such as 'healthy', 'rust', etc."
        )

    for class_dir in class_dirs:
        files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_exts]
        if not files:
            continue

        shuffle(files)
        split_index = max(1, int(len(files) * (1 - valid_ratio)))
        train_files = files[:split_index]
        valid_files = files[split_index:]

        for label, file_list in [("train", train_files), ("valid", valid_files)]:
            dest_dir = (train_path if label == "train" else valid_path) / class_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for image_path in file_list:
                target_path = dest_dir / image_path.name
                copy2(image_path, target_path)

    print(f"Prepared dataset from {dataset_path}")
    print(f"Train set: {train_path}")
    print(f"Validation set: {valid_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare a plant disease dataset for TensorFlow training.")
    parser.add_argument("--dataset_root", required=True, help="Directory containing labeled class folders")
    parser.add_argument("--train_dir", default="data/train")
    parser.add_argument("--valid_dir", default="data/valid")
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    args = parser.parse_args()

    prepare_dataset(
        dataset_root=args.dataset_root,
        train_dir=args.train_dir,
        valid_dir=args.valid_dir,
        valid_ratio=args.valid_ratio,
    )
