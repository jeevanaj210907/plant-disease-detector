from pathlib import Path

import tensorflow as tf

from src.disease_catalog import DEFAULT_LABELS
from src.model import build_model


def main():
    data_dir = Path("data")
    model_output = Path("model/plant_disease_model.keras")

    if not data_dir.exists():
        raise FileNotFoundError(
            "Dataset folder not found. Please organize your images under data/train and data/valid, "
            "with class folders such as healthy, rust, powdery_mildew, etc."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir / "train",
        image_size=(224, 224),
        batch_size=32,
        color_mode="rgb",
        validation_split=False,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir / "valid",
        image_size=(224, 224),
        batch_size=32,
        color_mode="rgb",
        validation_split=False,
        seed=42,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    if len(class_names) == 0:
        raise ValueError("No disease classes found in the training dataset.")

    model = build_model(num_classes=num_classes)
    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(model_output),
                save_best_only=True,
            ),
        ],
    )

    model.save(model_output)
    print(f"Training complete. Model saved to: {model_output}")


if __name__ == "__main__":
    main()
