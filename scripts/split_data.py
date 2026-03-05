import os
import json
from sklearn.model_selection import train_test_split
from utils.datasets import load_data  # Tu función actual


def save_split(data_dict: dict, out_path: str):
    """Guarda un diccionario en formato JSON limpio y seguro (UTF-8)."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    data_dict_path = os.path.join("dataset_kaggle", "images", "0annotation.json")
    data_dict_aug_path = os.path.join(
        "dataset_kaggle", "images", "synthetic_annotation.json"
    )
    out_dir = "dataset_kaggle"  # Guardamos los splits en la misma carpeta raíz

    print("Loading datasets")
    datadict_real = load_data(data_dict_path)
    datadict_aug = load_data(data_dict_aug_path)

    real_items = list(datadict_real.data.items())

    train_real, temp_split = train_test_split(
        real_items, test_size=0.20, random_state=42
    )

    val_real, test_real = train_test_split(temp_split, test_size=0.50, random_state=42)

    dict_train = dict(train_real)
    dict_val = dict(val_real)
    dict_test = dict(test_real)

    print(f"Train dataset size: {len(real_items)}")
    print(f" -> Train real: {len(dict_train)}")
    print(f" -> Val real:   {len(dict_val)}")
    print(f" -> Test real:  {len(dict_test)}")

    print(f"\nAugmenting {len(datadict_aug.data)} images of synthetic data...")
    dict_train.update(datadict_aug.data)

    print(f" -> Train Final size: {len(dict_train)}")

    save_split(dict_train, os.path.join(out_dir, "train_split.json"))
    save_split(dict_val, os.path.join(out_dir, "val_split.json"))
    save_split(dict_test, os.path.join(out_dir, "test_split.json"))

    print("\nSplit generated succesfully")
