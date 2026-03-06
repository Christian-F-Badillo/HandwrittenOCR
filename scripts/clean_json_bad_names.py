import os
import json


def is_valid_entry(filename: str, label: str) -> bool:
    """Evalúa si una imagen y su etiqueta deben conservarse."""

    # 1. Regla universal: Si tiene el carácter corrupto 'Ã', se va.
    if "Ã" in filename or "Ã" in label:
        return False

    # 2. Regla para datos sintéticos: Evitar 'ñ' y 'ü' por problemas en Kaggle
    if "synthetic" in filename.lower():
        caracteres_prohibidos = ["ñ", "ü", "Ñ", "Ü"]

        # Si alguno de los prohibidos está en el nombre o en la etiqueta, se rechaza
        if any(c in filename for c in caracteres_prohibidos) or any(
            c in label for c in caracteres_prohibidos
        ):
            return False

    # Si pasa todas las pruebas, es válida
    return True


def clean_json_file(filepath: str) -> None:
    """
    Lee un archivo JSON, elimina entradas corruptas o sintéticas conflictivas
    y sobrescribe el archivo con la versión limpia.
    """
    print(f"Looking: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_count = len(data)

    # Combinamos todas las reglas en una sola línea usando nuestra función validadora
    clean_data = {k: v for k, v in data.items() if is_valid_entry(k, v)}

    clean_count = len(clean_data)

    if original_count != clean_count:
        removed = original_count - clean_count
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=4)
        print(f" Clean completed. Removed: {removed} images.")
        print(f" Healthy images: {clean_count}.\n")
    else:
        print(" Not removed entries. All healthy!\n")


if __name__ == "__main__":
    base_dir = "dataset_kaggle"

    splits = ["train_split.json", "val_split.json", "test_split.json"]

    print("Celaning bad chars files...\n" + "=" * 50)

    for split_file in splits:
        full_path = os.path.join(base_dir, split_file)

        if os.path.exists(full_path):
            clean_json_file(full_path)
        else:
            print(f"File not Found: {full_path}\n")

    print("=" * 50 + "\nFinished Process.")
