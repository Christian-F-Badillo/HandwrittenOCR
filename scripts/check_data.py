import json
import os


def load_valid_chars(tokens_path: str) -> set:
    """Carga el archivo de tokens y devuelve un set con los caracteres válidos."""
    with open(tokens_path, "r", encoding="utf-8") as f:
        tokens = json.load(f)

    valid_chars = set(tokens.keys())

    if "<Blank>" in valid_chars:
        valid_chars.remove("<Blank>")

    return valid_chars


def check_json_vocab(json_path: str, valid_chars: set) -> None:
    """Escanea los valores de un JSON buscando caracteres no permitidos."""
    print(f"Loking  for..: {os.path.basename(json_path)}...")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_text = "".join(data.values()).lower()

    found_chars = set(all_text)

    invalid_chars = found_chars - valid_chars

    if not invalid_chars:
        print("Not bad chars found\n")
    else:
        print(f"Found bad chars {len(invalid_chars)}")
        print(f"{invalid_chars}")

        print("Bad words:")
        examples_shown = 0
        for filename, label in data.items():
            if any(bad_char in label.lower() for bad_char in invalid_chars):
                print(f"       - {filename}: '{label}'")
                examples_shown += 1
                if examples_shown >= 5:
                    break
        print()


if __name__ == "__main__":
    TOKENS_FILE = "include/tokens.json"
    BASE_DIR = "dataset_kaggle"

    splits = ["train_split.json", "val_split.json", "test_split.json"]

    try:
        valid_vocab = load_valid_chars(TOKENS_FILE)
        print(
            f"Vocabulario cargado: {len(valid_vocab)} caracteres permitidos.\n"
            + "=" * 50
        )

        for split in splits:
            json_target = os.path.join(BASE_DIR, split)
            if os.path.exists(json_target):
                check_json_vocab(json_target, valid_vocab)
            else:
                print(f"File not Found: {json_target}\n")

    except FileNotFoundError as e:
        print(f"Error: {e}")
