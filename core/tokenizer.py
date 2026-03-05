import enum
import os
import json
import re
import numpy as np
from typing import List, Optional, Set, Tuple


def _str_to_charlist(text: str) -> List[str]:
    return [char for char in text]


def _generate_bag_of_chars(src: Tuple[str] | List[str]) -> Set[str]:
    return {i for i in src}


def _tokenize_data_list(data: List[str]):
    chars = [_generate_bag_of_chars(_str_to_charlist(word)) for word in data]

    out_bag = set([])

    for bag in chars:
        out_bag.update(bag)

    return out_bag


def save_tokens_to_file(
    data: List[str], file_name: str = "tokens.json", out_dir: Optional[str] = None
) -> None:
    out_file = file_name

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, file_name)
        else:
            out_file = os.path.join(out_dir, file_name)

    tokens = _tokenize_data_list(data)

    tokens_dict = {tok: (num + 1) for num, tok in enumerate(tokens)}
    tokens_dict["<Blank>"] = 0

    with open(out_file, "w", encoding="utf-8") as file:
        json.dump(
            tokens_dict,
            file,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


class Tokenizer:
    def __init__(self, src: str) -> None:
        if not os.path.exists(src):
            raise FileNotFoundError(f"File {src} no found")

        self.file = src

        with open(src, "r", encoding="utf-8") as file:
            self._encode = json.load(file)

        self._decode = {v: k for k, v in self._encode.items()}

        self._ntokens = max(list(self._encode.values()))

    @property
    def ntokens(self):
        return self._ntokens + 1  # Sum the empty char (<Blank>)

    def encoding(self, text: str | List[str]) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(text, list):
            input_text = _str_to_charlist(str(text).lower())
            out_array = np.array([self._encode[char] for char in input_text])
            lengths = [len(input_text)]
            return out_array, np.array(lengths)

        else:
            input_text = [_str_to_charlist(word.lower()) for word in text]
            lengths = np.array([len(word) for word in input_text], dtype=np.int64)

            max_word_size = lengths.max()
            out_array = np.zeros(shape=(len(input_text), max_word_size), dtype=np.int64)

            for sample, word in enumerate(input_text):
                for idx, char in enumerate(word):
                    out_array[sample, idx] = self._encode[char]

            return out_array, lengths

    def decoding(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.ndim == 2:
            nrows = matrix.shape[0]
            result = np.empty(shape=(nrows,), dtype=object)
            for row in range(nrows):
                chars = [self._decode[code] for code in matrix[row]]
                result[row] = "".join(chars).replace("<Blank>", "")

            return result
        else:
            chars = [self._decode[code] for code in matrix]
            result = np.array(["".join(chars).replace("<Blank>", "")], dtype=object)
            return result


if __name__ == "__main__":
    tokens_file = os.path.join("include", "tokens.json")
    tokenizer = Tokenizer(tokens_file)

    data = ["hola", "adíos", "xd", "HOLA"]
    cadena = "Mucho"
    result, lenghts = tokenizer.encoding(data)
    result_str, lenghts_str = tokenizer.encoding(cadena)

    result_reverse = tokenizer.decoding(result)
    result_str_reverse = tokenizer.decoding(result_str)
    print("Encoding:\n", result)

    print("Decoding: \n", result_reverse)

    print("Encoding:\n", result_str)

    print("Decoding: \n", result_str_reverse)
