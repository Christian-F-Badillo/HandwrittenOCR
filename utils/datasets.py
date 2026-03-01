from dataclasses import dataclass, field
import os
import json
from typing import List, Dict


@dataclass
class Datadict:
    data: Dict[str, str] = field(default_factory=dict)
    index: List[str] = field(default_factory=list)


def _json_to_dict(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f_in:
        return json.load(f_in)


def load_data(path: str) -> Datadict:
    safe_path = os.path.join(".", path)

    if not os.path.exists(safe_path):
        raise FileNotFoundError(f"Not such file or directory found: {path}")

    temp_data = _json_to_dict(safe_path)
    temp_idx = list(temp_data.keys())

    return Datadict(data=temp_data, index=temp_idx)
