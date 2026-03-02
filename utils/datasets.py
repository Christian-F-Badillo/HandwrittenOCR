import os
import json
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image


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


class HandwrittenData(Dataset):
    def __init__(
        self, datadict: Datadict, data_path: str, transform: Optional[Callable] = None
    ) -> None:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Not file o dir found: {data_path}")

        super().__init__()
        self.datadict = datadict
        self.path = data_path
        self.transform = transform
        self.to_tensor = ToTensor()

    def __len__(self) -> int:
        return len(self.datadict.index)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        file_name = self.datadict.index[index]
        label = self.datadict.data.get(file_name, "")

        full_path = os.path.join(self.path, file_name)
        img = Image.open(full_path)

        if self.transform is not None:
            img = self.transform(img)

        if not isinstance(img, torch.Tensor):
            img = self.to_tensor(img)

        return img, label
