import torch
from torchvision import transforms
from typing import Union
from pathlib import Path
from PIL import Image
from core.crnn import HandwrittenCRNN
from core.tokenizer import Tokenizer


class CRNNInferencer:
    def __init__(
        self,
        model_path: str,
        tokenizer: Tokenizer,
        num_classes: int,
        device: str = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Inference in device: {self.device}")
        self.tokenizer = tokenizer

        self.idx2char = self.tokenizer._decode
        self.blank_idx = self.tokenizer._decode.get("<Blank>", 0)

        # Instance the model
        self.model = HandwrittenCRNN(num_classes=num_classes).to(self.device)

        # Load Model
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

        self.model.eval()

        # self.model = torch.jit.script(self.model)

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((32, 128)),
                transforms.ToTensor(),
            ]
        )

    def ctc_decode(self, indices: list) -> str:
        decoded_str = []
        prev_idx = -1  # Para rastrear el carácter anterior

        for idx in indices:
            # Regla 1: Ignorar si es el mismo carácter consecutivo sin un blank en medio
            if idx == prev_idx:
                continue

            # Regla 2: Ignorar el token <Blank>, pero agregamos las letras reales
            if idx != self.blank_idx:
                decoded_str.append(self.idx2char[idx])

            # ¡EL TRUCO MAESTRO!: Si llega un <Blank>, prev_idx se vuelve <Blank>.
            # Esto rompe la regla 1 en el siguiente ciclo, permitiendo letras repetidas.
            prev_idx = idx

        return "".join(decoded_str)

    @torch.inference_mode()
    def predict(self, input_data: Union[str, Path, torch.Tensor]) -> str:
        """
        Predice el texto de una imagen.
        Acepta una ruta (str/Path) o un tensor ya preprocesado.
        """

        # 1. Lógica de carga/validación
        if isinstance(input_data, (str, Path)):
            img = Image.open(input_data).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        elif isinstance(input_data, torch.Tensor):
            # Si ya es un tensor, aseguramos que tenga dimensión de batch [1, C, H, W]
            img_tensor = input_data if input_data.ndim == 4 else input_data.unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
        else:
            raise ValueError("El input debe ser una ruta (str/Path) o un torch.Tensor")

        # 2. Forward pass
        logits = self.model(img_tensor)  # Forma esperada: [1, seq_len, num_classes]

        # 3. Decoding
        pred_indices = torch.argmax(logits, dim=2)
        pred_list = pred_indices.squeeze(0).cpu().tolist()

        return self.ctc_decode(pred_list)
