import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv2d, MaxPool2d, Module, ReLU


class CNN(Module):
    def __init__(self) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            # First Conv Layer -> reduce input by a half
            Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Second Conv Layer -> reduce input by a half
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Third Conv Layer -> Only extract features
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            # Fourth Conv Layer -> Asymmetric Pooling
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # Fifth Conv Layer -> More features extraction
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            # Sixth Conv Layer -> Last Asymmetric Pooling
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # Seventh Conv Layer -> To 1 pixel feat representation
            Conv2d(
                in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=0
            ),
            ReLU(inplace=True),
        )

    def forward(self, x):
        return self.cnn(x)


class RNN_Encoder(nn.Module):
    def __init__(self, input_size=512, hidden_size=256):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )

    def forward(self, x):
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, *_ = self.rnn(x)
        return x


class HandwrittenCRNN(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256):
        super().__init__()

        self.vision = CNN()
        self.seq = RNN_Encoder(input_size=512, hidden_size=hidden_size)
        self.clf = nn.Linear(in_features=hidden_size * 2, out_features=num_classes)

    def forward(self, x):
        x = self.vision(x)
        x = self.seq(x)
        x = self.clf(x)

        return x

    def inference(self, x):
        logits = self.forward(x)

        probs = F.softmax(logits, dim=-1)

        return probs
