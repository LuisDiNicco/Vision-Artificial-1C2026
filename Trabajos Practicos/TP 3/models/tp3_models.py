import torch.nn as nn

from utils.tp3_config import NUM_CLASSES


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Bloque simple: convolucion + activacion + reduccion de tamano.
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class ModeloBase(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # Extrae caracteristicas con 3 bloques y luego clasifica.
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ModeloOptimizado(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # Version mas profunda para capturar patrones mas complejos.
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
