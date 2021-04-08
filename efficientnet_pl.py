from math import ceil

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim

import config as cf
from data_module import CifarDataModule

base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


train_accuracy = pl.metrics.Accuracy()
valid_accuracy = pl.metrics.Accuracy(compute_on_step=False)


class CNNBlock(pl.LightningModule):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, groups=1
    ):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()  # SiLU <-> Swish

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


class SqueezeExcitation(pl.LightningModule):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        expand_ratio,
        reduction=4,  # squeeze excitation
        survival_prob=0.8,  # for stochastic depth
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1,
            )

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = (
            torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        )
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(pl.LightningModule):
    def __init__(self, version, dataset_name):
        super(EfficientNet, self).__init__()
        num_classes = cf.num_classes[dataset_name]
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)

        last_channels = ceil(1280 * width_factor)

        phi, res, drop_rate = phi_values[version]

        self.pool1 = nn.AdaptiveAvgPool2d((res, res))
        self.pool2 = nn.AdaptiveAvgPool2d(1)

        self.features = self.create_features(width_factor, depth_factor, last_channels)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(last_channels, num_classes),
        )

        self.loss = nn.CrossEntropyLoss()

    @staticmethod
    def calculate_factors(version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    @staticmethod
    def create_features(width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                    )
                )
                in_channels = out_channels

        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)

    def forward(self, x):
        x = self.pool1(x)
        features = self.features(x)
        x = self.pool2(features)
        return self.classifier(x.view(x.shape[0], -1))

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, batch, idx_batch):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        acc = train_accuracy(logits.argmax(1), y)
        pbar = {"train_acc": acc}
        return {"loss": loss, "progress_bar": pbar}

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        results["progress_bar"]["val_acc"] = results["progress_bar"]["train_acc"]
        del results["progress_bar"]["train_acc"]
        return results

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x["loss"] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor(
            [x["progress_bar"]["val_acc"] for x in val_step_outputs]
        ).mean()
        pbar = {"avg_val_acc": avg_val_acc}
        return {"val_loss": avg_val_loss, "progress_bar": pbar}


def test():
    version = "b0"
    dataset_name = "cifar10"
    imagenet_dm = CifarDataModule(batch_size=20, dataset_name=dataset_name)
    model = EfficientNet(version=version, dataset_name=dataset_name)
    trainer = pl.Trainer(progress_bar_refresh_rate=20, max_epochs=1)
    trainer.fit(model, imagenet_dm)


if __name__ == "__main__":
    test()
