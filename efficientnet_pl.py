from math import ceil

import pytorch_lightning as pl
from pytorch_lightning import loggers
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

# phi_values = {
#     # tuple of: (phi_value, resolution, drop_rate)
#     "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
#     "b1": (0.5, 240, 0.2),
#     "b2": (1, 260, 0.3),
#     "b3": (2, 300, 0.3),
#     "b4": (3, 380, 0.4),
#     "b5": (4, 456, 0.4),
#     "b6": (5, 528, 0.5),
#     "b7": (6, 600, 0.5),
# }

phi_values = {
    # tuple of: (width_coefficient, depth_coefficient, resolution, drop_rate)
    "b0": (1, 1, 32, 0.2),
    "b1": (1, 1.1, int(32 * (240 / 224)), 0.2),  # resolution = 34
    "b2": (1.1, 1.2, int(32 * (260 / 224)), 0.3),  # resolution = 37
    "b3": (1.2, 1.4, int(32 * (300 / 224)), 0.3),  # resolution = 42
    "b4": (1.4, 1.8, int(32 * (380 / 224)), 0.4),  # resolution = 54
    "b5": (1.6, 2.2, int(32 * (456 / 224)), 0.4),  # resolution = 65
    "b6": (1.8, 2.6, int(32 * (528 / 224)), 0.5),  # resolution = 75
    "b7": (2, 3.1, int(32 * (600 / 224)), 0.5),  # resolution = 85
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
        self.iteration = 0
        self.num_classes = cf.num_classes[dataset_name]

        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)

        last_channels = ceil(1280 * width_factor)

        _, res, drop_rate = phi_values[version]

        print(_, res, drop_rate)

        self.pool1 = nn.AdaptiveAvgPool2d((res, res))
        self.pool2 = nn.AdaptiveAvgPool2d(1)

        self.features = self.create_features(width_factor, depth_factor, last_channels)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(last_channels, self.num_classes),
        )

        self.loss = nn.CrossEntropyLoss()

    @staticmethod
    def calculate_factors_old(version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    @staticmethod
    def calculate_factors(version):
        width_factor, depth_factor, res, drop_rate = phi_values[version]
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
                        padding=kernel_size // 2,
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

    def training_step(self, batch, idx_batch, log=True):
        self.iteration += 1
        x, y = batch
        z = self(x)
        loss = self.loss(z, y)
        acc = train_accuracy(nn.functional.softmax(z, 1).argmax(1).cpu(), y.cpu())

        if log:
            self.logger.experiment.add_scalar("train_loss", loss, self.iteration)

            self.log(
                "train_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

            self.logger.experiment.add_scalar(
                "train_acc", acc, self.iteration,
            )
            self.log(
                "train_acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return {"loss": loss, "train_acc": acc}

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx, False)

        self.logger.experiment.add_scalar(
            "test_acc", results["train_acc"], self.iteration,
        )

        self.logger.experiment.add_scalar(
            "test_loss", results["loss"], self.iteration,
        )

        self.log(
            "test_loss",
            results["loss"],
            on_epoch=True,
            prog_bar=False,
            on_step=False,
            logger=True,
        )

        self.log(
            "test_acc",
            results["train_acc"],
            on_epoch=True,
            prog_bar=False,
            on_step=False,
            logger=True,
        )

        return {"test_acc": results["train_acc"], "loss": results["loss"]}

    def validation_epoch_end(self, test_step_outputs):
        avg_test_loss = torch.tensor([x["loss"] for x in test_step_outputs]).mean()
        avg_test_acc = torch.tensor([x["test_acc"] for x in test_step_outputs]).mean()

        self.logger.experiment.add_scalar(
            "avg_test_acc", avg_test_acc, self.iteration,
        )

        self.logger.experiment.add_scalar(
            "avg_test_loss", avg_test_loss, self.iteration
        )

        self.log(
            "avg_test_acc", avg_test_acc, on_epoch=True, prog_bar=True, on_step=False
        )


def train(
    dataset_name="cifar10",
    version="b6",
    batch_size=10,
    epochs=100,
    checkpoint=None,
    output_path=None,
):
    cifar_dm = CifarDataModule(batch_size=batch_size, dataset_name=dataset_name)

    model = EfficientNet(dataset_name=dataset_name, version=version)

    if output_path is None:
        output_path = f"lightning_logs/{dataset_name}/{version}"

    logger = loggers.TensorBoardLogger(output_path)

    trainer = pl.Trainer(
        progress_bar_refresh_rate=20,
        max_epochs=epochs,
        gpus=1,
        logger=logger,
        resume_from_checkpoint=checkpoint,
    )

    trainer.fit(model, cifar_dm)


if __name__ == "__main__":
    train()
