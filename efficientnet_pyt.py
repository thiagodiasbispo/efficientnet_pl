from math import ceil

import pytorch_lightning as pl
from pytorch_lightning import loggers
import torch
import torch.nn as nn
from torch import optim
from efficientnet_pytorch import EfficientNet as EfficientNetPyt
from torch.nn import CrossEntropyLoss

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
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, int(32 * (240 / 224)), 0.2),
    "b2": (1, int(32 * (260 / 224)), 0.3),
    "b3": (2, int(32 * (300 / 224)), 0.3),
    "b4": (3, int(32 * (380 / 224)), 0.4),
    "b5": (4, int(32 * (456 / 224)), 0.4),
    "b6": (5, int(32 * (528 / 224)), 0.5),
    "b7": (6, int(32 * (600 / 224)), 0.5),
}


train_accuracy = pl.metrics.Accuracy()
valid_accuracy = pl.metrics.Accuracy(compute_on_step=False)


class EfficientNet(pl.LightningModule):
    def __init__(self, model_name, num_classes, **override_params):
        super(EfficientNet, self).__init__()
        self.num_classes = num_classes
        self.iteration = 0
        self.model = EfficientNetPyt.from_name(
            model_name, num_classes=num_classes, **override_params
        )
        self.loss = CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-2)
        return optimizer

    def loss_function(self, output, target):
        one_hot = nn.functional.one_hot(target.long(), self.num_classes).to(self.device)
        loss = nn.functional.binary_cross_entropy_with_logits(output, one_hot.float())

        return loss

    def training_step(self, batch, idx_batch):
        self.iteration += 1
        x, y = batch
        z = self(x)
        loss = self.loss(z, y)  # self.loss_function(z, y)
        acc = train_accuracy(nn.functional.softmax(z, 1).argmax(1).cpu(), y.cpu())

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
            "train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # pbar = {"train_acc": acc}
        # return {"loss": loss, "progress_bar": pbar}

        return {"loss": loss, "train_acc": acc}

    # def validation_step(self, batch, batch_idx):
    #     results = self.training_step(batch, batch_idx)
    #     results["progress_bar"]["val_acc"] = results["progress_bar"]["train_acc"]
    #     del results["progress_bar"]["train_acc"]
    #     return results

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)

        # results["progress_bar"]["test_acc"] = results["progress_bar"]["train_acc"]
        # results["progress_bar"]["test_loss"] = results["loss"]
        # del results["progress_bar"]["train_acc"]

        # return results

        self.logger.experiment.add_scalar(
            "test_acc", results["train_acc"], self.iteration,
        )

        self.logger.experiment.add_scalar(
            "test_loss", results["loss"], self.iteration,
        )

        self.log(
            "test_loss",
            results["loss"],
            on_epoch=False,
            prog_bar=False,
            on_step=True,
            logger=True,
        )

        self.log(
            "test_acc",
            results["train_acc"],
            on_epoch=False,
            prog_bar=False,
            on_step=True,
            logger=True,
        )

        return {"test_acc": results["train_acc"], "loss": results["loss"]}

    # def validation_epoch_end(self, test_step_outputs):
    #     avg_val_loss = torch.tensor([x["loss"] for x in test_step_outputs]).mean()
    #     avg_val_acc = torch.tensor(
    #         [x["progress_bar"]["val_acc"] for x in test_step_outputs]
    #     ).mean()
    #     pbar = {"avg_val_acc": avg_val_acc}
    #     return {"val_loss": avg_val_loss, "progress_bar": pbar}

    def validation_epoch_end(self, test_step_outputs):
        # avg_val_loss = torch.tensor([x["loss"] for x in test_step_outputs]).mean()
        # avg_val_acc = torch.tensor([x["test_acc"] for x in test_step_outputs]).mean()
        # pbar = {"avg_test_acc": avg_val_acc, "avg_val_loss": avg_val_loss}
        # return {"test_loss": avg_val_loss, "progress_bar": pbar}

        avg_test_loss = torch.tensor([x["loss"] for x in test_step_outputs]).mean()
        avg_test_acc = torch.tensor([x["test_acc"] for x in test_step_outputs]).mean()

        self.logger.experiment.add_scalar(
            "avg_test_acc", avg_test_acc, self.iteration,
        )

        self.logger.experiment.add_scalar(
            "avg_test_loss", avg_test_loss, self.iteration
        )

        # self.log(
        #     "avg_test_loss", avg_test_loss, on_epoch=True, prog_bar=True, on_step=False
        # )

        self.log(
            "avg_test_acc", avg_test_acc, on_epoch=True, prog_bar=True, on_step=False
        )

        # pbar = {"avg_test_acc": avg_test_acc}
        # return {"test_loss": avg_test_loss, "progress_bar": pbar}


def train(
    dataset_name="cifar10",
    version="efficientnet-b0",
    batch_size=10,
    epochs=100,
    checkpoint=None,
):
    cifar_dm = CifarDataModule(batch_size=batch_size, dataset_name=dataset_name)

    model = EfficientNet(
        model_name=version, num_classes=cf.num_classes[dataset_name], image_size=32
    )
    logger = loggers.TensorBoardLogger(f"lightning_logs/{dataset_name}/{version}")

    trainer = pl.Trainer(
        progress_bar_refresh_rate=20,
        max_epochs=epochs,
        gpus=1,
        logger=logger,
        resume_from_checkpoint=checkpoint,
    )

    trainer.fit(model, cifar_dm)


if __name__ == "__main__":
    train(batch_size=30)
