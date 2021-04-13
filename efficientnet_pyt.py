import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet as EfficientNetPyt
from pytorch_lightning import loggers
from sklearn.metrics import accuracy_score
from torch import optim
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

    def training_step(self, batch, idx_batch, log=True):
        self.iteration += 1
        x, y = batch
        z = self(x)
        loss = self.loss(z, y)
        print("y.shape = ", y.shape)
        print("z.shape = ", z.shape)
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
    version="efficientnet-b0",
    batch_size=10,
    epochs=100,
    checkpoint=None,
    output_path=None,
    **model_params,
):
    cifar_dm = CifarDataModule(batch_size=batch_size, dataset_name=dataset_name)

    model = EfficientNet(
        model_name=version,
        num_classes=cf.num_classes[dataset_name],
        image_size=32,
        **model_params,
    )

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


def test(model, dataset):
    preds = []
    targets = []
    losses = []
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    model.to(device)
    model.eval()

    for image, target in dataset:
        image = image.to(device)
        target_list = target.cpu().numpy().tolist()
        target = target.to(device)
        logit = model(image)
        loss = model.loss(logit, target)
        losses.append(loss.detach().cpu().numpy().tolist())
        pred = nn.functional.softmax(logit, 1).argmax(1).cpu().numpy().tolist()
        preds.extend(pred)
        targets.extend(target_list)

    loss = np.array(losses).mean()
    acc = accuracy_score(targets, preds)

    return loss, acc


def train_test(dataset_name, batch_size, model_name, checkpoint):

    num_classes = cf.num_classes[dataset_name]
    dataset = CifarDataModule(dataset_name=dataset_name, batch_size=batch_size)

    dataset.setup(None)

    model = EfficientNet.load_from_checkpoint(
        checkpoint, model_name=model_name, num_classes=num_classes
    )

    _, test_acc = test(model, dataset.test_dataloader())
    train_loss, train_acc = test(model, dataset.train_dataloader())

    return train_loss, train_acc, test_acc


if __name__ == "__main__":
    # train(batch_size=30)

    checkpoint = (
        "/home/thiago/projects/efficientnet_pl/lightning_logs/cifar10/"
        "efficientnet-b0/default/version_1/checkpoints/epoch=3-step=66.ckpt"
    )

    loss, train_acc, test_acc = train_test("cifar10", 30, "efficientnet-b0", checkpoint)

    print(f"Loss: {loss}")
    print(f"Test: {test_acc}")
    print(f"Train: {train_acc}")
