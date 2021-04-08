import math
import os

RANDOM_STATE = 2021

start_epoch = 1
num_epochs = 200
optim_type = "SGD"

is_colab = "COLAB_GPU" in os.environ

if is_colab:
    batch_size = 256
else:
    batch_size = 20

mean = {
    "cifar10": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.5071, 0.4867, 0.4408),
}

std = {
    "cifar10": (0.2023, 0.1994, 0.2010),
    "cifar100": (0.2675, 0.2565, 0.2761),
}

num_classes = {"cifar10": 10, "cifar100": 100}

# Only for cifar-10
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def learning_rate(init, epoch, steps=(25, 40, 45)):
    optim_factor = 0
    for i in range(len(steps), 0, -1):
        if epoch >= steps[i - 1]:
            optim_factor = i
            break
    return init * math.pow(0.2, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


if __name__ == "__main__":
    for i in range(1, 51):
        print("Epoch: ", i, learning_rate(0.1, i))
