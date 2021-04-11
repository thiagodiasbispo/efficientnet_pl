from glob import glob


def recover_last_check_point(path):
    path_versions = glob(f"{path}/*")
    last_path = max(path_versions, key=lambda p: int(p.split("_")[-1]))

    check_points = glob(f"{last_path}/checkpoints/*")
    return check_points[0] if check_points else None


if __name__ == "__main__":
    ckpt = recover_last_check_point("lightning_logs/cifar10/efficientnet-b0/default")
    print(ckpt)
