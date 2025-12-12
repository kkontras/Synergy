import os
import json
import random
import shutil
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ===========================================================
#          PREPARE CUB-200 USING OFFICIAL SPLIT
# ===========================================================

def prepare_cub200(data_root: str):
    """
    Prepares the CUB-200-2011 dataset.
    data_root must contain:
        CUB_200_2011/
            images/
            classes.txt
            images.txt
            image_class_labels.txt
            train_test_split.txt
            ...
    Creates:
        images/              (copied)
        train.txt
        val.txt              (10% of training)
        test.txt             (official test)
        labels.json
        captions.txt
        READY.flag
    """

    marker = os.path.join(data_root, "READY.flag")
    if os.path.exists(marker):
        print("[CUB-200] Already prepared.")
        return

    cub_root = data_root

    # -------------------------------------------------------
    # LOAD METADATA
    # -------------------------------------------------------
    # class_id → class name
    classes = {}
    with open(os.path.join(cub_root, "classes.txt")) as f:
        for line in f:
            cid, cname = line.strip().split()
            classes[int(cid)] = cname

    # image_id → file name
    id2fname = {}
    with open(os.path.join(cub_root, "images.txt")) as f:
        for line in f:
            img_id, fname = line.strip().split()
            id2fname[int(img_id)] = fname

    # image_id → class (0-based)
    img2class = {}
    with open(os.path.join(cub_root, "image_class_labels.txt")) as f:
        for line in f:
            img_id, cid = line.strip().split()
            img2class[int(img_id)] = int(cid) - 1

    # OFFICIAL: image_id → is_training_image (1 or 0)
    split_info = {}
    with open(os.path.join(cub_root, "train_test_split.txt")) as f:
        for line in f:
            img_id, flag = line.strip().split()
            split_info[int(img_id)] = int(flag)

    # -------------------------------------------------------
    # BUILD LABEL DICT + official train/test lists
    # -------------------------------------------------------
    labels = {}
    train_files = []
    test_files = []

    for img_id, fname in id2fname.items():
        cid = img2class[img_id]
        labels[fname] = cid

        if split_info[img_id] == 1:
            train_files.append(fname)
        else:
            test_files.append(fname)

    # create validation split from training (10%)
    random.shuffle(train_files)
    n_val = int(0.1 * len(train_files))

    val_files = train_files[:n_val]
    train_files = train_files[n_val:]

    # -------------------------------------------------------
    # WRITE SPLIT FILES
    # -------------------------------------------------------
    def write_list(lst, path):
        with open(path, "w") as f:
            for x in lst:
                f.write(x + "\n")

    write_list(train_files, os.path.join(data_root, "train.txt"))
    write_list(val_files,   os.path.join(data_root, "val.txt"))
    write_list(test_files,  os.path.join(data_root, "test.txt"))

    # -------------------------------------------------------
    # SAVE LABELS
    # -------------------------------------------------------
    with open(os.path.join(data_root, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)

    # -------------------------------------------------------
    # CAPTIONS = class names
    # -------------------------------------------------------
    print("[CUB-200] Writing captions.txt")
    with open(os.path.join(data_root, "captions.txt"), "w") as f:
        for fname, cid in labels.items():
            cname = classes[cid + 1]
            f.write(f"{fname}#0\t{cname}\n")

    open(marker, "w").write("OK")
    print("[CUB-200] Finished preparing dataset.")

class CUB200_Dataset(Dataset):
    def __init__(self, config, mode="train"):
        self.root = config.dataset.data_roots
        self.mode = mode.lower()

        prepare_cub200(self.root)

        self.img_dir = os.path.join(self.root, "images")
        labels_json = os.path.join(self.root, "labels.json")
        captions_file = os.path.join(self.root, "captions.txt")

        split_files = {
            "train": os.path.join(self.root, "train.txt"),
            "val":   os.path.join(self.root, "val.txt"),
            "test":  os.path.join(self.root, "test.txt"),
        }

        with open(split_files[self.mode], "r") as f:
            self.filenames = [line.strip() for line in f]

        self.labels = json.load(open(labels_json))

        self.caption_dict = {}
        with open(captions_file, "r") as f:
            for line in f:
                key, caption = line.strip().split("\t")
                fname = key.split("#")[0]
                self.caption_dict.setdefault(fname, []).append(caption)

        self.train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.eval_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.filenames)

    def _load_image(self, fname):
        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert("RGB")
        return self.train_tf(img) if self.mode == "train" else self.eval_tf(img)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        img = self._load_image(fname)
        caption = random.choice(self.caption_dict[fname])
        cid = self.labels[fname]

        return {
            "data": {0: caption, 1: img},
            "label": torch.tensor(cid, dtype=torch.long)
        }

class CUB200_Dataloader:
    def __init__(self, config):
        batch = config.training_params.batch_size

        self.train_loader = DataLoader(
            CUB200_Dataset(config, "train"),
            batch_size=batch, shuffle=True)

        self.valid_loader = DataLoader(
            CUB200_Dataset(config, "val"),
            batch_size=batch, shuffle=False)

        self.test_loader = DataLoader(
            CUB200_Dataset(config, "test"),
            batch_size=batch, shuffle=False)

if __name__ == "__main__":
    import types

    config = types.SimpleNamespace()
    config.dataset = types.SimpleNamespace()
    config.training_params = types.SimpleNamespace()

    config.dataset.data_roots = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/CUB200/CUB_200_2011"
    config.training_params.batch_size = 4

    loader = CUB200_Dataloader(config)
    batch = next(iter(loader.train_loader))

    print("Caption:", batch["data"][0][0])
    print("Image shape:", batch["data"][1].shape)
    print("Label:", batch["label"][0])
