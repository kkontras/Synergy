import os
import json
import shutil
import zipfile
import random
from typing import Dict, Any, List

import kagglehub
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ===========================================================
#                DOWNLOAD COCO-2017 VIA KAGGLEHUB
# ===========================================================

import os
import json
import shutil
import zipfile
from PIL import Image
from tqdm import tqdm

import fiftyone as fo
import fiftyone.zoo as foz


import os
import json
import shutil
from tqdm import tqdm
from PIL import Image

import fiftyone as fo
import fiftyone.zoo as foz


def download_coco2017_fiftyone(data_root: str):
    """
    Downloads COCO-2017 using FiftyOne WITHOUT MongoDB or ~/.fiftyone storage.
    Produces the following inside data_root:

        data_root/
            images/
                train2017/
                val2017/
                test2017/
            annotations/
                captions_train2017.json
                captions_val2017.json
            captions.txt
            labels.json
            train.txt
            val.txt
            test.txt
            READY.flag

    This function is safely rerunnable: if READY.flag exists, downloading is skipped.
    """

    os.makedirs(data_root, exist_ok=True)
    marker = os.path.join(data_root, "READY.flag")

    # -------------------------
    #   SKIP IF ALREADY DONE
    # -------------------------
    if os.path.exists(marker):
        print("[COCO-2017] Already prepared at:", data_root)
        return

    # -------------------------
    #   FIFTYONE CONFIG
    # -------------------------
    print("\n=== Configuring FiftyOne for MongoDB-free mode ===")
    fo.config.require_mongo = False
    fo.config.database_validation = False

    # Force all data to be downloaded HERE
    fo.config.dataset_zoo_dir = data_root
    fo.config.default_dataset_dir = data_root

    print("FiftyOne will store dataset at:", fo.config.dataset_zoo_dir)

    # -------------------------
    #   DOWNLOAD SPLITS
    # -------------------------
    print("\n=== Downloading COCO-2017 TRAIN split ===")
    dataset_train = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        overwrite=False,
        cleanup=False,
    )

    print("\n=== Downloading COCO-2017 VAL split ===")
    dataset_val = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        overwrite=False,
        cleanup=False,
    )

    print("\n=== Downloading COCO-2017 TEST split ===")
    dataset_test = foz.load_zoo_dataset(
        "coco-2017",
        split="test",
        label_types=None,   # test has no labels
        overwrite=False,
        cleanup=False,
    )

    # -------------------------
    #   PREP DIRECTORIES
    # -------------------------
    images_dir = os.path.join(data_root, "images")
    ann_dir = os.path.join(data_root, "annotations")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # -------------------------
    #   COPY IMAGES
    # -------------------------
    def copy_split(dataset, split_name):
        split_folder = os.path.join(images_dir, split_name)
        os.makedirs(split_folder, exist_ok=True)
        print(f"[Copy] {split_name} images...")

        filelist = []
        for sample in tqdm(dataset):
            src = sample.filepath
            fname = os.path.basename(src)
            dst = os.path.join(split_folder, fname)

            if not os.path.exists(dst):
                shutil.copy(src, dst)

            filelist.append(fname)

        return filelist

    train_list = copy_split(dataset_train, "train2017")
    val_list   = copy_split(dataset_val,   "val2017")
    test_list  = copy_split(dataset_test,  "test2017")

    # -------------------------
    #   WRITE SPLIT FILES
    # -------------------------
    def write_list(lst, outpath):
        with open(outpath, "w") as f:
            for x in lst:
                f.write(x + "\n")

    write_list(train_list, os.path.join(data_root, "train.txt"))
    write_list(val_list,   os.path.join(data_root, "val.txt"))
    write_list(test_list,  os.path.join(data_root, "test.txt"))

    # -------------------------
    #   BUILD 80-CLASS LABELS
    # -------------------------
    print("\n[COCO] Extracting 80-class detection labels...")
    categories = dataset_train.default_classes
    num_classes = len(categories)
    class2idx = {c: i for i, c in enumerate(categories)}

    image_labels = {}

    def update_labels(dataset, split_name):
        print(f"[COCO] Extracting labels for {split_name}...")
        for sample in dataset:
            fname = os.path.basename(sample.filepath)

            # Test split → no detections
            dets = getattr(sample, "detections", None)
            if dets is None:
                continue

            multi_hot = image_labels.get(fname, [0] * num_classes)

            if dets.detections:
                for det in dets.detections:
                    cid = class2idx[det.label]
                    multi_hot[cid] = 1

            image_labels[fname] = multi_hot

    update_labels(dataset_train, "train")
    update_labels(dataset_val, "validation")

    json.dump(
        image_labels,
        open(os.path.join(data_root, "labels.json"), "w"),
        indent=2,
    )

    # -------------------------
    #   DOWNLOAD CAPTIONS
    # -------------------------
    print("\n[COCO] Downloading captions...")
    foz.download_zoo_dataset(
        "coco-2017",
        splits=["train", "validation"],
        label_types=["captions"]
    )

    cap_train_json = os.path.join(data_root, "annotations", "captions_train2017.json")
    cap_val_json   = os.path.join(data_root, "annotations", "captions_val2017.json")

    print("[COCO] Writing captions.txt...")
    captions_out = os.path.join(data_root, "captions.txt")

    with open(captions_out, "w") as fcap:

        # TRAIN captions
        caps = json.load(open(cap_train_json))
        for ann in caps["annotations"]:
            fname = f"{ann['image_id']:012d}.jpg"
            caption = ann["caption"].strip()
            fcap.write(f"{fname}#0\t{caption}\n")

        # VAL captions
        caps = json.load(open(cap_val_json))
        for ann in caps["annotations"]:
            fname = f"{ann['image_id']:012d}.jpg"
            caption = ann["caption"].strip()
            fcap.write(f"{fname}#0\t{caption}\n")

    # -------------------------
    #   FINISH
    # -------------------------
    open(marker, "w").write("OK")
    print(f"\n[COCO-2017] Dataset fully prepared at:\n{data_root}\n")


# ===========================================================
#                    TORCH DATASET
# ===========================================================

class COCO2017_Dataset(Dataset):
    def __init__(self, config, mode="train"):
        self.root = config.dataset.data_roots
        self.mode = mode.lower()

        download_coco2017_fiftyone(self.root)

        self.img_dir = os.path.join(self.root, "images")
        self.caption_file = os.path.join(self.root, "captions.txt")
        labels_json = os.path.join(self.root, "labels.json")

        splits = {
            "train": os.path.join(self.root, "train.txt"),
            "val":   os.path.join(self.root, "val.txt"),
            "test":  os.path.join(self.root, "test.txt"),
        }

        # Load filenames
        with open(splits[self.mode], "r") as f:
            self.filenames = [x.strip() for x in f]

        # Captions
        self.caption_dict = {}
        with open(self.caption_file, "r") as f:
            for line in f:
                key, txt = line.strip().split("\t")
                fname = key.split("#")[0]
                self.caption_dict.setdefault(fname, []).append(txt)

        # Labels
        self.labels = json.load(open(labels_json))

        # Transforms
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
        caption = random.choice(self.caption_dict.get(fname, [""]))
        label = torch.tensor(self.labels.get(fname, [0]*80), dtype=torch.float32)

        return {"data": {0: caption, 1: img}, "label": label}


# ===========================================================
#                    DATALOADER
# ===========================================================

class COCO2017_Dataloader:
    def __init__(self, config):
        batch = config.training_params.batch_size

        self.train_loader = DataLoader(COCO2017_Dataset(config, "train"),
                                       batch_size=batch, shuffle=True)

        self.valid_loader = DataLoader(COCO2017_Dataset(config, "val"),
                                       batch_size=batch, shuffle=False)

        self.test_loader  = DataLoader(COCO2017_Dataset(config, "test"),
                                       batch_size=batch, shuffle=False)


# ===========================================================
#                       MAIN TEST
# ===========================================================

if __name__ == "__main__":
    import types

    config = types.SimpleNamespace()
    config.dataset = types.SimpleNamespace()
    config.training_params = types.SimpleNamespace()

    config.dataset.data_roots = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/COCO2017_Kaggle"
    config.training_params.batch_size = 4

    loader = COCO2017_Dataloader(config)
    batch = next(iter(loader.train_loader))

    print("\nBatch keys:", batch.keys())
    print("Caption:", batch["data"][0][0])
    print("Image shape:", batch["data"][1].shape)
    print("Label length:", len(batch["label"][0]))
