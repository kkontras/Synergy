import os
import random
import logging
from typing import Dict, List, Optional, Any

from PIL import Image
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from datasets import load_dataset  # HuggingFace downloader


# ============================================================
#   Helper: Download Flickr30k automatically into data_roots
# ============================================================

def download_flickr30k_to_folder(data_root: str):
    """
    Downloads Flickr30k from HuggingFace (nlphuji/flickr30k)
    and writes:
        <data_root>/
            images/
            captions.txt
            train.txt
            val.txt
            test.txt
    Only downloads once.
    """

    os.makedirs(data_root, exist_ok=True)

    images_dir = os.path.join(data_root, "images")
    captions_path = os.path.join(data_root, "captions.txt")
    train_txt = os.path.join(data_root, "train.txt")
    val_txt = os.path.join(data_root, "val.txt")
    test_txt = os.path.join(data_root, "test.txt")

    # If already downloaded, skip
    if (
        os.path.exists(images_dir)
        and os.path.exists(captions_path)
        and os.path.exists(train_txt)
        and os.path.exists(val_txt)
        and os.path.exists(test_txt)
    ):
        print(f"[Flickr30k] Already exists at {data_root}, skipping download.")
        return

    print("[Flickr30k] Downloading dataset from HuggingFace nlphuji/flickr30k ...")

    ds = load_dataset("nlphuji/flickr30k", cache_dir=data_root)

    # Create images folder
    os.makedirs(images_dir, exist_ok=True)

    # Build captions.txt format:
    #     <filename>#<idx>  <caption>
    with open(captions_path, "w") as fcap:

        def process_split(split, split_file_path):
            with open(split_file_path, "w") as fsplit:
                for i, item in enumerate(ds[split]):
                    # Save image file
                    fname = f"{item['image_id']}.jpg"
                    fsplit.write(fname + "\n")

                    # Save the image
                    img: Image.Image = item["image"]
                    img.save(os.path.join(images_dir, fname))

                    # 5 captions
                    for idx, caption in enumerate(item["captions"]):
                        caption = caption.strip()
                        fcap.write(f"{fname}#{idx}\t{caption}\n")

        process_split("train", train_txt)
        process_split("validation", val_txt)
        process_split("test", test_txt)

    print(f"[Flickr30k] Downloaded and prepared at: {data_root}")


# ============================================================
#               FLICKR30K DATASET CLASS
# ============================================================

class Flickr30kDataset(Dataset):
    """
    PyTorch Dataset for Flickr30k following the user's AVE dataset style.

    Directory structure expected (created automatically if downloading
    from HuggingFace):
        <data_root>/
            images/
                <image_id>.jpg
            captions.txt        # "<filename>#<idx>\t<caption>"
            train.txt
            val.txt
            test.txt

    Returns:
        {
            "data": { 0: caption_string, 1: image_tensor },
            "label": None
        }
        (In test: also includes negative distractors.)
    """

    def __init__(self, config: Any, mode: str = "train"):
        super().__init__()

        self.args = config
        self.mode = mode.lower()
        self.logger = logging.getLogger("Flickr30kDataset")

        # Root folder
        self.data_root: str = self.args.dataset.data_roots

        # Make sure Flickr30k is downloaded
        download_flickr30k_to_folder(self.data_root)

        # File paths
        self.img_root = os.path.join(self.data_root, "images")
        self.caption_file = os.path.join(self.data_root, "captions.txt")

        self.split_files = {
            "train": os.path.join(self.data_root, "train.txt"),
            "val": os.path.join(self.data_root, "val.txt"),
            "test": os.path.join(self.data_root, "test.txt"),
        }

        if self.mode not in self.split_files:
            raise ValueError(f"Invalid mode '{self.mode}'")

        # Load filenames
        with open(self.split_files[self.mode], "r") as f:
            self.filenames: List[str] = [line.strip() for line in f]

        # Load captions
        self.caption_dict: Dict[str, List[str]] = {}
        with open(self.caption_file, "r") as f:
            for line in f:
                img_id, caption = line.strip().split("\t")
                fname = img_id.split("#")[0]
                self.caption_dict.setdefault(fname, []).append(caption)

        # Image transforms
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        self.logger.info(f"[{self.mode}] Loaded {len(self.filenames)} Flickr30k samples.")

    # -------------------------------------------------------------
    def __len__(self):
        return len(self.filenames)

    # -------------------------------------------------------------
    def _load_image(self, filename: str) -> Tensor:
        img_path = os.path.join(self.img_root, filename)
        img = Image.open(img_path).convert("RGB")
        if self.mode == "train":
            return self.train_transform(img)
        return self.eval_transform(img)

    def _sample_caption(self, fname: str) -> str:
        return random.choice(self.caption_dict[fname])

    # -------------------------------------------------------------
    def __getitem__(self, idx: int):
        fname = self.filenames[idx]

        image_tensor = self._load_image(fname)
        caption = self._sample_caption(fname)

        if self.mode == "test":
            neg_idx = random.randint(0, len(self.filenames) - 1)
            neg_fname = self.filenames[neg_idx]

            neg_image_tensor = self._load_image(neg_fname)
            neg_caption = self._sample_caption(neg_fname)

            return {
                "data": {
                    0: caption,
                    1: image_tensor,
                    "0_random_indistr": neg_caption,
                    "1_random_indistr": neg_image_tensor,
                },
                "label": None,
            }

        return {"data": {0: caption, 1: image_tensor}, "label": None}


# ============================================================
#                   DATALOADER WRAPPER
# ============================================================

class Flickr30k_Dataloader:
    """
    Wraps Flickr30kDataset into train/val/test loaders.
    """

    def __init__(self, config: Any):
        self.config = config

        train_set, val_set, test_set, total_set = self._build_datasets()

        g = torch.Generator().manual_seed(0)

        cpu_cores = len(os.sched_getaffinity(0))
        num_workers = max(cpu_cores - 1, 0)

        logging.info(f"Flickr30k: using {num_workers} dataloader workers.")

        # TRAIN loader
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.config.training_params.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.config.training_params.pin_memory,
            generator=g,
            worker_init_fn=lambda wid: np.random.seed(15 + wid),
        )

        # VAL loader
        self.valid_loader = DataLoader(
            val_set,
            batch_size=self.config.training_params.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.config.training_params.pin_memory,
        )

        # TEST loader
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.config.training_params.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.config.training_params.pin_memory,
        )

        # TOTAL loader (train split by default)
        self.total_loader = DataLoader(
            total_set,
            batch_size=self.config.training_params.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.config.training_params.pin_memory,
        )

    # -------------------------------------------------------------
    def _build_datasets(self):
        train = Flickr30kDataset(self.config, mode="train")
        val = Flickr30kDataset(self.config, mode="val")
        test = Flickr30kDataset(self.config, mode="test")
        total = Flickr30kDataset(self.config, mode="train")
        return train, val, test, total



if __name__ == "__main__":
    import types

    # ---- Build a minimal config-like object ----
    config = types.SimpleNamespace()
    config.dataset = types.SimpleNamespace()
    config.training_params = types.SimpleNamespace()

    # Paths & hyperparameters
    config.dataset.data_roots = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Flickr30k"   # where to download
    config.training_params.batch_size = 4
    config.training_params.test_batch_size = 4
    config.training_params.pin_memory = False

    # ---- Instantiate loaders ----
    print("Initializing Flickr30k dataloaders...")
    loader = Flickr30k_Dataloader(config)

    # ---- Example: iterate over one training batch ----
    batch = next(iter(loader.train_loader))
    print("\nBatch keys:", batch.keys())
    print("Data keys:", batch["data"].keys())
    print("Caption example:", batch["data"][0][0])
    print("Image tensor shape:", batch["data"][1].shape)