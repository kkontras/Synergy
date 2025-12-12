import os
import json
import random
from PIL import Image
from tqdm import tqdm

import kagglehub

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTokenizer

# ================================================================
#   LOCATE OR DOWNLOAD DATASET
# ================================================================

def get_mmimdb_root(config):
    base = config.dataset.data_roots

    local_path = os.path.join(
        base,
        "datasets",
        "eduardschipatecua",
        "mmimdb",
        "versions",
        "1",
        "mmimdb"
    )

    if os.path.exists(local_path):
        print("[MM-IMDb] Using existing dataset:", local_path)
        return local_path

    if config.dataset.kaggle_cache_dir:
        os.environ["KAGGLEHUB_CACHE"] = config.dataset.kaggle_cache_dir

    print("[MM-IMDb] Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("eduardschipatecua/mmimdb")
    return os.path.join(path, "mmimdb")


# ================================================================
#   PREPARE DATASET STRUCTURE
# ================================================================

def prepare_mmimdb(root):
    flag = os.path.join(root, "READY.flag")
    if os.path.exists(flag):
        print("[MM-IMDb] Already prepared.")
        return

    split_path = os.path.join(root, "split.json")
    data_dir = os.path.join(root, "dataset")

    if not os.path.exists(split_path):
        raise FileNotFoundError("split.json missing")
    if not os.path.exists(data_dir):
        raise FileNotFoundError("dataset/ missing")

    print("[MM-IMDb] Loading split.json...")
    with open(split_path, "r") as f:
        splits = json.load(f)

    train_ids = splits["train"]
    val_ids   = splits["dev"]
    test_ids  = splits["test"]

    # ===========================================================
    #   Load JSON metadata from dataset/<id>.json
    # ===========================================================
    print("[MM-IMDb] Collecting metadata from JSON files...")

    labels = {}
    captions = {}
    all_genres = set()

    def read_item(mid):
        json_path = os.path.join(data_dir, f"{mid}.json")
        if not os.path.exists(json_path):
            return None
        with open(json_path, "r") as f:
            return json.load(f)

    # first gather genres
    for mid in tqdm(train_ids + val_ids + test_ids):
        meta = read_item(mid)
        if meta is None:
            continue
        for g in meta["genres"]:
            all_genres.add(g)

    all_genres = sorted(list(all_genres))
    genre2id = {g: i for i, g in enumerate(all_genres)}

    # build per-file labels + captions
    def process_ids(id_list):
        fnames = []
        for mid in id_list:
            meta = read_item(mid)
            if meta is None:
                continue

            fname = f"{mid}.jpeg"
            fnames.append(fname)

            captions[fname] = meta["plot"]

            y = [0] * len(all_genres)
            for g in meta["genres"]:
                y[genre2id[g]] = 1
            labels[fname] = y

        return fnames

    print("[MM-IMDb] Processing splits...")
    train_f = process_ids(train_ids)
    val_f   = process_ids(val_ids)
    test_f  = process_ids(test_ids)

    # write split files
    def write_list(lst, path):
        with open(path, "w") as f:
            for x in lst:
                f.write(x + "\n")

    write_list(train_f, os.path.join(root, "train.txt"))
    write_list(val_f,   os.path.join(root, "val.txt"))
    write_list(test_f,  os.path.join(root, "test.txt"))

    # save metadata
    with open(os.path.join(root, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)

    with open(os.path.join(root, "captions.txt"), "w") as f:
        for fname, text in captions.items():
            f.write(f"{fname}#0\t{text}\n")

    with open(os.path.join(root, "genres.json"), "w") as f:
        json.dump(all_genres, f, indent=2)

    open(flag, "w").write("OK")
    print("[MM-IMDb] Preparation complete.")


# ================================================================
#   DATASET
# ================================================================

class MMIMDb_Dataset(Dataset):
    def __init__(self, config, mode="train"):
        root = get_mmimdb_root(config)
        prepare_mmimdb(root)

        self.root = root
        self.mode = mode.lower()

        self.img_dir = os.path.join(self.root, "dataset")

        with open(os.path.join(self.root, f"{self.mode}.txt"), "r") as f:
            self.filenames = [line.strip() for line in f]

        self.labels = json.load(open(os.path.join(self.root, "labels.json")))

        # load captions
        self.caption_dict = {}
        with open(os.path.join(self.root, "captions.txt"), "r") as f:
            for line in f:
                key, caption = line.strip().split("\t")
                fname = key.split("#")[0]
                self.caption_dict[fname] = [caption]

        self.train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
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

        caption = self.caption_dict[fname][0]
        label = torch.tensor(self.labels[fname], dtype=torch.float32)

        sample_id = os.path.splitext(fname)[0]

        return {
            "id": sample_id,
            "data": {0: caption, 1: img},
            "label": label
        }


def pad_1d_tensors(tensor_list, pad_value=0):
    max_len = max(t.size(0) for t in tensor_list)
    out = torch.full((len(tensor_list), max_len), pad_value, dtype=torch.uint8)
    for i, t in enumerate(tensor_list):
        out[i, :t.size(0)] = t
    return out


def mmimdb_collate_dual(batch, bert_tokenizer, clip_tokenizer):
    texts = [b["data"][0] for b in batch]      # RAW captions (strings)
    images = [b["data"][1] for b in batch]
    labels = [b["label"] for b in batch]
    ids = [b["id"] for b in batch]
    # DistilBERT tokens
    bert_tok = bert_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # CLIP/SD tokenizer
    clip_tok = clip_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )

    image_batch = torch.stack(images, dim=0)
    label_batch = torch.stack(labels, dim=0)

    return {
        "data": {
            0: {"input_ids": bert_tok["input_ids"],
                "attention_mask": bert_tok["attention_mask"]},

            1: {"input_ids": clip_tok["input_ids"],
                "attention_mask": clip_tok["attention_mask"]},

            2: image_batch,

            3: texts
        },
        "ids": ids,
        "label": label_batch,
    }

class MMIMDb_Dataloader:
    def __init__(self, config):
        batch = config.training_params.batch_size
        bert_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        # clip_tok = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14")

        self.collate_fn = lambda batch: mmimdb_collate_dual(
            batch, bert_tok, bert_tok
        )

        self.train_loader = DataLoader(
            MMIMDb_Dataset(config, "train"),
            batch_size=batch,
            shuffle=True,
            collate_fn=self.collate_fn
        )

        self.valid_loader = DataLoader(
            MMIMDb_Dataset(config, "val"),
            batch_size=batch,
            shuffle=False,
            collate_fn=self.collate_fn
        )

        self.test_loader = DataLoader(
            MMIMDb_Dataset(config, "test"),
            batch_size=batch,
            shuffle=False,
            collate_fn=self.collate_fn
        )




# ================================================================
#   TEST
# ================================================================

if __name__ == "__main__":
    import types

    config = types.SimpleNamespace()
    config.dataset = types.SimpleNamespace()
    config.training_params = types.SimpleNamespace()

    config.dataset.data_roots = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/MMIMDb"
    config.dataset.kaggle_cache_dir = "/esat/smcdata/users/kkontras/kaggle_cache"
    config.training_params.batch_size = 4

    loader = MMIMDb_Dataloader(config)
    batch = next(iter(loader.train_loader))

    print("Caption:", batch["data"][0][0][:200], "…")
    print("Image shape:", batch["data"][1].shape)
    print("Label:", batch["label"][0])
