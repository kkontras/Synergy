import os
import json
import random
import pynvml
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from datasets import load_dataset
from collections import Counter
import multiprocessing
from accelerate import Accelerator
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class DAVE_Dataset(Dataset):
    """
    item = {
        "id": sample_id,
        "data": {
            0: hint_text,
            1: qa_text,          # question + lettered choices
            2: img_tensor,
            3: choices,          # permuted choice texts
            4: letters,          # permuted letters
        },
        "label": correct_index  # LongTensor scalar
    }
    """

    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "train",
        require_image: bool = True,
        require_outside_knowledge: bool = True,
        image_size: int = 224,
        drop_near_blank: bool = True,
        blank_std_thresh: float = 0.01,
    ):
        super().__init__()
        self.split = split.lower()
        self.image_size = image_size
        ego4d_dataset = load_dataset("gorjanradevski/dave", split="ego4d", keep_in_memory=True, trust_remote_code=True)

        self.raw_ds, self.keep_indices = load_scienceqa_filtered(
            data_root=config.dataset.data_roots,
            split=self.split,
            require_image=require_image,
            require_outside_knowledge=require_outside_knowledge,
            drop_near_blank=drop_near_blank,
            blank_std_thresh=blank_std_thresh,
        )

        self.text_lens, self.keep_indices = compute_lengths_and_filter(
            raw_ds=self.raw_ds,
            keep_indices=self.keep_indices,
            min_len=getattr(config.dataset, "min_chars", 200),
            max_len=getattr(config.dataset, "max_chars", 1800),
            mode=getattr(config.dataset, "length_mode", "chars"),
            build_hint_fn=build_scienceqa_hint_text,
            build_qa_fn=build_question_text_with_letters,
        )

        print( f"[ScienceQA] length filter: kept {len(self.keep_indices)} / {len(self.text_lens)} ")

        self.train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.eval_tf = self.train_tf

        if split=="train":
            stats = compute_label_stats_and_weights(self.raw_ds, self.keep_indices, weight_mode="inv_freq", normalize="mean1")
            self.weights = stats["weights"].clone().detach()

    def __len__(self):
        return len(self.keep_indices)

    def _load_image(self, pil_img: Image.Image):
        pil_img = pil_img.convert("RGB")
        if self.split == "train":
            return self.train_tf(pil_img)
        else:
            return self.eval_tf(pil_img)

    def __getitem__(self, idx: int):
        real_idx = self.keep_indices[idx]
        ex = self.raw_ds[real_idx]

        pil_img = ex["image"]
        img_tensor = self._load_image(pil_img)

        hint_text = build_scienceqa_hint_text(ex)
        choices = list(ex.get("choices", []))
        question = ex.get("question")
        orig_correct_idx = ex["answer"]
        n = len(choices)

        letters_for_question = LETTERS_POOL[:n]
        correct_idx = orig_correct_idx

        qa_text = build_question_text_with_letters(question=question,choices=choices,letters=letters_for_question)

        label = torch.tensor(correct_idx, dtype=torch.long)
        sample_id = f"{self.split}_{real_idx}"

        return {
            "id": sample_id,
            "data": {
                0: hint_text,
                1: qa_text,
                2: img_tensor,
                3: choices,
                4: letters_for_question,
            },
            "label": label,
        }

def dave_collate_qwen(batch):
    hint_texts   = [b["data"][0] for b in batch]
    qa_texts     = [b["data"][1] for b in batch]
    images       = [b["data"][2] for b in batch]
    choices_list = [b["data"][3] for b in batch]
    letter_lists = [b["data"][4] for b in batch]
    labels       = [b["label"] for b in batch]
    ids          = [b["id"] for b in batch]

    image_batch = torch.stack(images, dim=0)
    label_batch = torch.stack(labels, dim=0)

    return {
        "data": {
            0: hint_texts,
            1: qa_texts,
            2: image_batch,
            3: choices_list,
            4: letter_lists,
        },
        "ids": ids,
        "label": label_batch,
    }


# ================================================================
#   DATALOADER WRAPPER
# ================================================================

class DAVE_Dataloader:
    def __init__(self, config):
        batch_size = config.training_params.batch_size

        g = torch.Generator()
        g.manual_seed(0)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        self.collate_fn = dave_collate_qwen

        def get_physical_gpu_count():
            try:
                pynvml.nvmlInit()
                count = pynvml.nvmlDeviceGetCount()
                pynvml.nvmlShutdown()
                return count
            except Exception as e:
                return f"Could not query NVML: {e}"

        total_cpus = multiprocessing.cpu_count()
        num_gpus = get_physical_gpu_count()
        workers_per_gpu = max(1, (total_cpus - 1) // num_gpus)
        # workers_per_gpu = 16

        print(
            f"[ScienceQA] GPUs: {torch.cuda.device_count()} (Phys: {num_gpus}) | SLURM: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')} | CPUs: {total_cpus} | Workers: {torch.cuda.device_count()}x{workers_per_gpu}={torch.cuda.device_count() * workers_per_gpu}")

        self.train_loader = DataLoader(
            DAVE_Dataset(
                config=config,
                split="train"
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            worker_init_fn=seed_worker,
            collate_fn=self.collate_fn,
            # --- ADD THESE FOR H100 PERFORMANCE ---
            num_workers=workers_per_gpu,  # Start with 8-12 per GPU (e.g., 48 total if on one node)
            pin_memory=True,  # Speeds up CPU-to-GPU transfer
            prefetch_factor=4,  # Ensures workers stay ahead of the GPU
            persistent_workers=True  # Keeps workers alive between epochs
        )

        self.valid_loader = DataLoader(
            DAVE_Dataset(
                config=config,
                split="validation"
            ),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=workers_per_gpu,  # Start with 8-12 per GPU (e.g., 48 total if on one node)
            pin_memory=True,  # Speeds up CPU-to-GPU transfer
            prefetch_factor=4,  # Ensures workers stay ahead of the GPU
            persistent_workers=True  # Keeps workers alive between epochs
        )

        self.test_loader = DataLoader(
            DAVE_Dataset(
                config=config,
                split="test"
            ),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=workers_per_gpu,  # Start with 8-12 per GPU (e.g., 48 total if on one node)
            pin_memory=True,  # Speeds up CPU-to-GPU transfer
            prefetch_factor=4,  # Ensures workers stay ahead of the GPU
            persistent_workers=True  # Keeps workers alive between epochs
        )




if __name__ == "__main__":
    import types
    import matplotlib
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F
    from collections import Counter
    matplotlib.use("Agg")

    config = types.SimpleNamespace()
    config.dataset = types.SimpleNamespace()
    config.training_params = types.SimpleNamespace()
    config.training_params.batch_size = 16

    loader = DAVE_Dataloader(config)
    batch = next(iter(loader.train_loader))

    label_counts = Counter(batch["label"].tolist())
    print("Label distribution in batch:", label_counts)


    print("IDs:", batch["ids"])
    print("\n--- Hint text sample ---\n", batch["data"][0][0][:400], "…")
    print("\n--- Q/A text sample ---\n", batch["data"][1][0][:400], "…")
    print("Image batch shape:", batch["data"][2].shape)
    print("Label indices:", batch["label"])
    print("Choices for sample 0:", batch["data"][3][0])
    print("Letters for sample 0:", batch["data"][4][0])

    img_tensor = batch["data"][2][0]
    img = F.to_pil_image(img_tensor)

    plt.imshow(img)
    plt.title(f"Sample ID: {batch['ids'][0]}")
    plt.axis("off")
    plt.savefig("scienceqa_sample.png")
    print("Saved scienceqa_sample.png")

    print("Choices for sample 0:", batch["data"][3][0])
    print("Letters for sample 0:", batch["data"][4][0])

    img_tensor = batch["data"][2][0]
    img = F.to_pil_image(img_tensor)

    plt.imshow(img)
    plt.title(f"Sample ID: {batch['ids'][0]}")
    plt.axis("off")
    plt.savefig("scienceqa_sample.png")
    print("Saved scienceqa_sample.png")
