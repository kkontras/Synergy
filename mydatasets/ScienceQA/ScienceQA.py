import os
import json
import random
from typing import List, Dict, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from datasets import load_dataset
from collections import Counter

LETTERS_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ================================================================
#   LOAD + FILTER SCIENCEQA
# ================================================================

def load_scienceqa_filtered(
    data_root: str,
    split: str,
    require_image: bool = True,
    require_outside_knowledge: bool = True,
    drop_near_blank: bool = True,
    blank_std_thresh: float = 0.01,
):
    ds = load_dataset("derek-thomas/ScienceQA", split=split, cache_dir=data_root)

    keep_indices: List[int] = []

    for i, ex in enumerate(ds):
        pil_img = ex.get("image", None)
        has_img = pil_img is not None

        lecture = (ex.get("lecture") or "").strip()
        hint = (ex.get("hint") or "").strip()
        has_outside = (len(lecture) > 0) or (len(hint) > 0)

        if require_image and not has_img:
            continue
        if require_outside_knowledge and not has_outside:
            continue

        if drop_near_blank and has_img:
            t = to_tensor(pil_img)
            if t.std().item() < blank_std_thresh:
                continue

        keep_indices.append(i)

    print(
        f"[ScienceQA] Split={split}: kept {len(keep_indices)} / {len(ds)} examples "
        f"(image={require_image}, outside_k={require_outside_knowledge}, "
        f"drop_near_blank={drop_near_blank})"
    )

    return ds, keep_indices


# ================================================================
#   TEXT BUILDERS
# ================================================================

def build_scienceqa_hint_text(example: Dict[str, Any]) -> str:
    lecture = (example.get("lecture") or "").strip()
    hint = (example.get("hint") or "").strip()

    parts = []
    if lecture:
        parts.append("Science passage:\n" + lecture)
    if hint:
        parts.append("Hint:\n" + hint)

    return "\n\n".join(parts)


def build_question_text_with_letters(
    question: str,
    choices: List[str],
    letters: List[str],
) -> str:
    parts = []
    question = (question or "").strip()
    if question:
        parts.append("Question:\n" + question)

    assert len(choices) == len(letters)
    choice_lines = [f"({L}) {c}" for L, c in zip(letters, choices)]
    parts.append("Choices:\n" + "\n".join(choice_lines))

    return "\n\n".join(parts)


# ================================================================
#   DATASET
# ================================================================

class ScienceQA_Dataset(Dataset):
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

        self.raw_ds, self.keep_indices = load_scienceqa_filtered(
            data_root=config.dataset.data_roots,
            split=self.split,
            require_image=require_image,
            require_outside_knowledge=require_outside_knowledge,
            drop_near_blank=drop_near_blank,
            blank_std_thresh=blank_std_thresh,
        )

        self.train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.eval_tf = self.train_tf

        stats = compute_label_stats_and_weights(config=config, split="train", weight_mode="inv_freq", normalize="mean1")
        self.weights = torch.tensor(stats["weights"])

    def __len__(self):
        return len(self.keep_indices)
        # return 32

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
        #
        # if self.split == "train":
        #     target_pos = random.randrange(n)
        #
        #     perm = list(range(n))
        #     perm[orig_correct_idx], perm[target_pos] = (
        #         perm[target_pos],
        #         perm[orig_correct_idx],
        #     )
        #
        #     choices = [choices[i] for i in perm]
        #     letters_for_question = [letters_for_question[i] for i in perm]
        #
        #     correct_idx = LETTERS_POOL.index(letters_for_question[target_pos])
        # else:
        correct_idx = orig_correct_idx

        qa_text = build_question_text_with_letters(
            question=question,
            choices=choices,
            letters=letters_for_question,
        )

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

def scienceqa_collate_qwen(batch):
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

class ScienceQA_Dataloader:
    def __init__(self, config):
        batch_size = config.training_params.batch_size

        g = torch.Generator()
        g.manual_seed(0)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        self.collate_fn = scienceqa_collate_qwen

        self.train_loader = DataLoader(
            ScienceQA_Dataset(
                config= config,
                split="train",
                require_image=True,
                require_outside_knowledge=True,
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            worker_init_fn=seed_worker,
            collate_fn=self.collate_fn,
        )

        self.valid_loader = DataLoader(
            ScienceQA_Dataset(
                config=config,
                split="validation",
                require_image=True,
                require_outside_knowledge=True,
            ),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        self.test_loader = DataLoader(
            ScienceQA_Dataset(
                config=config,
                split="test",
                require_image=True,
                require_outside_knowledge=True,
            ),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

def compute_label_stats_and_weights(
    config,
    split="train",
    require_image=True,
    require_outside_knowledge=True,
    drop_near_blank=True,
    blank_std_thresh=0.01,
    weight_mode="inv_freq",   # "inv_freq" or "cui"
    beta=0.9999,              # only used for "cui"
    normalize="mean1",        # "none", "sumK", "mean1"
):
    ds, keep_indices = load_scienceqa_filtered(
        data_root=config.dataset.data_roots,
        split=split,
        require_image=require_image,
        require_outside_knowledge=require_outside_knowledge,
        drop_near_blank=drop_near_blank,
        blank_std_thresh=blank_std_thresh,
    )

    # Count answers and track max number of choices (K)
    counts = Counter()
    K = 0
    for i in keep_indices:
        ex = ds[i]
        ans = int(ex["answer"])
        counts[ans] += 1
        K = max(K, len(ex.get("choices", [])))

    # Build count vector of length K (e.g., 5 => A..E)
    count_vec = torch.zeros(K, dtype=torch.long)
    for k, v in counts.items():
        if 0 <= k < K:
            count_vec[k] = v

    total = int(count_vec.sum().item())
    freq = count_vec.float() / max(total, 1)

    # Convert counts -> weights
    if weight_mode == "inv_freq":
        # w_k = total / (K * count_k)
        w = total / (K * torch.clamp(count_vec.float(), min=1.0))
    elif weight_mode == "cui":
        # "Class-Balanced Loss" (Cui et al.): w_k ∝ (1 - beta) / (1 - beta^n_k)
        n = torch.clamp(count_vec.float(), min=1.0)
        w = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), n))
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    # Optional normalization
    if normalize == "sumK":
        w = w * (K / w.sum())
    elif normalize == "mean1":
        w = w / w.mean()
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize: {normalize}")

    return {
        "K": K,
        "counts": count_vec,
        "freq": freq,
        "weights": w,
        "keep_n": len(keep_indices),
        "total": total,
    }



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

    stats = compute_label_stats_and_weights(config=config, split="train", weight_mode="inv_freq", normalize="mean1")
    print("Max choices K =", stats["K"])
    print("Counts:", stats["counts"].tolist())
    print("Freq  :", [round(x, 6) for x in stats["freq"].tolist()])
    print("Wts   :", [round(x, 6) for x in stats["weights"].tolist()])


    loader = ScienceQA_Dataloader(config)
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
