import os
from tqdm import tqdm
import torch

from mydatasets.MMIMDB.MMIMDBLoader import MMIMDb_Dataloader
from models.MCR_Models import TextToImage_SDXL   # ← update import if needed


DEVICE = "cuda"

TRAIN_DIR = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/MMIMDb/generated_posters_train"
VAL_DIR   = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/MMIMDb/generated_posters_val"
TEST_DIR  = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/MMIMDb/generated_posters_test"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def build_config(data_root, kaggle_cache, batch_size):
    import types
    cfg = types.SimpleNamespace()
    cfg.dataset = types.SimpleNamespace()
    cfg.training_params = types.SimpleNamespace()

    cfg.dataset.data_roots = data_root
    cfg.dataset.kaggle_cache_dir = kaggle_cache
    cfg.training_params.batch_size = batch_size
    return cfg


def run_split(dataloader, out_dir, model):
    """
    Generate posters for a full dataloader split and save as PNG files.
    """
    ensure_dir(out_dir)

    for batch in tqdm(dataloader, desc=f"Generating posters → {out_dir}"):
        # The model takes batch and returns list of PIL images
        posters = model(batch, seed=5, num_steps=20, guidance_scale=1.5)


        for sample_id, posters_for_id in zip(batch["ids"], posters):

            # posters_for_id might be: [ [img1], [img2], ... ]
            # normalize it so we have a list of PIL images
            flattened = []
            for item in posters_for_id:
                if isinstance(item, list):
                    flattened.extend(item)
                else:
                    flattened.append(item)

            # now save them
            for k, img in enumerate(flattened):
                save_path = os.path.join(out_dir, f"{sample_id}_{k:02d}.png")
                img.save(save_path)


def main():
    # -----------------------------------------------------------
    # SETTINGS
    # -----------------------------------------------------------
    DATA_ROOT = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/MMIMDb"
    KAGGLE_CACHE = "/esat/smcdata/users/kkontras/kaggle_cache"
    BATCH_SIZE = 2

    cfg = build_config(DATA_ROOT, KAGGLE_CACHE, BATCH_SIZE)
    loader = MMIMDb_Dataloader(cfg)

    train_loader = loader.train_loader
    val_loader   = loader.valid_loader
    test_loader  = loader.test_loader

    # -----------------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------------
    model = TextToImage_SDXL(args={},device=DEVICE).eval()

    # -----------------------------------------------------------
    # RUN ALL SPLITS
    # -----------------------------------------------------------
    run_split(train_loader, TRAIN_DIR, model)
    run_split(val_loader,   VAL_DIR,   model)
    run_split(test_loader,  TEST_DIR,  model)

    print(f"[DONE] Saved posters to:\n{TRAIN_DIR}\n{VAL_DIR}\n{TEST_DIR}")


if __name__ == "__main__":
    main()
