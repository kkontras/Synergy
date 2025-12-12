import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn

from mydatasets.MMIMDB.MMIMDBLoader import MMIMDb_Dataloader

from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from torchvision.transforms.functional import to_pil_image


# ============================================================
#                   GLOBAL HF CACHE SETTINGS
# ============================================================

HF_CACHE = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy"
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_CACHE
os.environ["HF_MODULES_CACHE"] = HF_CACHE


# ============================================================
#                   STAGE 1 — BLIP MODEL
# ============================================================

class ImageToText_InstructBLIP(nn.Module):
    def __init__(self):
        super().__init__()

        model_name = "Salesforce/instructblip-flan-t5-xl"
        device = "cuda:0"
        self.device = device

        # Processor + BLIP model
        self.processor = InstructBlipProcessor.from_pretrained(model_name, cache_dir=HF_CACHE)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name, cache_dir=HF_CACHE
        ).to(device)

        # Pure visual grounding prompt
        self.default_prompt = (
            "Describe everything visible in the image with maximal detail, including characters, "
            "clothing, expressions, objects, background elements, text, colors, and layout. "
            "Do not infer or guess anything not strictly visible."
        )

    @torch.no_grad()
    def forward(self, input, prompt=None, num_variations=10):
        if prompt is None:
            prompt = self.default_prompt

        # Input images (B,3,H,W)
        x = input[2].detach()
        if x.max() > 1:
            x = x / 255.0
        x = x.clamp(0, 1)

        B = x.size(0)
        pil_images = [to_pil_image(img.float()) for img in x]

        # Duplicate V times
        batch_imgs, batch_prompts = [], []
        for _ in range(num_variations):
            for img in pil_images:
                batch_imgs.append(img)
                batch_prompts.append(prompt)

        inputs = self.processor(
            images=batch_imgs,
            text=batch_prompts,
            return_tensors="pt"
        ).to(self.device)

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=self.processor.tokenizer.eos_token_id
        )

        decoded = self.processor.batch_decode(gen_ids, skip_special_tokens=True)

        # Split per image
        results = [[] for _ in range(B)]
        idx = 0
        for _ in range(num_variations):
            for i in range(B):
                results[i].append(decoded[idx])
                idx += 1

        return results


# ============================================================
#                STAGE 2 — TEXT REWRITER MODEL
# ============================================================

class LocalLLM_Wrapper:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        print(f"[Rewriter] Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            import bitsandbytes
            quant = {"load_in_4bit": True}
        except ImportError:
            quant = {}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=HF_CACHE,
            device_map="auto",
            **quant
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )


class CaptionRewriter:
    def __init__(self, model):
        self.model = model.model           # raw HuggingFace model
        self.tokenizer = model.tokenizer   # tokenizer

        # Fixed rewrite prompt
        self.prompt_template = (
            "Transform the following visual description into a movie-style plot synopsis. "
            "Write only the plot. Do not mention images or descriptions. "
            "Start directly in the story world.\n\n"
            "Rules:\n"
            "- Use only the details present.\n"
            "- Turn static elements into events.\n"
            "- Maintain ambiguity.\n"
            "- Produce one continuous plot paragraph.\n\n"
            "Description:\n\"{caption}\"\n\n"
            "Plot:"
        )

    def rewrite_batch(self, captions, max_new_tokens=220):
        prompts = [self.prompt_template.format(caption=c) for c in captions]

        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # Batch generation on GPU
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)

        # Remove the prompt part
        cleaned = []
        for prompt, full_output in zip(prompts, decoded):
            if prompt in full_output:
                cleaned.append(full_output.split(prompt)[-1].strip())
            else:
                cleaned.append(full_output.strip())

        return cleaned

# ============================================================
#              FILE OPERATIONS FOR BOTH STAGES
# ============================================================

def run_blip_split(dataloader, output_path, blip_model, num_variations=10):
    with open(output_path, "w") as f:
        for i, batch in tqdm(enumerate(dataloader), desc=f"BLIP → {output_path}"):
            raw_caps = blip_model(batch["data"], num_variations=num_variations)

            for sample_id, caps in zip(batch["ids"], raw_caps):
                f.write(json.dumps({"id": sample_id, "raw_captions": caps}) + "\n")
            if i==5:
                break


def rewrite_with_microbatch(rewriter, captions, mb_size=2):
    outputs = []
    for i in range(0, len(captions), mb_size):
        chunk = captions[i:i+mb_size]
        out = rewriter.rewrite_batch(chunk)
        outputs.extend(out)
    return outputs


def run_rewriter(input_file, output_file, rewriter, keep_k=10):
    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in tqdm(f_in, desc=f"Rewriting → {output_file}"):
            rec = json.loads(line)
            raw_caps = rec["raw_captions"][:keep_k]

            rewritten = rewrite_with_microbatch(rewriter, raw_caps, mb_size=2)

            out = {
                "id": rec["id"],
                "raw_captions": raw_caps,
                "rewritten_captions": rewritten
            }
            f_out.write(json.dumps(out) + "\n")


# ============================================================
#                           MAIN
# ============================================================

def main():
    # -----------------------------
    # CONFIG: choose what to run
    # -----------------------------
    RUN_BLIP = True
    RUN_REWRITER = True

    DATA_ROOT = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/MMIMDb"
    KAGGLE_CACHE = "/esat/smcdata/users/kkontras/kaggle_cache"
    BATCH_SIZE = 1

    # -----------------------------
    # Paths for outputs
    # -----------------------------
    CAP_TRAIN = "captions_train.jsonl"
    CAP_VAL   = "captions_val.jsonl"
    CAP_TEST  = "captions_test.jsonl"

    PLOT_TRAIN = "plots_train.jsonl"
    PLOT_VAL   = "plots_val.jsonl"
    PLOT_TEST  = "plots_test.jsonl"

    # -----------------------------
    # Load dataset
    # -----------------------------
    import types
    cfg = types.SimpleNamespace(
        dataset=types.SimpleNamespace(
            data_roots=DATA_ROOT,
            kaggle_cache_dir=KAGGLE_CACHE,
        ),
        training_params=types.SimpleNamespace(batch_size=BATCH_SIZE)
    )
    loaders = MMIMDb_Dataloader(cfg)

    train_loader = loaders.train_loader
    val_loader   = loaders.valid_loader
    test_loader  = loaders.test_loader

    # ============================================================
    #                   STAGE 1: BLIP
    # ============================================================
    if RUN_BLIP:
        print("=== STAGE 1: BLIP VISUAL CAPTIONS ===")
        blip = ImageToText_InstructBLIP().eval()

        run_blip_split(train_loader, CAP_TRAIN, blip)
        # run_blip_split(val_loader,   CAP_VAL,   blip)
        # run_blip_split(test_loader,  CAP_TEST,  blip)

        print("BLIP stage finished.\n")
        del blip

    # ============================================================
    #                   STAGE 2: REWRITER
    # ============================================================
    if RUN_REWRITER:
        print("=== STAGE 2: REWRITING TO PLOTS ===")

        llm = LocalLLM_Wrapper()
        rewriter = CaptionRewriter(llm)

        run_rewriter(CAP_TRAIN, PLOT_TRAIN, rewriter)
        # run_rewriter(CAP_VAL,   PLOT_VAL,   rewriter)
        # run_rewriter(CAP_TEST,  PLOT_TEST,  rewriter)

        print("Rewriting stage finished.\n")

    print("=== ALL DONE ===")


if __name__ == "__main__":
    main()
