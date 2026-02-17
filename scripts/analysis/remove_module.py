import torch
from collections import OrderedDict

path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/unimodal_audio_fold0_lr0.001_wd0.0001.pth.tar"
path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/unimodal_video_fold0_lr0.001_wd0.0001.pth.tar"

ckpt = torch.load(path, map_location="cpu", weights_only=False)

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[len("module."):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

# fix model_state_dict
if "model_state_dict" in ckpt:
    ckpt["model_state_dict"] = remove_module_prefix(ckpt["model_state_dict"])

# fix best_model_state_dict if exists
if "best_model_state_dict" in ckpt:
    ckpt["best_model_state_dict"] = remove_module_prefix(ckpt["best_model_state_dict"])

# save back
torch.save(ckpt, path)
print("Saved cleaned checkpoint back to:", path)
