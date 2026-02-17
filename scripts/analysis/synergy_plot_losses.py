import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path
import copy
import matplotlib


# m1_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_fold0_w11_w21_mlp.pth.tar"
# m1_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_fold0_w11_w21_conformer.pth.tar"
# m1_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_pre_fold0_w11_w21_mlp.pth.tar"
# m1_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_pre_fold0_w11_w21_conformer.pth.tar"
m1_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_unireg_pre_fold0_w11_w21_mlp.pth.tar"
m1_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_unireg_fold0_w11_w21_mlp_dmetriccosine.pth.tar"
m1_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_onlyq_fold0_w11_w21_mlp_pre.pth.tar"
m1_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_onlyq_fold0_w11_w21_mlp.pth.tar"
m1_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_onlyq_fold0_w11_w20.1_mlp_pre.pth.tar"
# m1_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_unireg_pre_fold0_w11_w21_conformer.pth.tar"
# m1_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_unireg_fold0_w11_w21_mlp.pth.tar"
# m1_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_unireg_fold0_w11_w21_conformer.pth.tar"
m1 = torch.load(m1_path, map_location='cpu')

logs = m1["logs"]

# reg_0 = np.array([logs["train_logs"][i]["loss"]["sr_yz1z2"] for i in logs["train_logs"]])
reg_1 = np.array([logs["train_logs"][i]["loss"]["sr_z1z2"] for i in logs["train_logs"]])
# max_value = max(np.max(reg_0), np.max(reg_1))
best_step = logs["best_logs"]["step"]
steps = np.array([i for i in logs["train_logs"]])

# m2_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_fold0_w10_w20_mlp.pth.tar"
# m2_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_fold0_w10_w20_conformer.pth.tar"
m2_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_pre_fold0_w10_w20_mlp.pth.tar"
# m2_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_pre_fodld0_w10_w20_conformer.pth.tar"
# m2_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_unireg_pre_fold0_w10_w20_mlp.pth.tar"
# m2_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_unireg_pre_fold0_w10_w20_conformer.pth.tar"
# m2_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_unireg_fold0_w10_w20_mlp.pth.tar"
# m2_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/Res_Synprom_unireg_fold0_w10_w20_conformer.pth.tar"
m2 = torch.load(m2_path, map_location='cpu')
m2_logs = m2["logs"]
m2_best_step = m2_logs["best_logs"]["step"]


cue_syn_0 = np.array([logs["val_logs"][i]["ceu"]["combined"]["synergy"] for i in logs["val_logs"]])
cue_syn_1 = np.array([m2_logs["val_logs"][i]["ceu"]["combined"]["synergy"] for i in m2_logs["val_logs"]])
cue_steps_0 = np.array([i for i in logs["val_logs"]])
cue_steps_1 = np.array([i for i in m2_logs["val_logs"]])


# smooth the curves way more
# smoothing = 100
# cue_syn_0 = np.convolve(cue_syn_0, np.ones(smoothing)/smoothing, mode='valid')
# cue_syn_1 = np.convolve(cue_syn_1, np.ones(smoothing)/smoothing, mode='valid')

reg_1 = -1*reg_1


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ---- Subfigure 1 ----
# axes[0].plot(steps, reg_0, label="YZ1Z2")
axes[0].plot(steps, reg_1, label="Z1Z2")
for i in range(0, 5,5):
    axes[0].axhline(i, color='gray', linestyle='--', linewidth=1)

axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
# axes[0].set_ylim([0, max_value.astype(int)])
axes[0].set_xlabel("Optimization Steps", fontsize=14)
axes[0].set_ylabel("Loss Value", fontsize=14)
axes[0].legend(fontsize=14)

# ---- Subfigure 2 ----
# axes[1].plot(steps, reg_1 - reg_0, label="Z1Z2 - YZ1Z2", color="green")
for i in range(0, 5,5):
    axes[1].axhline(i, color='gray', linestyle='--', linewidth=1)

axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
# axes[1].set_ylim([0, max_value.astype(int)])
axes[1].set_xlabel("Optimization Steps", fontsize=14)
axes[1].set_ylabel("Loss Value", fontsize=14)
axes[1].legend(fontsize=14)


axes[2].plot(cue_steps_0, cue_syn_0, label="With Reg", color="red")
axes[2].plot(cue_steps_1, cue_syn_1, label="Without Reg", color="black")
# axes[2].plot(cue_steps_0, cue_syn_1[:len(cue_steps_0)], label="Without Reg", color="black")
for i in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    axes[2].axhline(i, color='gray', linestyle='--', linewidth=1)
axes[2].axvline(best_step, color='red', linestyle=':', linewidth=1)
# axes[2].axvline(m2_best_step, color='black', linestyle=':', linewidth=1)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
# axes[1].set_ylim([0, 3])
axes[2].set_xlabel("Optimization Steps", fontsize=14)
axes[2].set_ylabel("Accuracy on 'Both False' Cases", fontsize=14)
axes[2].legend(fontsize=14)

# ---- Common title ----
fig.suptitle("Regularization Loss Curves", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
plt.show()

