import copy
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from matplotlib.patches import Polygon
import seaborn as sns
from matplotlib.lines import Line2D
from collections import defaultdict

sys.exc_info()
os.chdir('/users/sista/kkontras/Documents/MCR_Release/')

import pickle

def plot_accuracies_bar_next(data, std_data, label, dataset_name):

    datasets = data.keys()
    base_color = sns.color_palette("muted", n_colors=len(datasets))  # Use one tone for simplicity

    fig = plt.figure(figsize=(15, 6))

    offset_model = 0.35
    offset_dataset = 0.3*offset_model
    # Plot individual points
    min_data = 100
    max_data = 0

    for di, dataset in enumerate(datasets):
        this_data = [data[dataset][i] if data[dataset][i] is not None else False for i in np.arange(0, len(data[dataset]))]
        this_std = [std_data[dataset][i] if data[dataset][i] is not None else False for i in np.arange(0, len(data[dataset]))]
        color_i = 0 if di==1 else 1
        # color_i = 1 if di==1 else 0
        if len(datasets)==1: color_i = 0
        this_data = [i - this_data[0] for i in this_data]
        #remove first element
        this_data = this_data[1:]
        this_std = this_std[1:]
        min_data = min(min(this_data), min_data)
        max_data = max(max(this_data), max_data)
        plt.bar(np.arange(len(data[dataset])-1)*offset_model+di*offset_dataset, this_data,  yerr=this_std, width=0.15, color=base_color[color_i], label=dataset, alpha=0.8,error_kw={'ecolor': base_color[color_i], 'alpha':0.5, 'capsize':5})
        # plt.errorbar(np.arange(len(data[dataset]))*offset_model+di*offset_dataset, this_data, yerr=this_std, color=base_color[di], alpha=0.2)
        #remove line from errorbar and keep only the errorbar
        # plt.errorbar(np.arange(len(data[dataset]))*offset_model+di*offset_dataset, this_data, yerr=this_std, capsize=5, fmt='none', color=base_color[di], alpha=0.5)
        # plt.plot(np.arange(len(data[dataset]))*offset_model+di*offset_dataset, this_data, color=base_color[di], alpha=0.2)
    values, labels = [], []
    for di, dataset in enumerate(datasets):
        if dataset_name == "CREMA-D" or dataset_name == "AVE":
            values.append(np.arange(len(data[dataset])-1)*offset_model+di*offset_dataset+0.05)
        else:
            values.append(np.arange(len(data[dataset])-1)*offset_model+di*offset_dataset)
        labels.append(label[1:])
        if di==0:
            break
    #flatten lists
    # print(values)
    # print(labels)
    values = [item for sublist in values for item in sublist]
    labels = [item for sublist in labels for item in sublist]



    if dataset_name == "AVE":
        tickfont = 18
        labels_font = 24
    elif dataset_name == "CREMA-D":
        tickfont = 18
        labels_font = 24
    elif dataset_name == "UCF":
        tickfont = 18
        labels_font = 24
    elif dataset_name == "CMU-MOSEI":
        tickfont = 18
        labels_font = 24


    plt.xticks(values, labels, rotation=30, ha="right", fontsize=tickfont)
    plt.yticks(fontsize=tickfont)
    # plt.title("Accuracy Comparison Across Models on {} Dataset".format(dataset_name), fontsize=20)
    # plt.ylabel("Accuracy (%)", fontsize=labels_font)
    plt.ylabel("Accuracy Difference \n with Ensemble (%)", fontsize=labels_font)

    for di, dataset in enumerate(datasets):
        ensemble_value = data[dataset][label.index("Ensemble")]

        color_i = 0 if di == 1 else 1
        if len(datasets) == 1: color_i = 0
        plt.plot([0*offset_model+di*offset_dataset, len(values)*offset_model+di*offset_dataset], [ensemble_value, ensemble_value], color=base_color[color_i], linestyle='--', linewidth=2)

    model_fusion_color = "#3b3b3b"

    if dataset_name == "AVE":
        bottom_part = 0.90
        line_offset = [[4.5, 17.8], [20.2, 26.6]]
    elif dataset_name == "CREMA-D":
        bottom_part = 0.90
        line_offset = [[2.8, 12.4], [13.6, 20.5]]
    elif dataset_name == "UCF":
        bottom_part = 0.89
        line_offset = [[4.5, 17.8], [20.2, 26.6]]


    ref_to_the_plot = [0.095,0.127]
    diff = ref_to_the_plot[1]-ref_to_the_plot[0]
    # line1 = Line2D([ref_to_the_plot[0]+diff*line_offset[0][0], ref_to_the_plot[0]+diff*line_offset[0][1]], [bottom_part, bottom_part], transform=fig.transFigure, figure=fig, color=model_fusion_color, linewidth=4)
    # line2 = Line2D([ref_to_the_plot[0]+diff*line_offset[1][0], ref_to_the_plot[0]+diff*line_offset[1][1]], [bottom_part, bottom_part], transform=fig.transFigure, figure=fig, color=model_fusion_color, linewidth=4)
    # line3 = Line2D([ref_to_the_plot[0]+diff*21.55, ref_to_the_plot[0]+diff*26.6], [bottom_part, bottom_part], transform=fig.transFigure, figure=fig, color=model_fusion_color, linewidth=4)
    # fig.lines.extend([line1])


    if dataset_name == "AVE":
        text_bottom = 0.99
        off_set_text = [5.4, 12.5]
    elif dataset_name == "CREMA-D":
        text_bottom = 0.98
        off_set_text = [7.9, 18.7]
    elif dataset_name == "UCF":
        text_bottom = 1.045
        off_set_text = [5.4, 12.5]


    # text_bottom = 0.98
    # plt.text(off_set_text[0]*offset_model, text_bottom, 'Late-Linear', transform=plt.gca().get_xaxis_transform(), ha='center', va='top', fontsize=labels_font, color=model_fusion_color)
    # plt.text(off_set_text[1]*offset_model, text_bottom, 'Mid-MLP', transform=plt.gca().get_xaxis_transform(), ha='center', va='top', fontsize=labels_font, color=model_fusion_color)
    # plt.text(17.7*offset_model, text_bottom, 'Mid', transform=plt.gca().get_xaxis_transform(), ha='center', va='top', fontsize=16, color=model_fusion_color)
    #change color of text
    # if dataset_name == "UCF":
    plt.xlabel("Methods", fontsize=labels_font)


    # plt.text(24.6*offset_model, -0.05, 'Models', transform=plt.gca().get_xaxis_transform(), ha='center', va='top', fontsize=20)
    # if dataset_name == "AVE":
    #     plt.ylim(30,82.39)
    #     # plt.ylim(bottom=30)
    # elif dataset_name == "CREMA-D":
    #     plt.ylim(-20,20)
    #     # plt.ylim(bottom=41)
    #
    # elif dataset_name == "UCF":
    #     plt.ylim(25,60)
    #     plt.ylim(bottom=30)

    plt.ylim(min_data-3, max_data+3)
    for i in values:
        plt.vlines(x=i, ymin=min_data-3, ymax=max_data+3, color='gray', alpha=0.5, linewidth=0.5)
    plt.hlines(y=0, xmin=0, xmax=len(values)*offset_model, color='gray', alpha=0.5, linewidth=0.5)
    # plt.subplots_adjust()  # Adjust the bottom parameter as needed
    csfont = {'fontname': 'Comic Sans MS', "fontweight": "bold", "color":"purple"}
    plt.title(dataset_name, fontsize=labels_font, loc='left', y=0.94,  x=0.05, **csfont)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    if dataset_name == "CREMA-D":
        plt.legend(bbox_to_anchor=(0.95, 0), loc='lower right',  fontsize=tickfont).set_title("Backbone Model", prop={'size': tickfont})
    plt.tight_layout()  # Adjust the bottom parameter as needed

    plt.savefig("./Figures/ICLR_bar_results_{}.pdf".format(dataset_name), format='pdf')# Ensure legend is shown
    plt.show()
def plot_accuracies_bar(multi_fold_results, dataset_names):
    mean_acc = {}
    ste_acc = {}
    for dataset_name in dataset_names:
        mean_acc[dataset_name] = {}
        ste_acc[dataset_name] = {}

        for each_conf in multi_fold_results[dataset_name]:
            acc_per_fold = []
            for fold_idx in range(0,num_folds):
                acc_per_fold.append(multi_fold_results[dataset_name][each_conf][fold_idx]["acc"]["combined"]*100)
            #can you estimate the mean value and the standard error of the mean?
            mean_acc[dataset_name][each_conf] = round(np.mean(acc_per_fold), 2)
            ste_acc[dataset_name][each_conf] = np.std(acc_per_fold)/np.sqrt(num_folds)
    print(mean_acc)
    label_sequence = ['Uni_Video', 'Uni_Audio','Joint_Training', 'OGM', 'MLB', 'MCR']
    label_sequence = ['Uni_Video', 'Uni_Audio','Joint_Training', 'MLB', 'MCR']
    label_sequence = ['Joint_Training', 'MLB', 'MCR']

    fig = plt.figure(figsize=(15, 6))

    offset_model = 0.35
    offset_dataset = 0.3*offset_model
    # Plot individual points
    min_data = 100
    max_data = 0
    base_color = sns.color_palette("muted", n_colors=len(dataset_names))  # Use one tone for simplicity

    for di, dataset_name in enumerate(dataset_names):
        data = mean_acc[dataset_name]
        std_data = ste_acc[dataset_name]

        color_i = 0 if di==1 else 1
        # color_i = 1 if di==1 else 0
        if len(dataset_names)==1: color_i = 0

        # subtract Ensemble acc from the rest of the values in the dict
        this_data = {method: data[method] - data["Ensemble"] for method in data}
        del this_data["Ensemble"]
        del std_data["Ensemble"]
        this_data = [this_data[label] for label in label_sequence]
        this_std = [std_data[label] for label in label_sequence]

        min_data = min(min(this_data), min_data)
        max_data = max(max(this_data), max_data)

        plt.bar(np.arange(len(this_data))*offset_model+di*offset_dataset, this_data,  yerr=this_std, width=0.15, color=base_color[color_i], label=dataset_name, alpha=0.8,error_kw={'ecolor': base_color[color_i], 'alpha':0.5, 'capsize':5})

    values = np.arange(len(label_sequence))*offset_model
    if "CREMAD" in dataset_names[0] or "AVE" in dataset_names[0]:
        values = values+0.05

    tickfont = 18
    labels_font = 24

    plt.xticks(values, label_sequence, rotation=30, ha="right", fontsize=tickfont)
    plt.yticks(fontsize=tickfont)
    # plt.title("Accuracy Comparison Across Models on {} Dataset".format(dataset_name), fontsize=20)
    # plt.ylabel("Accuracy (%)", fontsize=labels_font)
    plt.ylabel("Accuracy Difference \n with Ensemble (%)", fontsize=labels_font)

    for di, dataset_name in enumerate(dataset_names):
        data = mean_acc[dataset_name]
        ensemble_value = data["Ensemble"]

        color_i = 0 if di == 1 else 1
        if len(dataset_names) == 1: color_i = 0
        plt.plot([0*offset_model+di*offset_dataset, len(values)*offset_model+di*offset_dataset], [ensemble_value, ensemble_value], color=base_color[color_i], linestyle='--', linewidth=2)

    model_fusion_color = "#3b3b3b"

    # if dataset_name == "AVE":
    #     bottom_part = 0.90
    #     line_offset = [[4.5, 17.8], [20.2, 26.6]]
    # elif dataset_name == "CREMA-D":
    #     bottom_part = 0.90
    #     line_offset = [[2.8, 12.4], [13.6, 20.5]]
    # elif dataset_name == "UCF":
    #     bottom_part = 0.89
    #     line_offset = [[4.5, 17.8], [20.2, 26.6]]


    ref_to_the_plot = [0.095,0.127]
    diff = ref_to_the_plot[1]-ref_to_the_plot[0]
    # line1 = Line2D([ref_to_the_plot[0]+diff*line_offset[0][0], ref_to_the_plot[0]+diff*line_offset[0][1]], [bottom_part, bottom_part], transform=fig.transFigure, figure=fig, color=model_fusion_color, linewidth=4)
    # line2 = Line2D([ref_to_the_plot[0]+diff*line_offset[1][0], ref_to_the_plot[0]+diff*line_offset[1][1]], [bottom_part, bottom_part], transform=fig.transFigure, figure=fig, color=model_fusion_color, linewidth=4)
    # line3 = Line2D([ref_to_the_plot[0]+diff*21.55, ref_to_the_plot[0]+diff*26.6], [bottom_part, bottom_part], transform=fig.transFigure, figure=fig, color=model_fusion_color, linewidth=4)
    # fig.lines.extend([line1])


    # text_bottom = 0.98
    # plt.text(off_set_text[0]*offset_model, text_bottom, 'Late-Linear', transform=plt.gca().get_xaxis_transform(), ha='center', va='top', fontsize=labels_font, color=model_fusion_color)
    # plt.text(off_set_text[1]*offset_model, text_bottom, 'Mid-MLP', transform=plt.gca().get_xaxis_transform(), ha='center', va='top', fontsize=labels_font, color=model_fusion_color)
    # plt.text(17.7*offset_model, text_bottom, 'Mid', transform=plt.gca().get_xaxis_transform(), ha='center', va='top', fontsize=16, color=model_fusion_color)
    #change color of text
    # if dataset_name == "UCF":
    plt.xlabel("Methods", fontsize=labels_font)


    # plt.text(24.6*offset_model, -0.05, 'Models', transform=plt.gca().get_xaxis_transform(), ha='center', va='top', fontsize=20)
    # if dataset_name == "AVE":
    #     plt.ylim(30,82.39)
    #     # plt.ylim(bottom=30)
    # elif dataset_name == "CREMA-D":
    #     plt.ylim(-20,20)
    #     # plt.ylim(bottom=41)
    #
    # elif dataset_name == "UCF":
    #     plt.ylim(25,60)
    #     plt.ylim(bottom=30)

    plt.ylim(min_data-3, max_data+3)
    for i in values:
        plt.vlines(x=i, ymin=min_data-3, ymax=max_data+3, color='gray', alpha=0.5, linewidth=0.5)
    plt.hlines(y=0, xmin=0, xmax=len(values)*offset_model, color='gray', alpha=0.5, linewidth=0.5)
    # plt.subplots_adjust()  # Adjust the bottom parameter as needed
    csfont = {'fontname': 'Comic Sans MS', "fontweight": "bold", "color":"purple"}
    plt.title(dataset_name, fontsize=labels_font, loc='left', y=0.94,  x=0.05, **csfont)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    if "CREMA-D" in dataset_name:
        plt.legend(bbox_to_anchor=(0.95, 0), loc='lower right',  fontsize=tickfont).set_title("Backbone Model", prop={'size': tickfont})
    plt.tight_layout()  # Adjust the bottom parameter as needed

    plt.savefig("./Figures/ICLR_bar_results_{}.pdf".format(dataset_name), format='pdf')# Ensure legend is shown
    plt.show()

def plot_confusion_matrix(conf_matrices, exp_name):
    global_max = max([conf_matrices[cm].max() for cm in conf_matrices if conf_matrices[cm].mean()!=0])
    label_fontsize = 26
    xtick_fontsize = 22
    num_in_box_fontsize = 22
    title_fontsize = 26

    fig, axs = plt.subplots(1, len(conf_matrices), figsize=(32, 7), sharey=False)

    cmap = 'Blues'
    for ax, (config_name, cm) in zip(axs, conf_matrices.items()):
        # Use imshow for plotting with a consistent color scale across all matrices

        cm = np.round(cm.T,1)
        # new_cm = np.zeros((2,2))
        # new_cm[:,0] = cm[:,0]
        # new_cm[:,1] = cm[:,1:].sum(axis=1)
        # cm = new_cm

        cax = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=global_max, aspect='auto')
        ax.set_title(config_name, fontsize=title_fontsize)
        # ax.set_xlabel("Video-Audio", fontsize=label_fontsize)
        if ax == axs[0]:
            ax.set_ylabel("Multimodal Model", fontsize=label_fontsize)
        if ax == axs[2]:
            ax.set_xlabel("Audio-Video Unimodal Models", fontsize=label_fontsize)

        # Set tick labels manually
        ax.set_xticks(np.arange(4))
        # ax.set_xticklabels(['Both Wrong', 'At Least One \n Correct'], fontsize=xtick_fontsize, rotation=20)
        ax.set_xticklabels(['Both \n Wrong', 'Audio \n Correct', 'Video \n Correct', 'Both \n Correct'], fontsize=xtick_fontsize-4, rotation=0)
        # ax.set_yticks(np.arange(len(["False", "True"])), ["False", "True"], fontsize=xtick_fontsize)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Wrong', 'Correct'], fontsize=xtick_fontsize)

        # Annotate the heatmap with text
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'black' if cm[i, j] < 25 else 'white'
                text = ax.text(j, i, round(cm[i, j],2), ha="center", va="center", color=color, fontsize=num_in_box_fontsize)

    # plt.subplots_adjust(wspace=0.3, bottom=0.29)
    plt.subplots_adjust(
        left=0.06,  # Left margin
        right=0.97,  # Right margin
        top=0.85,  # Top margin
        bottom=0.21,  # Bottom margin
        wspace=0.3  # Space between subplots
    )

    cbar = fig.colorbar(cax, ax=axs, orientation='vertical', fraction=0.025, pad=0.04).set_label(label='% of samples',size=xtick_fontsize)
    fig.figure.axes[-1].tick_params(axis="y", labelsize=xtick_fontsize)
    # cbar.ax.tick_params(labelsize=xtick_fontsize)  # Change the font size of the colorbar's tick label
    # cbar.ax.tick_params(labelsize='large')# s
    # plt.subplots_adjust(top=0.05)
    csfont = {"fontweight": "bold", "color":"purple"}
    plt.title(exp_name.split("_")[0], fontsize=30, loc='left', y=1.12,  x=-5.6, **csfont)
    #tight layout without space on the left or right
    # plt.tight_layout(rect=[0.05, 0.05, 1.2, 0.95])
    plt.savefig("./Figures/Rebuttal_Confusion_Matrix_Combined_{}.png".format(exp_name))
    # plt.savefig("./Figures/MCR_Benefit_Confusion_Matrix_Combined_{}.pdf".format(exp_name), format='pdf', bbox_inches='tight')

    plt.show()
def plot_confusion_matrix_mosei(conf_matrices, exp_name):
    global_max = max([conf_matrices[cm].max() for cm in conf_matrices if conf_matrices[cm].mean()!=0])
    label_fontsize = 26
    xtick_fontsize = 22
    num_in_box_fontsize = 22
    title_fontsize = 26

    fig, axs = plt.subplots(1, len(conf_matrices), figsize=(32, 7), sharey=False)

    cmap = 'Blues'
    for ax, (config_name, cm) in zip(axs, conf_matrices.items()):
        # Use imshow for plotting with a consistent color scale across all matrices

        cm = np.round(cm.T,1)
        # new_cm = np.zeros((2,2))
        # new_cm[:,0] = cm[:,0]
        # new_cm[:,1] = cm[:,1:].sum(axis=1)
        # cm = new_cm

        cax = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=global_max, aspect='auto')
        ax.set_title(config_name, fontsize=title_fontsize)
        # ax.set_xlabel("Video-Audio", fontsize=label_fontsize)
        if ax == axs[0]:
            ax.set_ylabel("Multimodal Model", fontsize=label_fontsize)
        if ax == axs[2]:
            ax.set_xlabel("Video-Text Unimodal Models", fontsize=label_fontsize)

        # Set tick labels manually
        ax.set_xticks(np.arange(4))
        # ax.set_xticklabels(['Both Wrong', 'At Least One \n Correct'], fontsize=xtick_fontsize, rotation=20)
        ax.set_xticklabels(['Both \n Wrong', 'Video \n Correct', 'Text \n Correct', 'Both \n Correct'], fontsize=xtick_fontsize-4, rotation=0)
        # ax.set_yticks(np.arange(len(["False", "True"])), ["False", "True"], fontsize=xtick_fontsize)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Wrong', 'Correct'], fontsize=xtick_fontsize)

        # Annotate the heatmap with text
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'black' if cm[i, j] < 25 else 'white'
                text = ax.text(j, i, round(cm[i, j],2), ha="center", va="center", color=color, fontsize=num_in_box_fontsize)

    # plt.subplots_adjust(wspace=0.3, bottom=0.29)
    plt.subplots_adjust(
        left=0.06,  # Left margin
        right=0.97,  # Right margin
        top=0.8,  # Top margin
        bottom=0.25,  # Bottom margin
        wspace=0.3  # Space between subplots
    )

    cbar = fig.colorbar(cax, ax=axs, orientation='vertical', fraction=0.025, pad=0.04).set_label(label='% of samples',size=xtick_fontsize)
    fig.figure.axes[-1].tick_params(axis="y", labelsize=xtick_fontsize)
    # cbar.ax.tick_params(labelsize=xtick_fontsize)  # Change the font size of the colorbar's tick label
    # cbar.ax.tick_params(labelsize='large')# s
    # plt.subplots_adjust(top=0.05)
    csfont = {"fontweight": "bold", "color":"purple"}
    plt.title(exp_name.split("_")[0], fontsize=30, loc='left', y=1.12,  x=-5.6, **csfont)
    #tight layout without space on the left or right
    # plt.tight_layout(rect=[0.05, 0.05, 1.2, 0.95])
    plt.savefig("./Figures/Rebuttal_Confusion_Matrix_Combined_{}.png".format(exp_name))
    # plt.savefig("./Figures/MCR_Benefit_Confusion_Matrix_Combined_{}.pdf".format(exp_name), format='pdf', bbox_inches='tight')

    plt.show()
def plot_confusion_matrix_4d(conf_matrices, exp_name):
    global_max = max([conf_matrices[cm].max() for cm in conf_matrices if conf_matrices[cm].mean()!=0])
    label_fontsize = 26
    xtick_fontsize = 22
    num_in_box_fontsize = 22
    title_fontsize = 26

    fig, axs = plt.subplots(1, len(conf_matrices), figsize=(32, 7), sharey=False)

    cmap = 'Blues'
    for ax, (config_name, cm) in zip(axs, conf_matrices.items()):
        # Use imshow for plotting with a consistent color scale across all matrices

        cm = np.round(cm.T,1)
        # new_cm = np.zeros((2,2))
        # new_cm[:,0] = cm[:,0]
        # new_cm[:,1] = cm[:,1:].sum(axis=1)
        # cm = new_cm

        cax = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=global_max, aspect='auto')
        ax.set_title(config_name, fontsize=title_fontsize)
        # ax.set_xlabel("Video-Audio", fontsize=label_fontsize)
        if ax == axs[0]:
            ax.set_ylabel("Multimodal Model", fontsize=label_fontsize)
        if ax == axs[2]:
            ax.set_xlabel("Video-Text-Audio Unimodal Models", fontsize=label_fontsize)

        # Set tick labels manually
        # ax.set_xticks(np.arange(2))
        ax.set_xticks(np.arange(5))
        # ax.set_xticklabels(['Both Wrong', 'At Least One \n Correct'], fontsize=xtick_fontsize, rotation=20)
        ax.set_xticklabels(['Both \n Wrong', 'Video \n Correct', 'Text \n Correct', 'Audio \n Correct', 'At least \n two \n Correct'],
                           fontsize=xtick_fontsize - 6, rotation=0)

        # ax.set_xticklabels(['All Wrong', 'At Least One \n Correct'], fontsize=xtick_fontsize, rotation=20)
        # ax.set_yticks(np.arange(len(["False", "True"])), ["False", "True"], fontsize=xtick_fontsize)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Wrong', 'Correct'], fontsize=xtick_fontsize)

        # Annotate the heatmap with text
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'black' if cm[i, j] < 25 else 'white'
                text = ax.text(j, i, round(cm[i, j],2), ha="center", va="center", color=color, fontsize=num_in_box_fontsize)

    # plt.subplots_adjust(wspace=0.3, bottom=0.29)
    plt.subplots_adjust(
        left=0.06,  # Left margin
        right=0.97,  # Right margin
        top=0.8,  # Top margin
        bottom=0.2,  # Bottom margin
        wspace=0.3  # Space between subplots
    )

    cbar = fig.colorbar(cax, ax=axs, orientation='vertical', fraction=0.025, pad=0.04).set_label(label='% of samples',size=xtick_fontsize)
    fig.figure.axes[-1].tick_params(axis="y", labelsize=xtick_fontsize)
    # cbar.ax.tick_params(labelsize=xtick_fontsize)  # Change the font size of the colorbar's tick label
    # cbar.ax.tick_params(labelsize='large')# s
    # plt.subplots_adjust(top=0.05)
    csfont = {"fontweight": "bold", "color":"purple"}
    plt.title(exp_name.split("_")[0], fontsize=30, loc='left', y=1.17,  x=-5.6, **csfont)
    #tight layout without space on the left or right
    # plt.tight_layout(rect=[0.05, 0.05, 1.2, 0.95])
    plt.savefig("./Figures/Rebuttal__Confusion_Matrix_Combined_{}.png".format(exp_name))
    # plt.savefig("./Figures/MCR_Benefit_Confusion_Matrix_Combined_{}.pdf".format(exp_name), format='pdf', bbox_inches='tight')

    plt.show()
def plot_one_example_confusion_matrix(conf_matrices, exp_name):
    global_max = max([conf_matrices[cm].max() for cm in conf_matrices])
    label_fontsize = 26
    xtick_fontsize = 22
    num_in_box_fontsize = 22
    title_fontsize = 26

    fig, axs = plt.subplots(1, 1, figsize=(13, 7), sharey=False)

    cmap = 'Blues'
    ax = axs
    config_name = ""
    cm = conf_matrices["Ensemble"]
        # Use imshow for plotting with a consistent color scale across all matrices

    cm = np.round(cm.T,1)
    new_cm = np.zeros((2,2))
    new_cm[:,0] = cm[:,0]
    new_cm[:,1] = cm[:,1:].sum(axis=1)
    cm = new_cm

    cax = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=global_max, aspect='auto')
    ax.set_title(config_name, fontsize=title_fontsize)
    # ax.set_xlabel("Video-Audio", fontsize=label_fontsize)
    # if ax == axs[0]:
    ax.set_ylabel("Multimodal", fontsize=label_fontsize)
    # if ax == axs[2]:
    ax.set_xlabel("Video-Audio", fontsize=label_fontsize)

    # Set tick labels manually
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(['Both Wrong', 'At Least One \n Correct'], fontsize=xtick_fontsize, rotation=20)
    # ax.set_yticks(np.arange(len(["False", "True"])), ["False", "True"], fontsize=xtick_fontsize)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Wrong', 'Correct'], fontsize=xtick_fontsize)

    message = [["Multimodal Wrong \n Video and Audio Wrong",
                "Multimodal Wrong \n Video or Audio Correct"],
               ["Multimodal Correct \n Video and Audio Wrong",
                "Multimodal Correct \n Video or Audio Correct"]]

    # message = [["WWW",
    #             "CWW"],
    #            ["WCW or \n WWC or \n WCC    ",
    #             "CCW or \n CWC or \n CCC    "]]


    # Annotate the heatmap with text
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'black' if cm[i, j] < 25 else 'white'
            print(i,j, message[i][j])
            text = ax.text(j, i, message[i][j], ha="center", va="center", color=color, fontsize=num_in_box_fontsize)
    text = ax.text(0, -0.65, "Synergy", ha="center", va="center", fontsize=30)
    text = ax.text(1, -0.65, "Routing", ha="center", va="center", fontsize=30)

    plt.subplots_adjust(wspace=0.37, bottom=0.29)
    # cbar = fig.colorbar(cax, ax=axs, orientation='vertical', fraction=0.025, pad=0.04).set_label(label='% of samples',size=xtick_fontsize)
    fig.figure.axes[-1].tick_params(axis="y", labelsize=xtick_fontsize)
    # cbar.ax.tick_params(labelsize=xtick_fontsize)  # Change the font size of the colorbar's tick label
    # cbar.ax.tick_params(labelsize='large')# s
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.1)
    csfont = {"fontweight": "bold", "color":"purple"}
    plt.title(exp_name.split("_")[0], fontsize=30, loc='left', y=1.07,  x=-6.4, **csfont)

    plt.savefig("./Figures/Thesis_Confusion_Matrix_Combined_{}.png".format(exp_name))
    # plt.savefig("./Figures/Thesis_Benefit_example.pdf".format(exp_name), format='pdf')

    plt.show()

def plot_3d_confusion_matrix(conf_matrices, exp_name):

    global_max = max([conf_matrices[cm].max() for cm in conf_matrices])
    label_fontsize = 22
    xtick_fontsize = 16
    num_in_box_fontsize = 17
    title_fontsize = 22

    fig, axs = plt.subplots(1, len(conf_matrices), figsize=(26, 7), sharey=False)

    cmap = 'Blues'
    for ax, (config_name, cm) in zip(axs, conf_matrices.items()):
        # Use imshow for plotting with a consistent color scale across all matrices
        cm = np.round(cm.T,1)
        cax = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=global_max, aspect='auto')
        ax.set_title(config_name, fontsize=title_fontsize)
        ax.set_xlabel("Video-Audio", fontsize=label_fontsize)
        if ax == axs[0]:
            ax.set_ylabel("Multimodal", fontsize=label_fontsize)

        # Set tick labels manually
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(['Both Wrong', 'Audio Correct', 'Video Correct', 'Both Correct'], fontsize=xtick_fontsize, rotation=20)
        # ax.set_yticks(np.arange(len(["False", "True"])), ["False", "True"], fontsize=xtick_fontsize)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Wrong', 'Correct'], fontsize=xtick_fontsize)

        # Annotate the heatmap with text
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'black' if cm[i, j] < 25 else 'white'
                text = ax.text(j, i, round(cm[i, j],2), ha="center", va="center", color=color, fontsize=num_in_box_fontsize)

    plt.subplots_adjust(wspace=0.27, bottom=0.25)
    cbar = fig.colorbar(cax, ax=axs, orientation='vertical', fraction=0.025, pad=0.04).set_label(label='% of samples',size=xtick_fontsize)
    fig.figure.axes[-1].tick_params(axis="y", labelsize=xtick_fontsize)
    # cbar.ax.tick_params(labelsize=xtick_fontsize)  # Change the font size of the colorbar's tick label
    # cbar.ax.tick_params(labelsize='large')# s
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.1)
    plt.savefig("./Figures/Neurips_Confusion_Matrix_Combined_{}.png".format(exp_name))
    plt.savefig("./Figures/Neurips_Confusion_Matrix_Combined_{}.pdf".format(exp_name), format='pdf')

    plt.show()

def confusion_matrix_3d_aggr(multi_fold_results, num_folds, exp_name=""):
    def create_conf(predictions):
        predictions = np.array(predictions)
        all_false = np.all(predictions[:2] == 0, axis=0)
        only_mod0_true = (predictions[0] == 1) & (predictions[1] == 0)
        only_mod1_true = (predictions[1] == 1) & (predictions[0] == 0)
        both_mods_true = (predictions[1] == 1) & (predictions[0] == 1)
        mmodel_true = predictions[2] == 1


        cm = np.array([
            [ (~mmodel_true[all_false]).sum(),      mmodel_true[all_false].sum()],
            [ (~mmodel_true[only_mod0_true]).sum(), mmodel_true[only_mod0_true].sum()],
            [ (~mmodel_true[only_mod1_true]).sum(), mmodel_true[only_mod1_true].sum()],
            [ (~mmodel_true[both_mods_true]).sum(), mmodel_true[both_mods_true].sum()],
        ])
        mmodel_true[both_mods_true].sum()
        cm = 100 * cm.astype('float') /cm.sum() # Normalize by row
        return cm

    def average_conf_matrices(conf_matrices):
        # for i in conf_matrices:
        #     print(round(i.sum(axis=1)[0],2))
        avg_conf = np.mean(conf_matrices, axis=0)
        message = "Acc {}".format(round(avg_conf.sum(axis=0)[1], 2))
        # for i in conf_matrices:
        #     message += "-{}".format(round(i.sum(axis=0)[1], 2))
        # print(message)
        return avg_conf

    def print_conf_accuracy(conf_matrices, multi_fold_results):

        mean_acc = {}
        for each_conf in multi_fold_results:
            acc_per_fold = []
            for fold_idx in range(0, num_folds):
                acc_per_fold.append(multi_fold_results[each_conf][fold_idx]["acc"]["combined"] * 100)
            # can you estimate the mean value and the standard error of the mean?
            mean_acc[each_conf] = np.mean(acc_per_fold)
            print(each_conf, acc_per_fold)
        print(mean_acc)

        message = "Acc "
        for i in conf_matrices:
            message += "{}-{} ".format(i,round(conf_matrices[i].sum(axis=0)[1], 2))
        print(message)

    # Assuming we have 3 runs for each model configuration
    from collections import defaultdict

    conf_matrices = defaultdict(lambda:np.array([[0,0],[0,0]]))
    for each_conf in multi_fold_results:
        if each_conf == "Uni_Audio" or each_conf == "Uni_Video": continue
        # print(each_conf)
        conf_matrices_per_fold = []

        for fold_idx in range(0,num_folds):
            predictions = [
                multi_fold_results['Uni_Audio'][fold_idx]["total_preds"]["combined"].argmax(-1) ==
                multi_fold_results['Uni_Audio'][fold_idx]["total_preds_target"],
                multi_fold_results['Uni_Video'][fold_idx]["total_preds"]["combined"].argmax(-1) ==
                multi_fold_results['Uni_Video'][fold_idx]["total_preds_target"],

            ]
            predictions.append(
                multi_fold_results[each_conf][fold_idx]["total_preds"]["combined"].argmax(-1) ==
                multi_fold_results[each_conf][fold_idx]["total_preds_target"],
            )
            # for i in predictions:
            #     print(i.shape)
            cm = create_conf(predictions)
            conf_matrices_per_fold.append(cm)
        conf_matrices[each_conf] = average_conf_matrices(conf_matrices_per_fold)

    print_conf_accuracy(conf_matrices, multi_fold_results)
    #make default dict with conf matrices 0s

    conf_matrices = {
                    "MCR": conf_matrices["MCR"],
                    "Ensemble": conf_matrices["Ensemble"],
                    "Joint Training": conf_matrices["Joint_Training"],
                     "AGM": conf_matrices["AGM"],
                     "MLB": conf_matrices["MLB"]
                     # "Uni-Pre Finetuned": conf_matrices["Uni-Pre Finetuned"]}
                     }

    keys = "_".join(conf_matrices.keys())

    plot_confusion_matrix(conf_matrices, exp_name.format(keys))
    # plot_one_example_confusion_matrix(conf_matrices, exp_name.format(keys))
    # plot_3d_confusion_matrix(conf_matrices, exp_name)
def confusion_matrix_3d_sthsth_aggr(multi_fold_results, num_folds, exp_name=""):
    def create_conf(predictions):
        predictions = np.array(predictions)
        all_false = np.all(predictions[:2] == 0, axis=0)
        only_mod0_true = (predictions[0] == 1) & (predictions[1] == 0)
        only_mod1_true = (predictions[1] == 1) & (predictions[0] == 0)
        both_mods_true = (predictions[1] == 1) & (predictions[0] == 1)
        mmodel_true = predictions[2] == 1


        cm = np.array([
            [ (~mmodel_true[all_false]).sum(),      mmodel_true[all_false].sum()],
            [ (~mmodel_true[only_mod0_true]).sum(), mmodel_true[only_mod0_true].sum()],
            [ (~mmodel_true[only_mod1_true]).sum(), mmodel_true[only_mod1_true].sum()],
            [ (~mmodel_true[both_mods_true]).sum(), mmodel_true[both_mods_true].sum()],
        ])
        mmodel_true[both_mods_true].sum()
        cm = 100 * cm.astype('float') /cm.sum() # Normalize by row
        return cm

    def average_conf_matrices(conf_matrices):
        # for i in conf_matrices:
        #     print(round(i.sum(axis=1)[0],2))
        avg_conf = np.mean(conf_matrices, axis=0)
        message = "Acc {}".format(round(avg_conf.sum(axis=0)[1], 2))
        # for i in conf_matrices:
        #     message += "-{}".format(round(i.sum(axis=0)[1], 2))
        # print(message)
        return avg_conf

    def print_conf_accuracy(conf_matrices, multi_fold_results):

        mean_acc = {}
        for each_conf in multi_fold_results:
            mean_acc[each_conf] = multi_fold_results[each_conf][fold_idx]["acc"]["combined"] * 100
        print(mean_acc)

        message = "Acc "
        for i in conf_matrices:
            message += "{}-{} ".format(i,round(conf_matrices[i].sum(axis=0)[1], 2))
        print(message)

    # Assuming we have 3 runs for each model configuration

    conf_matrices = defaultdict(lambda:np.array([[0,0],[0,0]]))
    for each_conf in multi_fold_results:
        if each_conf == "Uni_Audio" or each_conf == "Uni_Video": continue
        print(each_conf)
        conf_matrices_per_fold = []
        fold_idx=None
        predictions = [
            multi_fold_results['Uni_Audio'][fold_idx]["total_preds"]["combined"].argmax(-1) ==
            multi_fold_results['Uni_Audio'][fold_idx]["total_preds_target"],
            multi_fold_results['Uni_Video'][fold_idx]["total_preds"]["combined"].argmax(-1) ==
            multi_fold_results['Uni_Video'][fold_idx]["total_preds_target"],

        ]
        predictions.append(
            multi_fold_results[each_conf][fold_idx]["total_preds"]["combined"].argmax(-1) ==
            multi_fold_results[each_conf][fold_idx]["total_preds_target"],
        )
        for i in predictions:
            print(i.shape)
        conf_matrices[each_conf] = create_conf(predictions)


    print_conf_accuracy(conf_matrices, multi_fold_results)
    #make default dict with conf matrices 0s

    conf_matrices = {
                    "MCR": conf_matrices["MCR"],
                    "Ensemble": conf_matrices["Ensemble"],
                     "Joint Training": conf_matrices["Joint_Training"],
                     "AGM": conf_matrices["AGM"],
                     "MLB": conf_matrices["MLB"],
                     # "Uni-Pre Finetuned": conf_matrices["Uni-Pre Finetuned"]}
                     }

    keys = "_".join(conf_matrices.keys())

    plot_confusion_matrix(conf_matrices, exp_name.format(keys))
    # plot_one_example_confusion_matrix(conf_matrices, exp_name.format(keys))
    # plot_3d_confusion_matrix(conf_matrices, exp_name)

def confusion_matrix_3d_mosei_aggr(multi_fold_results, num_folds, exp_name=""):
    def create_conf(predictions):
        predictions = np.array(predictions)
        all_false = np.all(predictions[:2] == 0, axis=0)
        only_mod0_true = (predictions[0] == 1) & (predictions[1] == 0)
        only_mod1_true = (predictions[1] == 1) & (predictions[0] == 0)
        both_mods_true = (predictions[1] == 1) & (predictions[0] == 1)
        mmodel_true = predictions[2] == 1


        cm = np.array([
            [ (~mmodel_true[all_false]).sum(),      mmodel_true[all_false].sum()],
            [ (~mmodel_true[only_mod0_true]).sum(), mmodel_true[only_mod0_true].sum()],
            [ (~mmodel_true[only_mod1_true]).sum(), mmodel_true[only_mod1_true].sum()],
            [ (~mmodel_true[both_mods_true]).sum(), mmodel_true[both_mods_true].sum()],
        ])
        mmodel_true[both_mods_true].sum()
        cm = 100 * cm.astype('float') /cm.sum() # Normalize by row
        return cm

    def average_conf_matrices(conf_matrices):
        # for i in conf_matrices:
        #     print(round(i.sum(axis=1)[0],2))
        avg_conf = np.mean(conf_matrices, axis=0)
        message = "Acc {}".format(round(avg_conf.sum(axis=0)[1], 2))
        # for i in conf_matrices:
        #     message += "-{}".format(round(i.sum(axis=0)[1], 2))
        # print(message)
        return avg_conf

    def print_conf_accuracy(conf_matrices, multi_fold_results):

        mean_acc = {}
        for each_conf in multi_fold_results:
            mean_acc[each_conf] = multi_fold_results[each_conf][fold_idx]["acc"]["combined"] * 100
        print(mean_acc)

        message = "Acc "
        for i in conf_matrices:
            message += "{}-{} ".format(i,round(conf_matrices[i].sum(axis=0)[1], 2))
        print(message)

    # Assuming we have 3 runs for each model configuration
    from collections import defaultdict

    conf_matrices = defaultdict(lambda:np.array([[0,0],[0,0]]))
    for each_conf in multi_fold_results:
        if each_conf == "Uni_Video" or each_conf == "Uni_Text" : continue
        print(each_conf)
        conf_matrices_per_fold = []
        fold_idx = None

        print(multi_fold_results["Uni_Video"].keys())
        # centers = np.unique(multi_fold_results["Uni_Video"][fold_idx]["total_preds_target"])
        # rounded_values_video = [min(centers, key=lambda center: abs(center - value)) for value in multi_fold_results['Uni_Video'][fold_idx]["total_preds"]]
        # rounded_values_text = [min(centers, key=lambda center: abs(center - value)) for value in multi_fold_results['Uni_Text'][fold_idx]["total_preds"]]
        # #keep
        targets = multi_fold_results["Uni_Video"][fold_idx]["total_preds_target"]
        rounded_values_video = (multi_fold_results['Uni_Video'][fold_idx]["total_preds"][targets!=0]> 0)
        rounded_values_text = (multi_fold_results['Uni_Text'][fold_idx]["total_preds"][targets!=0]> 0)
        binary_targets = (targets[targets!=0]> 0)

        predictions = [
            rounded_values_video == binary_targets,
            rounded_values_text == binary_targets,
        ]
        rounded_values = (multi_fold_results[each_conf][fold_idx]["total_preds"][targets!=0]> 0)
        # rounded_values = [min(centers, key=lambda center: abs(center - value)) for value in multi_fold_results[each_conf][fold_idx]["total_preds"]]
        predictions.append(
            rounded_values == binary_targets,
        )
        for i in predictions:
            print(i.shape)
        cm = create_conf(predictions)
        conf_matrices_per_fold.append(cm)
        conf_matrices[each_conf] = average_conf_matrices(conf_matrices_per_fold)


    print_conf_accuracy(conf_matrices, multi_fold_results)
    #make default dict with conf matrices 0s

    conf_matrices = {
                    "MCR": conf_matrices["MCR"],
                    "Ensemble": conf_matrices["Ensemble"],
                     "Joint Training": conf_matrices["Joint_Training"],
                     "AGM": conf_matrices["AGM"],
                     "MLB": conf_matrices["MLB"]
                     # "Uni-Pre Finetuned": conf_matrices["Uni-Pre Finetuned"]}
                     }

    keys = "_".join(conf_matrices.keys())

    # plot_confusion_matrix(conf_matrices, exp_name.format(keys))
    plot_confusion_matrix_mosei(conf_matrices, exp_name.format(keys))
    # plot_one_example_confusion_matrix(conf_matrices, exp_name.format(keys))
    # plot_3d_confusion_matrix(conf_matrices, exp_name)
def confusion_matrix_4d_mosei_aggr(multi_fold_results, num_folds, exp_name=""):
    def create_conf(predictions):
        predictions = np.array(predictions)
        all_false = np.all(predictions[:2] == 0, axis=0)
        only_mod0_true = (predictions[0] == 1) & (predictions[1] == 0) & (predictions[2] == 0)
        only_mod1_true = (predictions[1] == 1) & (predictions[0] == 0) & (predictions[2] == 0)
        only_mod2_true = (predictions[2] == 1) & (predictions[0] == 0) & (predictions[1] == 0)
        both_mods_true = ((predictions[1] == 1) & (predictions[0] == 1) & (predictions[2] == 1)) | ((predictions[1] == 0) & (predictions[0] == 1) & (predictions[2] == 1)) | ((predictions[1] == 1) & (predictions[0] == 0) & (predictions[2] == 1)) | ((predictions[1] == 1) & (predictions[0] == 1) & (predictions[2] == 0))
        mmodel_true = predictions[3] == 1


        cm = np.array([
            [ (~mmodel_true[all_false]).sum(),      mmodel_true[all_false].sum()],
            [ (~mmodel_true[only_mod0_true]).sum(), mmodel_true[only_mod0_true].sum()],
            [ (~mmodel_true[only_mod1_true]).sum(), mmodel_true[only_mod1_true].sum()],
            [ (~mmodel_true[only_mod2_true]).sum(), mmodel_true[only_mod2_true].sum()],
            [ (~mmodel_true[both_mods_true]).sum(), mmodel_true[both_mods_true].sum()],
        ])
        mmodel_true[both_mods_true].sum()
        cm = 100 * cm.astype('float') /cm.sum() # Normalize by row
        return cm

    def average_conf_matrices(conf_matrices):
        # for i in conf_matrices:
        #     print(round(i.sum(axis=1)[0],2))
        avg_conf = np.mean(conf_matrices, axis=0)
        message = "Acc {}".format(round(avg_conf.sum(axis=0)[1], 2))
        # for i in conf_matrices:
        #     message += "-{}".format(round(i.sum(axis=0)[1], 2))
        # print(message)
        return avg_conf

    def print_conf_accuracy(conf_matrices, multi_fold_results):

        mean_acc = {}
        for each_conf in multi_fold_results:
            mean_acc[each_conf] = multi_fold_results[each_conf][fold_idx]["acc"]["combined"] * 100
        print(mean_acc)

        message = "Acc "
        for i in conf_matrices:
            message += "{}-{} ".format(i,round(conf_matrices[i].sum(axis=0)[1], 2))
        print(message)

    # Assuming we have 3 runs for each model configuration
    from collections import defaultdict

    conf_matrices = defaultdict(lambda:np.array([[0,0],[0,0]]))
    for each_conf in multi_fold_results:
        if each_conf == "Uni_Video" or each_conf == "Uni_Text" : continue
        print(each_conf)
        conf_matrices_per_fold = []
        fold_idx = None
        # centers = np.unique(multi_fold_results["Uni_Video"][fold_idx]["total_preds_target"])
        # rounded_values_video = [min(centers, key=lambda center: abs(center - value)) for value in multi_fold_results['Uni_Video'][fold_idx]["total_preds"]]
        # rounded_values_text = [min(centers, key=lambda center: abs(center - value)) for value in multi_fold_results['Uni_Text'][fold_idx]["total_preds"]]
        # #keep
        targets = multi_fold_results["Uni_Video"][fold_idx]["total_preds_target"]
        rounded_values_video = (multi_fold_results['Uni_Video'][fold_idx]["total_preds"][targets!=0]> 0)
        rounded_values_audio = (multi_fold_results['Uni_Audio'][fold_idx]["total_preds"][targets!=0]> 0)
        rounded_values_text = (multi_fold_results['Uni_Text'][fold_idx]["total_preds"][targets!=0]> 0)
        binary_targets = (targets[targets!=0]> 0)

        predictions = [
            rounded_values_video == binary_targets,
            rounded_values_text == binary_targets,
            rounded_values_audio == binary_targets,
        ]
        rounded_values = (multi_fold_results[each_conf][fold_idx]["total_preds"][targets!=0]> 0)
        # rounded_values = [min(centers, key=lambda center: abs(center - value)) for value in multi_fold_results[each_conf][fold_idx]["total_preds"]]
        predictions.append(
            rounded_values == binary_targets,
        )
        for i in predictions:
            print(i.shape)
        cm = create_conf(predictions)
        conf_matrices_per_fold.append(cm)
        conf_matrices[each_conf] = average_conf_matrices(conf_matrices_per_fold)


    # print_conf_accuracy(conf_matrices, multi_fold_results)
    #make default dict with conf matrices 0s

    conf_matrices = {
                    "MCR": conf_matrices["MCR"],
                    "Ensemble": conf_matrices["Ensemble"],
                     "Joint Training": conf_matrices["Joint_Training"],
                     "AGM": conf_matrices["AGM"],
                     "MLB": conf_matrices["MLB"]
                     # "Uni-Pre Finetuned": conf_matrices["Uni-Pre Finetuned"]}
                     }

    keys = "_".join(conf_matrices.keys())

    # plot_confusion_matrix(conf_matrices, exp_name.format(keys))
    plot_confusion_matrix_4d(conf_matrices, exp_name.format(keys))
    # plot_one_example_confusion_matrix(conf_matrices, exp_name.format(keys))
    # plot_3d_confusion_matrix(conf_matrices, exp_name)
# def confusion_matrix_4d_mosei_aggr(multi_fold_results, num_folds, exp_name=""):
#     def create_conf(predictions):
#         predictions = np.array(predictions)
#         all_false = np.all(predictions[:3] == 0, axis=0)
#         atleastone_mod_true = np.any(predictions[:3] == 1, axis=0)
#         mmodel_true = predictions[3] == 1
#
#         cm = np.array([
#             [ (~mmodel_true[all_false]).sum(),      mmodel_true[all_false].sum()],
#             [ (~mmodel_true[atleastone_mod_true]).sum(), mmodel_true[atleastone_mod_true].sum()],
#         ])
#         mmodel_true[atleastone_mod_true].sum()
#         cm = 100 * cm.astype('float') /cm.sum() # Normalize by row
#         return cm
#
#     def average_conf_matrices(conf_matrices):
#         avg_conf = np.mean(conf_matrices, axis=0)
#         return avg_conf
#
#     def print_conf_accuracy(conf_matrices, multi_fold_results):
#
#         mean_acc = {}
#         for each_conf in multi_fold_results:
#             mean_acc[each_conf] = multi_fold_results[each_conf][fold_idx]["acc"]["combined"] * 100
#         print(mean_acc)
#
#         message = "Acc "
#         for i in conf_matrices:
#             message += "{}-{} ".format(i,round(conf_matrices[i].sum(axis=0)[1], 2))
#         print(message)
#
#     # Assuming we have 3 runs for each model configuration
#     from collections import defaultdict
#
#     conf_matrices = defaultdict(lambda:np.array([[0,0],[0,0]]))
#     for each_conf in multi_fold_results:
#         if each_conf == "Uni_Video" or each_conf == "Uni_Text": continue
#         print(each_conf)
#         conf_matrices_per_fold = []
#         fold_idx = None
#
#         # print(multi_fold_results["Uni_Video"].keys())
#         # centers = np.unique(multi_fold_results["Uni_Video"][fold_idx]["total_preds_target"])
#         # rounded_values_video = [min(centers, key=lambda center: abs(center - value)) for value in multi_fold_results['Uni_Video'][fold_idx]["total_preds"]]
#         # rounded_values_text = [min(centers, key=lambda center: abs(center - value)) for value in multi_fold_results['Uni_Text'][fold_idx]["total_preds"]]
#         # rounded_values_audio = [min(centers, key=lambda center: abs(center - value)) for value in multi_fold_results['Uni_Audio'][fold_idx]["total_preds"]]
#
#         targets = multi_fold_results["Uni_Video"][fold_idx]["total_preds_target"]
#         rounded_values_video = (multi_fold_results['Uni_Video'][fold_idx]["total_preds"][targets!=0]> 0)
#         rounded_values_text = (multi_fold_results['Uni_Text'][fold_idx]["total_preds"][targets!=0]> 0)
#         rounded_values_audio = (multi_fold_results['Uni_Audio'][fold_idx]["total_preds"][targets!=0]> 0)
#         binary_targets = (targets[targets!=0]> 0)
#
#
#         predictions = [
#             rounded_values_video == binary_targets,
#             rounded_values_text == binary_targets,
#             rounded_values_audio == binary_targets,
#
#         ]
#         rounded_values = (multi_fold_results[each_conf][fold_idx]["total_preds"][targets!=0]> 0)
#         predictions.append(
#             rounded_values == binary_targets,
#         )
#         for i in predictions:
#             print(i.shape)
#         conf_matrices[each_conf] = create_conf(predictions)
#
#
#     print_conf_accuracy(conf_matrices, multi_fold_results)
#     #make default dict with conf matrices 0s
#
#     conf_matrices = {
#                     "MCR": conf_matrices["MCR"],
#                     "Ensemble": conf_matrices["Ensemble"],
#                      "Joint Training": conf_matrices["Joint_Training"],
#                      "AGM": conf_matrices["AGM"],
#                      "MLB": conf_matrices["MLB"]
#                      # "Uni-Pre Finetuned": conf_matrices["Uni-Pre Finetuned"]}
#                      }
#
#     keys = "_".join(conf_matrices.keys())
#
#     # plot_confusion_matrix(conf_matrices, exp_name.format(keys))
#     plot_confusion_matrix_4d(conf_matrices, exp_name.format(keys))
#     # plot_one_example_confusion_matrix(conf_matrices, exp_name.format(keys))
#     # plot_3d_confusion_matrix(conf_matrices, exp_name)

num_folds = 3
# with open('./conf_vit_uni_val.pkl', 'wb') as f:
#     pickle.dump(multi_fold_results, f)
# with open('./conf_vit_late_ogm_mlb.pkl', 'rb') as f:
#     multi_fold_results_vit = pickle.load(f)
# with open('./conf_res_late_agm_mlb.pkl', 'rb') as f:
#     multi_fold_results_res = pickle.load(f)
# with open('./conf_res_ave_late_agm_mlb.pkl', 'rb') as f:
#     multi_fold_results_res_ave = pickle.load(f)
# with open('./conf_res_ucf_late_agm_mlb.pkl', 'rb') as f:
#     multi_fold_results_res_ucf = pickle.load(f)

with open('./confmatrix_results.pkl', 'rb') as f:
    total_results = pickle.load(f)


# confusion_matrix_3d_aggr(total_results["CREMA-D ResNet"], num_folds, "CREMA-D ResNet_{}")
# confusion_matrix_3d_aggr(total_results["CREMA-D Conformer"], num_folds, "CREMA-D Conformer_{}")
# confusion_matrix_3d_aggr(total_results["AVE ResNet"], num_folds, "AVE ResNet_{}")
# confusion_matrix_3d_aggr(total_results["AVE Conformer"], num_folds, "AVE Conformer_{}")
# confusion_matrix_3d_aggr(total_results["UCF ResNet"], num_folds, "UCF ResNet_{}")
# confusion_matrix_3d_mosei_aggr(total_results["MOSEI_V-T_Transformer"], 1, "MOSEI V-T Transformer_{}")
confusion_matrix_4d_mosei_aggr(total_results["MOSEI_V-T-A_Transformer"], 1, "MOSEI V-T-Α Transformer_{}")
# confusion_matrix_3d_mosei_aggr(total_results["MOSI_V-T_Transformer"], 1, "MOSI V-T Transformer_{}")
# confusion_matrix_4d_mosei_aggr(total_results["MOSI_V-T-A_Transformer"], 1, "MOSI V-T-Α Transformer_{}")
# confusion_matrix_3d_sthsth_aggr(total_results["SthSth_V-OF_Swin"], 1, "Sth-Sth V-OF Swin-TF_{}")






# plot_accuracies_bar(total_results, ["AVE Conformer", "AVE ResNet"])
# plot_accuracies_bar(total_results, ["CramedD_Dataloader"])


# total_results["AVE ResNet"]["AGM"] = {}
# total_results["AVE ResNet"]["AGM"][0] = total_results["AVE_Dataloader"]['singleloss_AGM'][0]
# total_results["AVE ResNet"]["AGM"][1] = total_results["AVE_Dataloader"]['singleloss_AGM'][1]
# total_results["AVE ResNet"]["AGM"][2] = total_results["AVE_Dataloader"]['singleloss_AGM'][2]

# total_results["AVE Conformer"]["Uni-Pre Frozen"] = {}
# total_results["AVE Conformer"]["Uni-Pre Frozen"] = total_results["AVE_Dataloader_vit"]['pre_frozen']
#
# total_results["CREMA-D ResNet"]["AGM"] = total_results["CramedD_Dataloader"]["concat_singleloss_AGM_test"]
# total_results["AVE Conformer"]["Uni-Pre Frozen"][0] = total_results["AVE_Dataloader_vit"]['pre_frozen'][0]
# total_results["AVE Conformer"]["Uni-Pre Frozen"][1] = total_results["AVE_Dataloader_vit"]['pre_frozen'][1]
# total_results["AVE Conformer"]["Uni-Pre Frozen"][2] = total_results["AVE_Dataloader_vit"]['pre_frozen'][2]


# total_results["CREMA-D ResNet"] = total_results["CREMAD_Res"]
# total_results["AVE ResNet"] = total_results["AVE_Res"]
# total_results["UCF ResNet"] = total_results["UCF_Res"]
# total_results["CREMA-D Conformer"] = total_results["CREMAD_Vit"]
# total_results["AVE Conformer"] = total_results["AVE_Vit"]


# total_results["MOSI_V-T_Transformer"] = {}
# total_results["MOSI_V-T_Transformer"]["Uni_Video"] = total_results["FactorCL_Dataloader"]["uni_video"]
# total_results["MOSI_V-T_Transformer"]["Uni_Text"] = total_results["FactorCL_Dataloader"]["uni_text"]
# # total_results["MOSI_V-T_Transformer"]["Uni_Audio"] = total_results["FactorCL_Dataloader"]["uni_audio"]
# total_results["MOSI_V-T_Transformer"]["Ensemble"] = total_results["FactorCL_Dataloader"]["ens"]
# total_results["MOSI_V-T_Transformer"]["Joint_Training"] = total_results["FactorCL_Dataloader"]["singleloss"]
# total_results["MOSI_V-T_Transformer"]["Multiloss"] = total_results["FactorCL_Dataloader"]["multiloss"]
# total_results["MOSI_V-T_Transformer"]["OGM"] = total_results["FactorCL_Dataloader"]["ogm"]
# total_results["MOSI_V-T_Transformer"]["AGM"] = total_results["FactorCL_Dataloader"]["agm"]
# total_results["MOSI_V-T_Transformer"]["MLB"] = total_results["FactorCL_Dataloader"]["mlb"]
# total_results["MOSI_V-T_Transformer"]["MCR"] = total_results["FactorCL_Dataloader"]["shufflegrad_test"]
# total_results["MOSI_V-T_Transformer"]["Uni-Pre Frozen"] = total_results["FactorCL_Dataloader"]["pre_frozen"]
# total_results["MOSI_V-T_Transformer"]["Uni-Pre Finetuned"] = total_results["FactorCL_Dataloader"]["pre"]

# total_results["MOSI_V-T-A_Transformer"] = {}
# total_results["MOSI_V-T-A_Transformer"]["Uni_Video"] = total_results["FactorCL_Dataloader"]["uni_video"]
# total_results["MOSI_V-T-A_Transformer"]["Uni_Text"] = total_results["FactorCL_Dataloader"]["uni_text"]
# total_results["MOSI_V-T-A_Transformer"]["Uni_Audio"] = total_results["FactorCL_Dataloader"]["uni_audio"]
# total_results["MOSI_V-T-A_Transformer"]["Ensemble"] = total_results["FactorCL_Dataloader"]["ens"]
# total_results["MOSI_V-T-A_Transformer"]["Joint_Training"] = total_results["FactorCL_Dataloader"]["singleloss"]
# total_results["MOSI_V-T-A_Transformer"]["Multiloss"] = total_results["FactorCL_Dataloader"]["multiloss"]
# # total_results["MOSI_V-T_Transformer"]["OGM"] = total_results["FactorCL_Dataloader"]["ogm"]
# total_results["MOSI_V-T-A_Transformer"]["AGM"] = total_results["FactorCL_Dataloader"]["agm"]
# total_results["MOSI_V-T-A_Transformer"]["MLB"] = total_results["FactorCL_Dataloader"]["mlb"]
# total_results["MOSI_V-T-A_Transformer"]["MCR"] = total_results["FactorCL_Dataloader"]["shufflegrad_test"]
# total_results["MOSI_V-T-A_Transformer"]["Uni-Pre Frozen"] = total_results["FactorCL_Dataloader"]["pre_frozen"]
# total_results["MOSI_V-T-A_Transformer"]["Uni-Pre Finetuned"] = total_results["FactorCL_Dataloader"]["pre"]

# total_results["SthSth_V-OF_Swin"] = {}
# total_results["SthSth_V-OF_Swin"]["Uni_Video"] = total_results["SthSth_VideoLayoutFlow"]["video_test"]
# total_results["SthSth_V-OF_Swin"]["Uni_Audio"] = total_results["SthSth_VideoLayoutFlow"]["flow_test_kinetics"]
# total_results["SthSth_V-OF_Swin"]["Ensemble"] = total_results["SthSth_VideoLayoutFlow"]["video_flow_Ens_mine"]
# total_results["SthSth_V-OF_Swin"]["Joint_Training"] = total_results["SthSth_VideoLayoutFlow"]["video_flow_Late"]
# total_results["SthSth_V-OF_Swin"]["Multiloss"] = total_results["SthSth_VideoLayoutFlow"]["video_flow_Late_Multi"]
# # total_results["MOSI_V-T_Transformer"]["OGM"] = total_results["SthSth_VideoLayoutFlow"]["ogm"]
# total_results["SthSth_V-OF_Swin"]["AGM"] = total_results["SthSth_VideoLayoutFlow"]["video_flow_Late_AGM"]
# total_results["SthSth_V-OF_Swin"]["MLB"] = total_results["SthSth_VideoLayoutFlow"]["video_flow_Late_MLB"]
# total_results["SthSth_V-OF_Swin"]["MCR"] = total_results["SthSth_VideoLayoutFlow"]["video_flow_Late_ShuffleGradEPIB_pre"]
# total_results["SthSth_V-OF_Swin"]["Uni-Pre Frozen"] = total_results["SthSth_VideoLayoutFlow"]["video_flow_Late_pre_frozen_mine"]
# # total_results["SthSth_V-OF_Swin"]["Uni-Pre Finetuned"] = total_results["SthSth_VideoLayoutFlow"]["pre"]
#
#
#
# # "Uni_Audio", "Uni_Audio", "Ensemble", "Joint Training", "AGM",  "MLB", "Uni-Pre Finetuned", "MCR"
# #
# del total_results["SthSth_VideoLayoutFlow"]
# # # # #save result in pickle
# # # # total_results["MOSEI V-T Transformer"] = total_results["FactorCL_Dataloader"]
# with open('./confmatrix_results.pkl', 'wb') as f:
#     pickle.dump(total_results, f)
