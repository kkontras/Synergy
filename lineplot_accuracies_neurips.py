import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

def plot_accuracies_bar_next(data, std_data, label, dataset_name):

    datasets = data.keys()
    base_color = sns.color_palette("muted", n_colors=len(datasets))  # Use one tone for simplicity

    fig = plt.figure(figsize=(15, 6))

    offset_model = 0.35
    offset_dataset = 0.1*offset_model
    # Plot individual points
    for di, dataset in enumerate(datasets):
        this_data = [data[dataset][i] if data[dataset][i] is not None else False for i in np.arange(0, len(data[dataset]))]
        this_std = [std_data[dataset][i] if data[dataset][i] is not None else False for i in np.arange(0, len(data[dataset]))]
        color_i = 0 if di==1 else 1
        if len(datasets)==1: color_i = 0
        plt.bar(np.arange(len(data[dataset]))*offset_model+di*offset_dataset, this_data,  yerr=this_std, width=0.25, color=base_color[color_i], label=dataset, alpha=0.8,error_kw={'ecolor': base_color[color_i], 'alpha':0.5, 'capsize':5})
        # plt.errorbar(np.arange(len(data[dataset]))*offset_model+di*offset_dataset, this_data, yerr=this_std, color=base_color[di], alpha=0.2)
        #remove line from errorbar and keep only the errorbar
        # plt.errorbar(np.arange(len(data[dataset]))*offset_model+di*offset_dataset, this_data, yerr=this_std, capsize=5, fmt='none', color=base_color[di], alpha=0.5)
        # plt.plot(np.arange(len(data[dataset]))*offset_model+di*offset_dataset, this_data, color=base_color[di], alpha=0.2)
    values, labels = [], []
    for di, dataset in enumerate(datasets):
        values.append(np.arange(len(data[dataset]))*offset_model+di*offset_dataset)
        labels.append(label)
        if di==0:
            break
    #flatten lists
    # print(values)
    # print(labels)
    values = [item for sublist in values for item in sublist]
    labels = [item for sublist in labels for item in sublist]



    if dataset_name == "AVE":
        tickfont = 28
        labels_font = 32
    elif dataset_name == "CREMA-D":
        tickfont = 18
        labels_font = 24
    elif dataset_name == "UCF101":
        tickfont = 28
        labels_font = 32


    plt.xticks(values, labels, rotation=30, ha="right", fontsize=tickfont)
    plt.yticks(fontsize=tickfont)
    # plt.title("Accuracy Comparison Across Models on {} Dataset".format(dataset_name), fontsize=20)
    plt.ylabel("Accuracy (%)", fontsize=labels_font)

    for di, dataset in enumerate(datasets):
        ensemble_value = data[dataset][label.index("Ensemble")]

        color_i = 0 if di == 1 else 1
        if len(datasets) == 1: color_i = 0
        plt.plot([0*offset_model+di*offset_dataset, 23*offset_model+di*offset_dataset], [ensemble_value, ensemble_value], color=base_color[color_i], linestyle='--', linewidth=2)

    model_fusion_color = "#3b3b3b"

    if dataset_name == "AVE":
        bottom_part = 0.90
        line_offset = [[4.5, 17.8], [20.2, 26.6]]
    elif dataset_name == "CREMA-D":
        bottom_part = 0.90
        line_offset = [[2.8, 12.4], [13.6, 20.5]]
    elif dataset_name == "UCF101":
        bottom_part = 0.89
        line_offset = [[4.5, 17.8], [20.2, 26.6]]


    ref_to_the_plot = [0.095,0.127]
    diff = ref_to_the_plot[1]-ref_to_the_plot[0]
    line1 = Line2D([ref_to_the_plot[0]+diff*line_offset[0][0], ref_to_the_plot[0]+diff*line_offset[0][1]], [bottom_part, bottom_part], transform=fig.transFigure, figure=fig, color=model_fusion_color, linewidth=4)
    line2 = Line2D([ref_to_the_plot[0]+diff*line_offset[1][0], ref_to_the_plot[0]+diff*line_offset[1][1]], [bottom_part, bottom_part], transform=fig.transFigure, figure=fig, color=model_fusion_color, linewidth=4)
    # line3 = Line2D([ref_to_the_plot[0]+diff*21.55, ref_to_the_plot[0]+diff*26.6], [bottom_part, bottom_part], transform=fig.transFigure, figure=fig, color=model_fusion_color, linewidth=4)
    fig.lines.extend([line1, line2])


    if dataset_name == "AVE":
        text_bottom = 0.99
        off_set_text = [5.4, 12.5]
    elif dataset_name == "CREMA-D":
        text_bottom = 0.98
        off_set_text = [7.9, 18.7]
    elif dataset_name == "UCF101":
        text_bottom = 1.045
        off_set_text = [5.4, 12.5]


    # text_bottom = 0.98
    plt.text(off_set_text[0]*offset_model, text_bottom, 'Late-Linear', transform=plt.gca().get_xaxis_transform(), ha='center', va='top', fontsize=labels_font, color=model_fusion_color)
    plt.text(off_set_text[1]*offset_model, text_bottom, 'Mid-MLP', transform=plt.gca().get_xaxis_transform(), ha='center', va='top', fontsize=labels_font, color=model_fusion_color)
    # plt.text(17.7*offset_model, text_bottom, 'Mid', transform=plt.gca().get_xaxis_transform(), ha='center', va='top', fontsize=16, color=model_fusion_color)
    #change color of text

    plt.xlabel("Models", fontsize=labels_font)


    # plt.text(24.6*offset_model, -0.05, 'Models', transform=plt.gca().get_xaxis_transform(), ha='center', va='top', fontsize=20)
    if dataset_name == "AVE":
        plt.ylim(30,82.39)
        # plt.ylim(bottom=30)
    elif dataset_name == "CREMA-D":
        plt.ylim(41,95.25)
        # plt.ylim(bottom=41)

    elif dataset_name == "UCF101":
        plt.ylim(25,60)
        # plt.ylim(bottom=30)

    # plt.subplots_adjust()  # Adjust the bottom parameter as needed

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    if dataset_name == "CREMA-D":
        plt.legend(bbox_to_anchor=(1.25, 0), loc='lower right',  fontsize=tickfont).set_title("Backbone Model", prop={'size': tickfont})
    plt.tight_layout()  # Adjust the bottom parameter as needed

    plt.savefig("./Figures/Neurips_bar_results_{}.pdf".format(dataset_name), format='pdf')# Ensure legend is shown
    plt.show()
def plot_accuracies_bar_fusiongates(data, std_data, label, dataset_name):

    datasets = data.keys()
    base_color = sns.color_palette("muted", n_colors=2)  # Use one tone for simplicity

    fig = plt.figure(figsize=(10, 6))

    offset_model = 0.35
    offset_dataset = 0.1*offset_model
    # Plot individual points
    for di, dataset in enumerate(datasets):
        this_data = [data[dataset][i] if data[dataset][i] is not None else False for i in np.arange(0, len(data[dataset]))]
        this_std = [std_data[dataset][i] if data[dataset][i] is not None else False for i in np.arange(0, len(data[dataset]))]
        color_i = 0 if di==1 else 1
        # color_i = di
        # if len(mydatasets)==1: color_i = 0
        plt.bar(np.arange(len(data[dataset]))*offset_model+di*offset_dataset, this_data,  yerr=this_std, width=0.25, color=base_color[color_i], label=dataset, alpha=0.8,error_kw={'ecolor': base_color[color_i], 'alpha':0.5, 'capsize':5})
        # plt.errorbar(np.arange(len(data[dataset]))*offset_model+di*offset_dataset, this_data, yerr=this_std, color=base_color[di], alpha=0.2)
        #remove line from errorbar and keep only the errorbar
        # plt.errorbar(np.arange(len(data[dataset]))*offset_model+di*offset_dataset, this_data, yerr=this_std, capsize=5, fmt='none', color=base_color[di], alpha=0.5)
        # plt.plot(np.arange(len(data[dataset]))*offset_model+di*offset_dataset, this_data, color=base_color[di], alpha=0.2)
    values, labels = [], []
    for di, dataset in enumerate(datasets):
        values.append(np.arange(len(data[dataset]))*offset_model+di*offset_dataset)
        labels.append(label)
        if di==0:
            break
    #flatten lists
    # print(values)
    # print(labels)
    values = [item for sublist in values for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    plt.xticks(values, labels, rotation=30, ha="right", fontsize=15)
    plt.yticks(fontsize=16)
    labels_font = 22

    # plt.title("Accuracy Comparison Across Models on {} Dataset".format(dataset_name), fontsize=20)
    plt.ylabel("Accuracy (%)", fontsize=labels_font)


    # for di, dataset in enumerate(mydatasets):
    #     ensemble_value = data[dataset][label.index("Ensemble")]
    #     plt.plot([0*offset_model+di*offset_dataset, 23*offset_model+di*offset_dataset], [ensemble_value, ensemble_value], color=base_color[di], linestyle='--', linewidth=2)

    model_fusion_color = "#3b3b3b"

    if dataset_name == "AVE":
        bottom_part = 0.13
    elif dataset_name == "CREMA-D":
        bottom_part = 0.13
    elif dataset_name == "UCF101":
        bottom_part = 0.13

    bottom_part = 0.90

    ref_to_the_plot = [0.095,0.127]
    diff = ref_to_the_plot[1]-ref_to_the_plot[0]
    line1 = Line2D([ref_to_the_plot[0]+diff*0.9, ref_to_the_plot[0]+diff*5.8], [bottom_part, bottom_part], transform=fig.transFigure, figure=fig, color=model_fusion_color, linewidth=4)
    line2 = Line2D([ref_to_the_plot[0]+diff*7.6, ref_to_the_plot[0]+diff*12.5], [bottom_part, bottom_part], transform=fig.transFigure, figure=fig, color=model_fusion_color, linewidth=4)
    line3 = Line2D([ref_to_the_plot[0]+diff*14.2, ref_to_the_plot[0]+diff*19.1], [bottom_part, bottom_part], transform=fig.transFigure, figure=fig, color=model_fusion_color, linewidth=4)
    fig.lines.extend([line1, line2, line3])


    if dataset_name == "AVE":
        text_bottom = -0.38
    elif dataset_name == "CREMA-D":
        text_bottom = -0.38
    elif dataset_name == "UCF101":
        text_bottom = -0.38

    text_bottom = 0.97
    plt.text(1.5*offset_model, text_bottom, 'FiLM', transform=plt.gca().get_xaxis_transform(), ha='center', va='top', fontsize=22, color=model_fusion_color)
    plt.text(6.5*offset_model, text_bottom, 'Gated', transform=plt.gca().get_xaxis_transform(), ha='center', va='top', fontsize=22, color=model_fusion_color)
    plt.text(11.5*offset_model, text_bottom, 'TF', transform=plt.gca().get_xaxis_transform(), ha='center', va='top', fontsize=22, color=model_fusion_color)
    #change color of text

    plt.xlabel("Models", fontsize=labels_font)


    # plt.text(24.6*offset_model, -0.05, 'Models', transform=plt.gca().get_xaxis_transform(), ha='center', va='top', fontsize=20)
    if dataset_name == "AVE":
        plt.ylim(30,82.39)
        # plt.ylim(bottom=30)
    elif dataset_name == "CREMA-D":
        plt.ylim(41,95.25)
        # plt.ylim(bottom=41)

    elif dataset_name == "UCF101":
        plt.ylim(30,60)
        # plt.ylim(bottom=30)


    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    plt.legend(bbox_to_anchor=(1.33, 0), loc='lower right',  fontsize=14).set_title("Backbone Model", prop={'size': 16})
    plt.tight_layout()
    # plt.legend(title="Backbone Model", loc='lower right', fontsize=10)
    plt.savefig("./Figures/Neurips_bar_results_fusiongates_{}.pdf".format(dataset_name), format='pdf')# Ensure legend is shown
    plt.show()

# Original data setup
label = [ "Ensemble", "Video", "Audio",
         "Joint Training", "MMCosine [27]", "MSLR [28]",
          "PMR [5]", "OGM [18]", "AGM [15]", "Multi-Loss", "MLB", "Pre Frozen", "Pre Finetuned", "Rergularizer",
         "", "Joint Training", "PMR [5]", "AGM [15]", "Multi-Loss", "MLB", "Pre Frozen", "Pre Finetuned", "Rergularizer" ]
cremad_data = {
    "ResNet": [
        71.70,
        55.38, 60.60, #checked
        62.56, 58.83, 56.53,
        53.74,  65.57, 69.28, 69.22, 71.17, 72.28, 73.27, 76.01,
        None,   62.64, 20.82, 57.00, 70.62, 71.81, 70.95, 74.09, 74.83],
        # None,   55.60, 0,    69.12, 69.18, 70.69
    # ],

    "Conformer": [
        84.57,
        69.42, 76.82,
        75.13, 73.48, 77.05,

        80.49, 82.41, 78.54, 82.59, 85.16, 82.43, 84.09, 0,
        None, 70.29,  77.67, 72.45, 82.88, 85.19, 0, 0, 0]
        # None, 70.3, 0, 72.86, 83.75, 84.37]

}
cremad_std_data = {
    "ResNet": [
        0,
        2.31, 2.97,
        1.36, 1.57, 2.24,
        2.34, 3.84, 1.39, 1.82, 1.31, 1.76, 2.82, 2.07,
        0, 3.46, 0.34, 1.96, 2.51, 1.43, 3.8, 3.77, 2.93],
        # 0, 0, 1.14, 1.93, 1.64, 1.06],
    "Conformer": [
        0,
        2.75, 1.97,
        1.70, 2.12, 2.40,
        0.20,   0,    2.56, 0.92, 0.85, 2.04, 0.9, 0,
        0,      1.48, 1.87, 2.06, 1.50, 1.97, 0,0,0]
        # 0,      0,    0, 1.34, 1.35, 2.16]
}

cremad_data_fusiongates = {
    "Conformer": [
        73.81, 68.86, 84.72, 84.35, None,
        71.69, 69.04, 83.29, 84.19, None,
        71.04, 70.91, 83.42, 85.53
    ],
    "ResNet": [
        64.79,  57.81, 68.26, 72.19, None,
        59.06,  55.70, 70.72, 71.14, None,
        58.69,  54.04, 69.66, 69.69
    ]
}
cremad_std_data_fusiongates = {
    "Conformer": [
        1.47, 0.56, 0.71, 0.46, 0,
        2.48, 0.68, 1.10, 1.37, 0,
        2.46, 0.68, 1.49, 0.88, 0],
    "ResNet": [
        1.13,    1.30, 2.90, 0.53, 0,
        4.66,    4.67, 2.42, 1.74, 0,
        2.22,    4.47, 3.39, 1.11, 0]

}

# plot_accuracies(cremad_data, cremad_std_data, label, "CREMA-D")
# plot_accuracies_bar_seq(cremad_data, cremad_std_data, label, "CREMA-D")

cremad_data = {"Conformer": cremad_data["Conformer"], "ResNet": cremad_data["ResNet"]}
cremad_std_data = {"Conformer": cremad_std_data["Conformer"], "ResNet": cremad_std_data["ResNet"]}
# plot_accuracies_bar_next(cremad_data, cremad_std_data, label, "CREMA-D")

fusiongate_label = ["Joint Training","AGM [15]", "Multi-Loss", "MLB", "", "Joint Training","AGM [15]", "Multi-Loss", "MLB", "", "Joint Training","AGM [15]", "Multi-Loss", "MLB"]
# plot_accuracies_bar_fusiongates(cremad_data_fusiongates, cremad_std_data_fusiongates, fusiongate_label, "CREMA-D")

ave_ucf_label = [ "Ensemble", "Video", "Audio",
         "Joint Training", "MMCosine [27]", "MSLR [28]",
          "PMR [5]", "OGM [18]", "AGM [15]", "Multi-Loss", "MLB", "Pre Frozen", "Pre Finetuned", "Rergularizer",
         "", "Joint Training", "AGM [15]", "Multi-Loss", "MLB", "Pre Frozen", "Pre Finetuned", "Rergularizer" ]

ave_data = {
    "ResNet": [
        62.60, 45.69,66.67,
        66.67,66.75,
        37.40, 67.91,  65.84, 70.07, 70.56,
        None, 64.26, 61.69, 71.14, 69.82]
}
ave_std_data = {
    "ResNet": [
        0.94, 1.63,
        1.53, 0.92, 1.73,
              7.55, 0.54, 2.65, 0.94, 1.38,
        None, 3.77, 3.42, 1.33, 0.92]
}

# plot_accuracies_bar_next(ave_data, ave_std_data, ave_ucf_label, "AVE")


ucf_data = {
    "ResNet": [
        0,
        30.33,38.34,
        47.70,

        46.9,47.91,
        40.86,51.79,50.46,51.12, 51.99, 0, 0, 0,
        None,40.27, 39.88,44.72, 46.08, 0, 0, 0]
}
ucf_std_data = {
    "ResNet": [
        0,
        1.49, 0.83,
        1.47,3.61,3.80,
        2.01,1.93,1.48,1.79,2.23, 0, 0, 0,
        None,2.04,4.75,2.04,1.15,0, 0, 0]
}
print(len(ucf_data["ResNet"]))
print(len(ucf_std_data["ResNet"]))
plot_accuracies_bar_next(ucf_data, ucf_std_data, ave_ucf_label, "UCF101")
