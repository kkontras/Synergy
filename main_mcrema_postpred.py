import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

sys.exc_info()
os.chdir('/users/sista/kkontras/Documents/Balance/')

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from posthoc.Helpers.Helper_Importer import Importer
from posthoc.Helpers.Helper_Validator import Validator

from utils.deterministic_pytorch import deterministic
device = "cuda:0"
def main(config_path, default_config_path, args):

    importer = Importer(config_name=config_path, default_files=default_config_path, device="cuda:0")
    m = ""
    enc_m = ""

    if "fold" in args and args.fold is not None:
        importer.config.dataset.data_split.fold = int(args.fold)
        importer.config.dataset.fold = int(args.fold)
        m += "fold{}".format(args.fold)
        # enc_m += "fold{}".format(args.fold)
        seeds = [0, 109, 19, 337] if "UCF" in config_path else [109, 19, 337]
        importer.config.training_params.seed = int(seeds[int(args.fold)])
        print("Seed: ", importer.config.training_params.seed)
        if "norm_wav_path" in importer.config.dataset:
            importer.config.dataset.norm_wav_path = importer.config.dataset.norm_wav_path.format(args.fold)
        if "norm_face_path" in importer.config.dataset:
            importer.config.dataset.norm_face_path = importer.config.dataset.norm_face_path.format(args.fold)
        if hasattr(importer.config.model, "encoders"):
            for i in range(len(importer.config.model.encoders)):
                importer.config.model.encoders[i].pretrainedEncoder.dir = importer.config.model.encoders[i].pretrainedEncoder.dir.format(args.fold)
    if "alpha" in args and args.alpha is not None:
        importer.config.model.args.bias_infusion.alpha = float(args.alpha)
        m += "_alpha{}".format(args.alpha)

    if "recon_weight1" in args and args.recon_weight1 is not None:
        importer.config.model.args.bias_infusion.weight1 = float(args.recon_weight1)
        m += "_w1{}".format(args.recon_weight1)
    if "recon_weight2" in args and args.recon_weight2 is not None:
        importer.config.model.args.bias_infusion.weight2 = float(args.recon_weight2)
        m += "_w2{}".format(args.recon_weight2)
    if "recon_epochstages" in args and args.recon_epochstages is not None:
        importer.config.model.args.bias_infusion.epoch_stages = int(args.recon_epochstages)
        m += "_epochstage{}".format(args.recon_epochstages)
    if "recon_ensemblestages" in args and args.recon_ensemblestages is not None:
        importer.config.model.args.bias_infusion.ensemble_stages = int(args.recon_ensemblestages)
        m += "_ensstage{}".format(args.recon_ensemblestages)


    if "tanh_mode" in args and args.tanh_mode is not None and args.tanh_mode != "None":
        importer.config.model.args.bias_infusion.tanh_mode = args.tanh_mode
        if args.tanh_mode == "trial8":
            importer.config.early_stopping.max_epoch=100
        if args.tanh_mode == "cls":
            importer.config.task="classification"

        enc_m += "_tanhmode{}".format(args.tanh_mode)
        m += "_tanhmode{}".format(args.tanh_mode)
    if "num_classes" in args and args.num_classes is not None:
        importer.config.model.args.num_classes = int(args.num_classes)
        if hasattr(importer.config.model, "encoders"):
            for i in range(len(importer.config.model.encoders)):
                importer.config.model.encoders[i].args.num_classes = int(args.num_classes)
        enc_m += "_numclasses{}".format(args.num_classes)
        m += "_numclasses{}".format(args.num_classes)
    if "tanh_mode_beta" in args and args.tanh_mode_beta is not None:
        importer.config.model.args.bias_infusion.tanh_mode_beta = float(args.tanh_mode_beta)
        m += "_beta{}".format(args.tanh_mode_beta)
    if "regby" in args and args.regby is not None:
        importer.config.model.args.bias_infusion.regby = args.regby
        m += "_regby{}".format(args.regby)
    if "clip" in args and args.clip is not None:
        importer.config.model.args.clip_grad = True
        importer.config.model.args.clip_value = float(args.clip)
        m += "_clip{}".format(args.clip)
    if "l" in args and args.l is not None:
        importer.config.model.args.bias_infusion.l = float(args.l)
        m += "_l{}".format(args.l)
    if "multil" in args and args.multil is not None:
        for i in importer.config.model.args.multi_loss.multi_supervised_w:
            if i != "combined" and importer.config.model.args.multi_loss.multi_supervised_w[i] !=0:
                importer.config.model.args.multi_loss.multi_supervised_w[i] = float(args.multil)
        m += "_multil{}".format(args.multil)
    if "l_diffsq" in args and args.l_diffsq is not None:
        importer.config.model.args.bias_infusion.l_diffsq = float(args.l_diffsq)
        m += "_ldiffsq{}".format(args.l_diffsq)

    "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/SyntheticGaussian_models/uni0__tanhmodetrial5_numclasses2_ratio1_bs64.pth.tar"
    if "lib" in args and args.lib is not None:
        importer.config.model.args.bias_infusion.lib = float(args.lib)
        m += "_lib{}".format(args.lib)
    if "ratio_us" in args and args.ratio_us is not None:
        importer.config.dataset.ratio_us = float(args.ratio_us)
        m += "_ratio{}".format(args.ratio_us)
        enc_m += "_ratio{}".format(args.ratio_us)
    if "ratio_snr" in args and args.ratio_snr is not None:
        importer.config.dataset.ratio_snr = float(args.ratio_snr)
        m += "_ratiosnr{}".format(args.ratio_snr)
        enc_m += "_ratiosnr{}".format(args.ratio_snr)
    if "kmepoch" in args and args.kmepoch is not None:
        importer.config.model.args.bias_infusion.keep_memory_epoch = int(args.kmepoch)
        m += "_kmepoch{}".format(args.kmepoch)

    if "mmcosine_scaling" in args and args.mmcosine_scaling is not None:
        importer.config.model.args.bias_infusion.mmcosine_scaling = float(args.mmcosine_scaling)
        m += "_mmcosinescaling{}".format(args.mmcosine_scaling)

    if "load_ongoing" in args and args.load_ongoing is not None:
        importer.config.model.load_ongoing = args.load_ongoing

    if "ilr_c" in args and "ilr_g" in args and args.ilr_c is not None and args.ilr_g is not None:
        importer.config.model.args.bias_infusion.init_learning_rate = {
          "c" : float(args.ilr_c),
          "g" : float(args.ilr_g)
        }
        m += "_ilrcg{}_{}".format(args.ilr_c, args.ilr_g)
        enc_m += "_lr{}".format(args.lr)



    if "ending_epoch" in args and args.ending_epoch is not None:
        importer.config.model.args.bias_infusion.ending_epoch = int(args.ending_epoch)
        m += "_endingepoch{}".format(args.ending_epoch)
    if "num_samples" in args and args.num_samples is not None:
        importer.config.model.args.bias_infusion.num_samples = int(args.num_samples)
        m += "_numsamples{}".format(args.num_samples)
    if "pow" in args and args.pow is not None:
        importer.config.model.args.bias_infusion.pow = int(args.pow)
        m += "_pow{}".format(args.pow)
    if "nstep" in args and args.nstep is not None:
        importer.config.model.args.bias_infusion.nstep = int(args.nstep)
        m += "_nstep{}".format(args.nstep)
    if "contr_coeff" in args and args.contr_coeff is not None:
        importer.config.model.args.bias_infusion.contr_coeff = float(args.contr_coeff)
        m += "_contrcoeff{}".format(args.contr_coeff)
    if "kde_coeff" in args and args.kde_coeff is not None:
        importer.config.model.args.bias_infusion.kde_coeff = float(args.kde_coeff)
        m += "_kde_coeff{}".format(args.kde_coeff)
    if "etube" in args and args.etube is not None:
        importer.config.model.args.bias_infusion.etube = float(args.etube)
        m += "_etube{}".format(args.etube)
    if "temperature" in args and args.temperature is not None:
        importer.config.model.args.bias_infusion.temperature = float(args.temperature)
        m += "_temp{}".format(args.temperature)
    if "shuffle_type" in args and args.shuffle_type is not None:
        importer.config.model.args.bias_infusion.shuffle_type = args.shuffle_type
        m += "_st{}".format(args.shuffle_type)
    if "contr_type" in args and args.contr_type is not None:
        importer.config.model.args.bias_infusion.contr_type = args.contr_type
        m += "_contrtype{}".format(args.contr_type)
    if "validate_with" in args and args.validate_with is not None:
        importer.config.early_stopping.validate_with = args.validate_with
        enc_m += "_vld{}".format(args.validate_with)
        m += "_vld{}".format(args.validate_with)
    if "base_alpha" in args and args.base_alpha is not None:
        importer.config.dataset.base_alpha = float(args.base_alpha)
        m += "_basealpha{}".format(args.base_alpha)
    if "alpha_var" in args and args.alpha_var is not None:
        importer.config.dataset.alpha_var = float(args.alpha_var)
        m += "_alphavar{}".format(args.alpha_var)
    if "base_beta" in args and args.base_beta is not None:
        importer.config.dataset.base_beta = float(args.base_beta)
        importer.config.model.args.layers = int(args.base_beta)
        if hasattr(importer.config.model, "encoders"):
            for i in range(len(importer.config.model.encoders)):
                importer.config.model.encoders[i].args.layers = int(args.base_beta)
        enc_m += "_basebeta{}".format(args.base_beta)
        m += "_basebeta{}".format(args.base_beta)
    if "beta_var" in args and args.beta_var is not None:
        importer.config.dataset.beta_var = float(args.beta_var)
        m += "_betavar{}".format(args.beta_var)
        if hasattr(importer.config.model, "encoders"):
            for i in range(len(importer.config.model.encoders)):
                m_enc = ""
                m_enc += "_basealpha{}".format(args.base_alpha)
                m_enc += "_alphavar{}".format(args.alpha_var)
                m_enc += "_basebeta{}".format(args.base_beta)
                m_enc += "_betavar{}".format(args.beta_var)
                importer.config.model.encoders[i].pretrainedEncoder.dir = importer.config.model.encoders[i].pretrainedEncoder.dir.format(m_enc)
    if "optim_method" in args and args.optim_method is not None:
        importer.config.model.args.bias_infusion.optim_method = args.optim_method
        m += "_optim{}".format(args.optim_method)
    if "lr" in args and args.lr is not None:
        importer.config.optimizer.learning_rate = float(args.lr)
        m += "_lr{}".format(args.lr)
        enc_m += "_lr{}".format(args.lr)
    if "wd" in args and args.wd is not None:
        importer.config.optimizer.weight_decay = float(args.wd)
        m += "_wd{}".format(args.wd)
        enc_m += "_wd{}".format(args.wd)
    if "cls" in args and args.cls is not None:
        importer.config.model.args.cls_type = args.cls
        m += "_{}".format(args.cls)
    if "batch_size" in args and args.batch_size is not None:
        importer.config.training_params.batch_size = int(args.batch_size)
        m += "_bs{}".format(args.batch_size)
        enc_m += "_bs{}".format(args.batch_size)
    if "commonlayers" in args and args.commonlayers is not None:
        importer.config.model.args.common_layer = int(args.commonlayers)
        m += "_commonlayers{}".format(args.commonlayers)

    importer.config.model.save_dir = importer.config.model.save_dir.format(m)
    # if enc_m != "":
    if hasattr(importer.config.model, "encoders"):
        for i in range(len(importer.config.model.encoders)):
            importer.config.model.encoders[i].pretrainedEncoder.dir = importer.config.model.encoders[i].pretrainedEncoder.dir.format(enc_m)


    importer.config.training_params.test_batch_size = 6


    importer.load_checkpoint()
    # deterministic(importer.config.training_params.seed)

    # print(importer.checkpoint["configs"])
    # plotter = LogsPlotter(config=importer.config, logs=importer.checkpoint["logs"])

    best_model = importer.get_model(return_model="best_model")
    # best_model = importer.get_model(return_model="running_model")

    data_loader = importer.get_dataloaders()

    validator = Validator(model=best_model, data_loader=data_loader, config=importer.config, device=device)
    # test_results = validator.get_results(set="Validation", print_results=True)

    test_results = validator.get_results(set="Test", print_results=True)
    # validator.save_test_results(checkpoint=importer.checkpoint,
    #                             save_dir=importer.config.model.save_dir, test_results=test_results)
    import pickle
    if not os.path.exists('./game_conf_matrix.pkl'):
        total_results = {}
    else:
        with open('./game_conf_matrix.pkl', 'rb') as f:
            total_results = pickle.load(f)
    if "bias_infusion" not in importer.config.model.args or "regby" not in importer.config.model.args.bias_infusion:
        name = config_path.split("/")[2]+"-"+config_path.split("/")[3]+"-"+config_path.split("/")[4]
    else:
        name = config_path.split("/")[2]+"-"+config_path.split("/")[3]+"-"+importer.config.model.args.bias_infusion.regby
    print(name)
    if name not in total_results:
        total_results[name] = {}
    total_results[name][args.fold] = test_results

    # if importer.config.dataset.dataloader_class not in total_results:
    #     total_results[importer.config.dataset.dataloader_class] = {}
    # if config_path.split("/")[-1].split(".")[0] not in total_results[importer.config.dataset.dataloader_class]:
    #     total_results[importer.config.dataset.dataloader_class][config_path.split("/")[-1].split(".")[0]] = {}
    # total_results[importer.config.dataset.dataloader_class][config_path.split("/")[-1].split(".")[0]][args.fold] = test_results

    # save result in pickle
    with open('./game_conf_matrix.pkl', 'wb') as f:
        pickle.dump(total_results, f)


parser = argparse.ArgumentParser(description="My Command Line Program")
parser.add_argument('--config', help="Number of config file")
parser.add_argument('--default_config', help="Number of config file")
parser.add_argument('--fold', help="Fold")
parser.add_argument('--alpha', help="Alpha")
parser.add_argument('--tanh_mode', help="tanh_mode")
parser.add_argument('--tanh_mode_beta', help="tanh_mode_beta")
parser.add_argument('--regby', help="regby")
parser.add_argument('--clip', help="Gradient Clip Value")
parser.add_argument('--batch_size', help="batch_size")
parser.add_argument('--l', help="L for Gat")
parser.add_argument('--multil', help="Coeff of Multi-Loss")
parser.add_argument('--l_diffsq', help="L for Gat")
parser.add_argument('--lib', help="lib for Gat")
parser.add_argument('--ratio_us', help="lib for Gat")
parser.add_argument('--ratio_snr', help="lib for Gat")
parser.add_argument('--kmepoch', help="keep memory epoch")
parser.add_argument('--num_samples', help="Number of samples for Gat")
parser.add_argument('--pow', help="ShuffleGrad power")
parser.add_argument('--nstep', help="ShuffleGrad nstep Reg-Dist-Sep")
parser.add_argument('--contr_coeff', help="ShuffleGrad Contrastive Coefficient")
parser.add_argument('--kde_coeff', help="ShuffleGrad kde_coeff Coefficient")
parser.add_argument('--etube', help="ShuffleGrad Etube")
parser.add_argument('--temperature', help="ShuffleGrad Contrastive Temperature")
parser.add_argument('--contr_type', help="ShuffleGrad Contrastive type")
parser.add_argument('--shuffle_type', help="shuffle_type")
parser.add_argument('--validate_with', help="validate_with")
parser.add_argument('--num_classes', help="num_classes")
parser.add_argument('--base_alpha', help="Synthetic Alpha")
parser.add_argument('--alpha_var', help="Synthetic Alpha Variance")
parser.add_argument('--base_beta', help="Synthetic Beta")
parser.add_argument('--beta_var', help="Synthetic Beta Variance")
parser.add_argument('--optim_method', help="Optim for Gat")
parser.add_argument('--ilr_c', help="Initial Learning Rate Audio")
parser.add_argument('--ilr_g', help="Initial Learning Rate Video")
parser.add_argument('--mmcosine_scaling', help="mmcosine_scaling")
parser.add_argument('--ending_epoch', help="Ending epoch")
parser.add_argument('--load_ongoing', help="Ending epoch")
parser.add_argument('--commonlayers', help="Fusion with Conformer Layers")
parser.add_argument('--recon_weight1', help="ReconBoost Parameters")
parser.add_argument('--recon_weight2', help="ReconBoost Parameters")
parser.add_argument('--recon_epochstages', help="ReconBoost Parameters")
parser.add_argument('--recon_ensemblestages', help="ReconBoost Parameters")
parser.add_argument('--lr', required=False, help="Learning Rate", default=None)
parser.add_argument('--wd', required=False, help="Weight Decay", default=None)
parser.add_argument('--cls', required=False, help="CLS linear, nonlinear, highlynonlinear", default=None)
args = parser.parse_args()
for var_name in vars(args):
    var_value = getattr(args, var_name)
    if var_value == "None":
        setattr(args, var_name, None)

if "UCF101" in args.config:
    for i in range(1, 4):
        args.fold = i
        main(config_path=args.config, default_config_path=args.default_config, args=args)
elif "Mosei" in args.config or "Mosi" in args.config or "SthSth" in args.config:
    # args.fold = 0
    main(config_path=args.config, default_config_path=args.default_config, args=args)
else:
    for i in range(3):
        args.fold = i
        main(config_path=args.config, default_config_path=args.default_config, args=args)
