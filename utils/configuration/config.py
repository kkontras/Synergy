import os

import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler
import importlib

import json
from easydict import EasyDict
from pprint import pprint

from .dirs import create_dirs
import sys
import copy

def _safe_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _resolve_hf_cache_dir(config):
    dataset = _safe_get(config, "dataset")
    model = _safe_get(config, "model")

    candidates = []

    if dataset is not None:
        for key in ("hf_cache", "cache_dir", "cache_root"):
            val = _safe_get(dataset, key)
            if isinstance(val, str) and val.strip():
                candidates.append(val.strip())

        data_roots = _safe_get(dataset, "data_roots")
        if isinstance(data_roots, str) and data_roots.strip():
            candidates.append(os.path.join(data_roots.strip(), "hf_cache"))

    if model is not None:
        save_base_dir = _safe_get(model, "save_base_dir")
        if isinstance(save_base_dir, str) and save_base_dir.strip():
            candidates.append(os.path.join(save_base_dir.strip(), "hf_cache"))

    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except Exception:
            continue
    return None

def configure_hf_cache(config, logger=None):
    cache_root = _resolve_hf_cache_dir(config)
    if not cache_root:
        return None

    env_paths = {
        "HF_HOME": cache_root,
        "TRANSFORMERS_CACHE": os.path.join(cache_root, "transformers"),
        "HF_DATASETS_CACHE": os.path.join(cache_root, "datasets"),
        "HF_HUB_CACHE": os.path.join(cache_root, "hub"),
    }

    for path in env_paths.values():
        os.makedirs(path, exist_ok=True)

    for key, path in env_paths.items():
        if not os.environ.get(key):
            os.environ[key] = path

    # Keep HF cache path logging opt-in to avoid noisy repeated logs
    hf_cache_log = os.environ.get("HF_CACHE_LOG", "").strip().lower()
    if logger is not None and hf_cache_log in ("1", "true", "yes", "on"):
        logger.info(
            "HF cache dirs: HF_HOME=%s TRANSFORMERS_CACHE=%s HF_DATASETS_CACHE=%s HF_HUB_CACHE=%s",
            os.environ.get("HF_HOME"),
            os.environ.get("TRANSFORMERS_CACHE"),
            os.environ.get("HF_DATASETS_CACHE"),
            os.environ.get("HF_HUB_CACHE"),
        )
    return cache_root

def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    logging.shutdown()

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)

def setup_logger():



    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    # main_logger = logging.getLogger()
    # main_logger.setLevel(logging.WARNING)

    logger = logging.getLogger('wandb')
    logger.setLevel(logging.WARNING)


def merge_dicts(default_dict: EasyDict, dict2: EasyDict)-> EasyDict:
    """
    Recursively merges two dictionaries, combining their values.
    If a key exists in both dictionaries, the value from default_dict takes precedence.
    """
    merged = copy.deepcopy(default_dict)  # Start with a copy of default_dict

    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], EasyDict) and isinstance(value, EasyDict):
            # Recursively merge nested dictionaries
            merged[key] = merge_dicts(merged[key], value)
        else:
            # Otherwise, simply update the value
            merged[key] = value

    return merged

def process_config_default(json_file, default_files=False, printing = True):
    """
    Get the json file
    Processing it with EasyDict to be accessible as attributes
    then editing the path of the experiments folder
    creating some important directories in the experiment folder
    Then setup the logging in the whole program
    Then return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    config_logger = logging.getLogger("Config")
    config_logger.setLevel(logging.INFO)

    if printing:
        config_logger.info(json_file)
    config, _ = get_config_from_json(json_file)

    if default_files:
        default_config, _ = get_config_from_json(default_files)
        config = merge_dicts(default_config, config)

    config.hf_cache_dir = configure_hf_cache(config, logger=config_logger)

    if printing:
        config_logger.info(" THE Configuration of your experiment ..")
        pprint(config)
        # making sure that you have provided the exp_name.
        try:
            config_logger.info(" *************************************** ")
            config_logger.info("The experiment name is {}".format(config.exp_name))
            config_logger.info(" *************************************** ")
        except AttributeError:
            config_logger.info("ERROR!!..Please provide the exp_name in json file..")
            exit(-1)

    # create some important directories to be used for that experiment.
    # config.summary_dir = os.path.join("experiments", config.exp_name, "summaries/")
    # config.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoints/")
    # config.out_dir = os.path.join("experiments", config.exp_name, "out/")
    # config.log_dir = os.path.join("experiments", config.exp_name, "logs/")
    # create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir])
    # logging.shutdown()
    # importlib.reload(logging)

    return config


def process_config(json_file, printing = True):
    """
    Get the json file
    Processing it with EasyDict to be accessible as attributes
    then editing the path of the experiments folder
    creating some important directories in the experiment folder
    Then setup the logging in the whole program
    Then return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    config_logger = logging.getLogger("Config")
    config_logger.setLevel(logging.INFO)

    if printing:
        config_logger.info(json_file)
    config, _ = get_config_from_json(json_file)
    config.hf_cache_dir = configure_hf_cache(config, logger=config_logger)
    if printing:
        config_logger.info(" THE Configuration of your experiment ..")
        pprint(config)
        # making sure that you have provided the exp_name.
        try:
            config_logger.info(" *************************************** ")
            config_logger.info("The experiment name is {}".format(config.exp_name))
            config_logger.info(" *************************************** ")
        except AttributeError:
            config_logger.info("ERROR!!..Please provide the exp_name in json file..")
            exit(-1)

    # create some important directories to be used for that experiment.
    # config.summary_dir = os.path.join("experiments", config.exp_name, "summaries/")
    # config.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoints/")
    # config.out_dir = os.path.join("experiments", config.exp_name, "out/")
    # config.log_dir = os.path.join("experiments", config.exp_name, "logs/")
    # create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir])
    # logging.shutdown()
    # importlib.reload(logging)

    return config
