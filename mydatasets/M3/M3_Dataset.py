from datasets import load_dataset
import copy
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from torch import Tensor
from typing import Tuple
from sklearn.model_selection import train_test_split


class M3_Dataset():

    def __init__(self, config, mode, **kwargs):
        self.config = config

        self.dataset = load_dataset("arsaporta/symile-m3", "symile-m3-2-xs", cache_dir=self.config.dataset.data_roots, streaming=True)
        from huggingface_hub import snapshot_download

        local_dir = "./symile-m3-5-xs"  # where to save

        # download parquet files
        snapshot_download(
            repo_id="arsaporta/symile-m3",
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=["symile-m3-5-xs/*"]  # which configuration to download
        )

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224,antialias=True),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224),antialias=True),
                # transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.this_stored_data = copy.deepcopy(self.dataset)
        # print("somehow we need to split the train")
        if mode!="test":
            self.split_dataset(mode)
        else:
            del self.this_stored_data


    def __len__(self):
        return len(self.this_data)

    def split_dataset(self, mode):
        # labels = np.array(self.samples)[np.array(self.indices)][:,1].astype(int)

        # indices = np.array([i for i in self.this_data]).astype(int)
        indices = {"_".join(self.this_data[i]["filename"].split("_")[1:3]): self.this_data[i]["label"]
                   for i in self.this_data}
        ids = list(indices.keys())
        labels = np.array([indices[i] for i in indices]).astype(int)
        # labels = np.array([self.this_data[i]["label"] for i in self.this_data]).astype(int)

        X_train, X_val, y_train, y_val = train_test_split(ids,
                                                          labels,
                                                          test_size=self.config.dataset.get("val_split_rate",0.1),
                                                          random_state=self.config.training_params.seed,
                                                          stratify=labels)
        if mode=="train":
            split_data = {}
            for c, id in enumerate(X_train):
                for i in self.this_data:
                    if id in self.this_data[i]["filename"] and self.this_data[i]["filename"] not in split_data:
                        split_data[self.this_data[i]["filename"]] = self.this_data[i]

        elif mode =="val":
            split_data = {}
            for c, id in enumerate(X_val):
                for i in self.this_data:
                    if id in self.this_data[i]["filename"] and self.this_data[i]["filename"] not in split_data:
                        split_data[self.this_data[i]["filename"]] = self.this_data[i]

        split_data = {c: split_data[id] for c, id in enumerate(split_data)}
        self.this_data = split_data

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:

        label = self.this_data[idx]["label"]

        frames = self.get_frames(idx)
        audio = self.get_audio(idx)
        spectrogram = self.get_spectrogram(idx)

        if self.mode=="test":
            random_idx = random.randint(0, len(self.this_data)-1)
            sh_frames = self.get_frames(random_idx)
            sh_spectrogram = self.get_audio(random_idx)

            return {"data": {0:spectrogram, 1:frames, "0_random_indistr": sh_spectrogram, "1_random_indistr": sh_frames}, "label": label}

        return {"data":{0:spectrogram,1:frames, 2:audio},"label": label}


class M3_Dataloader():

    def __init__(self, config):

        self.config = config

        dataset_train, dataset_val, dataset_test = self._get_datasets()

        g = torch.Generator()
        g.manual_seed(0)

        num_cores = len(os.sched_getaffinity(0))-1

        print("Available cores {}".format(len(os.sched_getaffinity(0))))
        print("We are changing dataloader workers to num of cores {}".format(num_cores))


        self.train_loader = torch.utils.data.DataLoader(dataset_train,
                                                        batch_size=self.config.training_params.batch_size,
                                                        num_workers=num_cores,
                                                        shuffle=True,
                                                        pin_memory=self.config.training_params.pin_memory,
                                                        generator=g,
                                                        worker_init_fn=lambda worker_id: np.random.seed(15 + worker_id))
        self.valid_loader = torch.utils.data.DataLoader(dataset_val,
                                                        batch_size=self.config.training_params.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=num_cores,
                                                        pin_memory=self.config.training_params.pin_memory)
        self.test_loader = torch.utils.data.DataLoader(dataset_test,
                                                       batch_size=self.config.training_params.test_batch_size,
                                                       shuffle=False,
                                                       num_workers=num_cores,
                                                       pin_memory=self.config.training_params.pin_memory)

    def _get_datasets(self):
        train_dataset = M3_Dataset(config=self.config, mode="train")
        val_dataset = M3_Dataset(config=self.config, mode="val")
        test_dataset = M3_Dataset(config=self.config, mode="test")

        return train_dataset, val_dataset, test_dataset

