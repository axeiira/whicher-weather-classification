import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class SimpleTorchDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir : str, aug : list = []) -> None:
        self.dataset : list[tuple[str, np.ndarray]] = []
        self.root_dir = root_dir
        
        # self.__add_dataset__("Ac",      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # self.__add_dataset__("As",      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # self.__add_dataset__("Cb",      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        # self.__add_dataset__("Cc",      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        # self.__add_dataset__("Ci",      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        # self.__add_dataset__("Cs",      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        # self.__add_dataset__("Ct",      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        # self.__add_dataset__("Cu",      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        # self.__add_dataset__("Ns",      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        # self.__add_dataset__("Sc",      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        # self.__add_dataset__("St",      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        # self.__add_dataset__("cats",      [1, 0, 0])
        # self.__add_dataset__("dogs",      [0, 1, 0])
        # self.__add_dataset__("snakes",    [0, 0, 1])

        # self.__add_dataset__("Altocumulus",      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # self.__add_dataset__("Altostratus",      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        # self.__add_dataset__("Cirroculumulus",   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        # self.__add_dataset__("Cirrostratus",     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        # self.__add_dataset__("Cirrus",           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        # self.__add_dataset__("Cumulonimbus",     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        # self.__add_dataset__("Cumulus",          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        # self.__add_dataset__("Nimbostratus",     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        # self.__add_dataset__("Stratocumulus",    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        # self.__add_dataset__("Stratus",          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        self.__add_dataset__("1_cumulus",               [1, 0, 0, 0, 0, 0, 0])
        self.__add_dataset__("2_altocumulus",           [0, 1, 0, 0, 0, 0, 0])
        self.__add_dataset__("3_cirrus",                [0, 0, 1, 0, 0, 0, 0])
        self.__add_dataset__("4_clearsky",              [0, 0, 0, 1, 0, 0, 0])
        self.__add_dataset__("5_stratocumulus",         [0, 0, 0, 0, 1, 0, 0])
        self.__add_dataset__("6_cumulonimbus",          [0, 0, 0, 0, 0, 1, 0])
        self.__add_dataset__("7_mixed",                 [0, 0, 0, 0, 0, 0, 1])

        post_processing = [
            transforms.CenterCrop((128, 128)),
            transforms.ToTensor()
        ]

        self.augmentation = transforms.Compose(
            [transforms.Resize((177, 177))] +   # List Concatination
            aug                             +   # List Concatination
            post_processing
        )
    
    def __add_dataset__(self, dir_name : str, class_label : list[int]) -> None:
        full_path = os.path.join(self.root_dir, dir_name)
        label     = np.array(class_label)
        added = 0
        for fname in os.listdir(full_path):
            fpath = os.path.join(full_path, fname)
            fpath = os.path.abspath(fpath)
            self.dataset.append(
                (fpath, label)
            )
            added += 1
            if added >= 200:
                break

    # return the size of the dataset
    def __len__(self) -> int:
        return len(self.dataset)

    # grab one item form the dataset
    def __getitem__(self, index: int):
        fpath, label = self.dataset[index]

        # load image into numpy RGB numpy array in pytorch format
        image = Image.open(fpath).convert('RGB')
        image = self.augmentation(image)

        # minmax norm the image
        image = (image - image.min()) / (image.max() - image.min())
        label = torch.Tensor(label)

        return image, label
