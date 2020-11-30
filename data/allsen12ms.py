import torch
import os
import pandas as pd
import h5py
import numpy as np
from skimage.exposure import equalize_hist
from skimage.transform import resize

s1bands = ["S1VV", "S1VH"]
s2bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11", "S2B12"]
bands = s1bands + s2bands

H5URL = "https://syncandshare.lrz.de/dl/fiDJwH3ZgzcoDts3srTT8XaA/sen12ms.h5"
CSVURL = "https://syncandshare.lrz.de/dl/fiHr4oDKXzPSPYnPRWNxAqnk/sen12ms.csv"
CSVSIZE = 47302099
H5SIZE = 115351475848

allregions = [143, 131, 77, 114, 41, 137, 127, 147, 44, 119, 133, 93, 72,
              53, 82, 66, 142, 86, 20, 7, 64, 26, 15, 45, 58, 128,
              57, 29, 129, 97, 4, 121, 132, 112, 25, 116, 55, 36, 105,
              140, 75, 104, 68, 124, 76, 71, 139, 107, 125, 87, 37, 6,
              84, 39, 79, 94, 35, 31, 40, 28, 80, 19, 59, 69, 148,
              109, 62, 106, 9, 83, 88, 113, 138, 43, 24, 61, 33, 63, 56, 52, 11, 85, 117, 89, 136, 115, 27, 42,
              78, 102, 47, 120, 100, 3, 30, 14, 1, 149, 144, 146,
              118, 126, 122, 145, 108, 8, 65, 101, 81, 134, 123, 103, 135,
              110, 95, 130, 49, 91, 17, 22, 32, 73, 90, 141, 21
              ]

trainregions = [57, 27, 77, 94, 61, 3, 142, 43, 79, 14, 39, 100, 56, 53, 147, 4, 15, 58, 112, 44, 124, 59, 114, 113, 71,
                125, 127, 146, 117, 33, 80, 11, 47, 9, 6, 29, 20, 35, 69, 24, 131, 19, 68, 104, 41, 66, 86, 75, 105,
                137, 120, 28, 143, 25, 129, 37, 93, 116, 45, 84, 133, 121, 62, 31, 52, 115, 132, 136, 82, 102, 7, 97,
                87, 149, 144]

valregions = [83, 64, 30, 138, 63, 128, 36, 85, 139, 140, 109, 40, 72, 78, 88, 42, 55, 26, 89, 119, 1, 106, 148, 107,
              76]

holdout_regions = [118, 126, 122, 145, 108, 8, 65, 101, 81, 134, 123, 103, 135,
                   110, 95, 130, 49, 91, 17, 22, 32, 73, 90, 141, 21]

# IGBP classes
IGBP_classes = [
    "Evergreen Needleleaf Forests",
    "Evergreen Broadleaf Forests",
    "Deciduous Needleleaf Forests",
    "Deciduous Broadleaf Forests",
    "Mixed Forests",
    "Closed (Dense) Shrublands",
    "Open (Sparse) Shrublands",
    "Woody Savannas",
    "Savannas",
    "Grasslands",
    "Permanent Wetlands",
    "Croplands",
    "Urban and Built-Up Lands",
    "Cropland Natural Vegetation Mosaics",
    "Permanent Snow and Ice",
    "Barren",
    "Water Bodies"
]

# simplified IGBP classes (DFC2020) Schmitt et al. 2020, Yokoya et al. 2020
IGBP_simplified_classes = [
    "Forests",
    "Shrubland",
    "Savanna",
    "Grassland",
    "Wetlands",
    "Croplands",
    "Urban Build-up",
    "Snow Ice",
    "Barren",
    "Water"
]

IGBP_simplified_class_mapping = [
    0,  # Evergreen Needleleaf Forests
    0,  # Evergreen Broadleaf Forests
    0,  # Deciduous Needleleaf Forests
    0,  # Deciduous Broadleaf Forests
    0,  # Mixed Forests
    1,  # Closed (Dense) Shrublands
    1,  # Open (Sparse) Shrublands
    2,  # Woody Savannas
    2,  # Savannas
    3,  # Grasslands
    4,  # Permanent Wetlands
    5,  # Croplands
    6,  # Urban and Built-Up Lands
    5,  # Cropland Natural Vegetation Mosaics
    7,  # Permanent Snow and Ice
    8,  # Barren
    9,  # Water Bodies
]

def classification_transform():
    def transform(s1, s2, label):
        s2 = s2 * 1e-4
        s1 = s1 * 1e-2

        input = np.vstack([s1, s2])

        igbp_label = np.bincount(label.reshape(-1)).argmax() - 1
        target = IGBP_simplified_class_mapping[igbp_label]

        if np.random.rand() < 0.5:
            input = input[:, ::-1, :]

        # horizontal flip
        if np.random.rand() < 0.5:
            input = input[:, :, ::-1]

        # rotate
        n_rotations = np.random.choice([0, 1, 2, 3])
        input = np.rot90(input, k=n_rotations, axes=(1, 2)).copy()

        if np.isnan(input).any():
            print("found nan in input! replacing with 0")
            input = np.nan_to_num(input)
        assert not np.isnan(target).any()

        rgb_band_idxs = np.array([bands.index("S2B4"),bands.index("S2B3"),bands.index("S2B2")])
        rgb = input[rgb_band_idxs]
        rgb = equalize_hist(rgb)

        rgb = resize(rgb, output_shape=(3,250,250))

        return torch.from_numpy(rgb).float(), target

    return transform

class AllSen12MSDataset(torch.utils.data.Dataset):
    def __init__(self, root, fold, classes=None, seasons=None, split_by_region=True):
        super(AllSen12MSDataset, self).__init__()

        self.transform = classification_transform()

        self.h5file_path = os.path.join(root, "sen12ms.h5")
        index_file = os.path.join(root, "sen12ms.csv")
        self.paths = pd.read_csv(index_file, index_col=0)

        if split_by_region:
            if fold == "train":
                regions = trainregions
            elif fold == "val":
                regions = valregions
            elif fold == "test":
                regions = holdout_regions
            elif fold == "all":
                regions = holdout_regions + valregions + trainregions
            else:
                raise AttributeError("one of meta_train, meta_val, meta_test must be true or "
                                     "fold must be in 'train','val','test'")

            mask = self.paths.region.isin(regions)
            print(f"fold {fold} specified. splitting by regions. Keeping {mask.sum()} of {len(mask)} tiles")
            self.paths = self.paths.loc[mask]
        else:
            rands = np.random.RandomState(0).rand(len(self.paths))
            if fold == "train":
                mask = (rands < 0.75)
            elif fold == "val":
                mask = (rands > 0.75) & (rands < 0.90)
            elif fold == "test":
                mask = (rands > 0.90)
            print(f"fold {fold} specified. random splitting. Keeping {mask.sum()} of {len(mask)} tiles")
            self.paths = self.paths.loc[mask]
        if classes is not None:
            mask = self.paths.maxclass.isin(classes)
            print(f"classes {classes} specified. Keeping {mask.sum()} of {len(mask)} tiles")
            self.paths = self.paths.loc[mask]
        if seasons is not None:
            mask = self.paths.season.isin(seasons)
            print(f"seasons {seasons} specified. Keeping {mask.sum()} of {len(mask)} tiles")
            self.paths = self.paths.loc[mask]

        # shuffle the tiles once
        self.paths = self.paths.sample(frac=1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths.iloc[index]

        with h5py.File(self.h5file_path, 'r') as data:
            s2 = data[path.h5path + "/s2"][()]
            s1 = data[path.h5path + "/s1"][()]
            label = data[path.h5path + "/lc"][()]

        image, target = self.transform(s1, s2, label)

        return image, target, index # path.h5path

def return_sen12ms_loader(args):
    ds = AllSen12MSDataset(args.sen12ms_path, "train")
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)