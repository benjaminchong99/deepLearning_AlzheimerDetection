import os
import pandas as pd
from tqdm import tqdm
import torch
import cv2
import ad_labels
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
import matplotlib.pyplot as plt


class AlzheimersDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transforms=None,
    ):
        self.df = df
        self.transforms = transforms

    # def set_df(self, df, transforms):
    #     self.df = df
    #     self.transforms = transforms

    def test_print(self):
        print(self.df.head())

    def __getitem__(self, index):
        """Returns tuple(img, label)"""
        img = cv2.imread(self.df.image[index])
        img = cv2.resize(img, (299, 299))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.df.label[index]
        # plt.imshow(img)
        # print(type(img), type(label))
        outimg = self.transforms(img)
        return outimg, label

    def __len__(self):
        return len(self.df)


def create_dataset():
    """
    Helper function to convert images to dataframe and initialise custom dataset
    Returns: AlzheimersDataset instance
    """

    # convert images into a dataframe
    transforms_data = Compose(
        [ToTensor(), Resize(299), Normalize((0.1307,), (0.3081,))]
    )
    images = []
    labels = []
    for subfolder in tqdm(os.listdir("./Dataset")):
        subfolder_path = os.path.join("./Dataset", subfolder)
        for image_filename in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_filename)
            images.append(image_path)
            labels.append(subfolder)
    df = pd.DataFrame({"image": images, "label": labels})
    df = df.replace({"label": ad_labels.LABELS_MAP})

    df = df.sample(frac=1).reset_index(drop=True)

    return AlzheimersDataset(df, transforms=transforms_data)
