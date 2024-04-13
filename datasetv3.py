import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch

import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from torchvision.datasets import MNIST
import torch.nn.functional as F
import torch.nn as nn
from numpy import bincount

############################################
### Alzheimer's Dataset Object Creation
############################################
class AlzheimersDataset(torch.utils.data.Dataset):
    def __init__(self):
      self.df = None
      self.transforms= None

    def set_df(self, df, transforms):
      self.df = df
      self.transforms = transforms

    def test_print(self):
      print(self.df.head())

    def __getitem__(self, index):
      img=cv2.imread(self.df.image[index])
      img=cv2.resize(img,(224,224))
      img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      label = self.df.label[index]
      # plt.imshow(img)
      # print(type(img), type(label))
      outimg = self.transforms(img)
      return outimg, label

    def __len__(self):
      return len(self.df)
        # raise NotImplementedError

############################################
### Balanced Sampler Function
############################################
def ws(df_col):
  counts = bincount(df_col)
  labels_weights = 1. / counts
  list(zip(range(4), counts))
  weights = labels_weights[df_col]
  ws = WeightedRandomSampler(weights, len(weights), replacement=True)
  return ws

def datasetv4():
    torch.manual_seed(42)

    ############################################
    ### Convert images into a dataframe
    ############################################
    images = []
    labels = []
    for subfolder in tqdm(os.listdir('Dataset')):
        subfolder_path = os.path.join('Dataset', subfolder)
        for image_filename in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_filename)
            images.append(image_path)
            labels.append(subfolder)
    df = pd.DataFrame({'image': images, 'label': labels})

    ############################################
    ### Label Mapping Conversion For WS usage
    ############################################
    di = {'Non_Demented':0, 'Very_Mild_Demented':1, 'Mild_Demented':2, 'Moderate_Demented':3}
    di_list = ['Non_Demented','Very_Mild_Demented', 'Mild_Demented', 'Moderate_Demented']

    df = df.replace({"label": di})

    transform_data = Compose([ToTensor(),
                            Resize(28),
                            Normalize((0.1307,), (0.3081,))])

    ############################################
    ### Test and training set creation
    ############################################

    df = df.sample(frac=1).reset_index(drop=True)
    
    train_size = int(0.8 * len(df))
    test_size = len(df) - train_size

    az_dataset_train = AlzheimersDataset()
    az_dataset_train.set_df(df[:train_size], transform_data)
    # az_dataset_train.test_print()

    df_test = df[train_size:]
    df_test.reset_index(drop=True, inplace=True)
    az_dataset_test = AlzheimersDataset()
    az_dataset_test.set_df(df_test, transform_data)
    # az_dataset_test.test_print()

    ws_train = ws(df[:train_size]['label'])
    ws_test = ws(df[train_size:]['label'])

    batch_size = 128
    az_dataloader_train = DataLoader(az_dataset_train, sampler=ws_train, batch_size=batch_size)
    az_dataloader_test = DataLoader(az_dataset_test, sampler=ws_test, batch_size=batch_size)
    # print(len(az_dataset_train), len(az_dataloader_train))
    # print(len(az_dataset_test), len(az_dataloader_test))
    

    return az_dataloader_train, az_dataloader_test, az_dataset_train, az_dataset_test